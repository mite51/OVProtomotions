"""Body-velocity helper for the SMPL articulation.

This module exists to work around an ovphysx-vs-trainer parity gap on
``ARTICULATION_LINK_VELOCITY``. The trainer (IsaacLab/IsaacGym) reports
per-link velocities at the body's **center of mass** in the simulation
world frame. ovphysx's per-body link-velocity binding disagrees in
non-trivial ways for SMPL on the same
``(root_*, dof_pos, dof_vel, body_pos, body_rot)`` state — for the
captured 0-th frame, by tick 1 the deepest links (L_Hand / L_Wrist /
the toes) drift to ~12 m/s and ~32 rad/s, which throws the obs /
policy / PD pipeline far outside the trainer's distribution.

Rather than reverse-engineering ovphysx's internal Jacobian, PPP
sidesteps the link-velocity binding entirely:

1. Read ``body_pos`` / ``body_rot`` from ``ARTICULATION_LINK_POSE``
   before and after each policy tick (or each substep, for the
   real-time loop). These are at the **link's actor frame** (= the
   joint-anchor frame for non-root SMPL bodies, see
   ``smpl_humanoid.usda``) and round-trip through ovphysx at fp32.
2. Finite-difference link-origin velocities and angular velocities
   over the elapsed dt — see :func:`finite_difference_velocities`.
3. Apply the **COM correction**
   ``v_at_COM = v_at_L + omega x (R_body . com_offset_local)`` to
   convert each body's link-origin velocity into its COM velocity,
   which is what IsaacLab's ``body_lin_vel_w`` / ``body_ang_vel_w``
   return — see :func:`apply_com_correction`. Angular velocity is
   invariant under reference-point shifts on a rigid body, so only
   linear velocities need this conversion.

The COM offset table ``SMPL_COM_OFFSETS_LOCAL`` is derived directly
from ``data/assets/mjcf/smpl_humanoid.xml``: each SMPL body has a
single, uniform-density geom (a box or capsule), so the body's COM
in its own local frame is the geom centroid (box ``pos`` for boxes,
midpoint of ``fromto`` for capsules). The same MJCF is what the
USDA was generated from, so PhysX's auto-computed inertia produces
the same COMs.

Numerical accuracy: FD over the elapsed dt approximates the
**average** velocity over ``[t-dt, t]`` rather than the
**instantaneous** velocity at ``t``; the two differ by
``0.5 * a * dt`` with ``a`` the body acceleration. For SMPL with
``dt_physics = 1/120 s`` and a-tolerable accelerations the error
ceiling is well below the policy's training noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


# SMPL center-of-mass offsets in each body's local link frame, in the
# same body order as ``SMPL_BODY_NAMES``. Derived from
# ``protomotions/data/assets/mjcf/smpl_humanoid.xml`` — for boxes the
# offset is the geom ``pos``; for capsules it's the midpoint of the
# ``fromto`` segment. Each SMPL body has exactly one geom of uniform
# density, so the centroid coincides with the COM. PhysX
# auto-recovers the same COMs from the USDA at load time.
SMPL_COM_OFFSETS_LOCAL: np.ndarray = np.array(
    [
        [-0.005500,  0.000000, -0.012100],  # 0  Pelvis     box
        [-0.002250,  0.017150, -0.187600],  # 1  L_Hip      capsule
        [-0.021850, -0.006800, -0.199000],  # 2  L_Knee     capsule
        [ 0.024200,  0.023300, -0.023900],  # 3  L_Ankle    box
        [ 0.024800, -0.003000,  0.005500],  # 4  L_Toe      box
        [-0.004450, -0.019150, -0.191300],  # 5  R_Hip      capsule
        [-0.021150,  0.007900, -0.199200],  # 6  R_Knee     capsule
        [ 0.025600, -0.021200, -0.017400],  # 7  R_Ankle    box
        [ 0.022700,  0.004200,  0.004500],  # 8  R_Toe      box
        [ 0.000550,  0.002750,  0.067550],  # 9  Torso      capsule
        [ 0.012700,  0.000750,  0.026450],  # 10 Spine      capsule
        [-0.019250, -0.000950,  0.075750],  # 11 Chest      capsule
        [ 0.025700,  0.002550,  0.032500],  # 12 Neck       capsule
        [-0.011600, -0.004200,  0.087600],  # 13 Head       box
        [-0.004450,  0.045500,  0.015250],  # 14 L_Thorax   capsule
        [-0.013750,  0.129800, -0.006400],  # 15 L_Shoulder capsule
        [-0.000550,  0.124600,  0.004500],  # 16 L_Elbow    capsule
        [-0.007500,  0.042000, -0.004050],  # 17 L_Wrist    capsule
        [-0.005800,  0.049300,  0.001000],  # 18 L_Hand     box
        [-0.004550, -0.048000,  0.016250],  # 19 R_Thorax   capsule
        [-0.010700, -0.126850, -0.006700],  # 20 R_Shoulder capsule
        [-0.002750, -0.127650,  0.003900],  # 21 R_Elbow    capsule
        [-0.005200, -0.042300, -0.003050],  # 22 R_Wrist    capsule
        [-0.007900, -0.046200, -0.000900],  # 23 R_Hand     box
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RobotChain:
    """Per-body COM offset table for an SMPL-like articulation.

    Holds the (L, 3) COM offsets in each body's local link frame.
    Constructed once at startup; immutable thereafter.

    The COM table is the only piece of robot-specific data the FD+COM
    velocity pipeline needs (see :func:`apply_com_correction`). The
    chain Jacobian (joint-axis decomposition) is *not* used by that
    path — see the module docstring for why we finite-difference link
    poses instead.
    """

    com_offset_local: np.ndarray   # (L, 3) float32, body-local frame
    num_bodies: int

    @classmethod
    def for_smpl(cls) -> "RobotChain":
        """Built-in SMPL chain (24 bodies). See module docstring."""
        return cls(
            com_offset_local=SMPL_COM_OFFSETS_LOCAL.copy(),
            num_bodies=SMPL_COM_OFFSETS_LOCAL.shape[0],
        )

    @classmethod
    def from_body_names(cls, body_names: Sequence[str]) -> "RobotChain":
        """Build a chain matched to a specific body-name ordering.

        ``body_names`` should match SMPL's 24-body ordering (the
        canonical one in :data:`ppp.config_loader.SMPL_BODY_NAMES`);
        non-SMPL morphologies aren't currently supported (raises
        ``ValueError``). The argument exists so callers can plumb the
        ``ResolvedConfig.body_names`` they already have without an
        extra import.
        """
        from .config_loader import SMPL_BODY_NAMES
        if list(body_names) != list(SMPL_BODY_NAMES):
            raise ValueError(
                "RobotChain currently only supports the SMPL 24-body "
                "tree. Got body_names with "
                f"{len(body_names)} bodies; SMPL expects 24."
            )
        return cls.for_smpl()


# ----------------------------------------------------------------------
# Velocity helpers
# ----------------------------------------------------------------------
def quat_xyzw_to_mat(quats_xyzw: np.ndarray) -> np.ndarray:
    """Vectorised XYZW-quaternion -> rotation matrix.

    Returns ``(..., 3, 3)``. Quaternions are assumed unit (PhysX
    normalises on read, so this holds at fp32 within ovphysx's
    tolerance). No scipy dependency — formula expanded inline so the
    helper is import-cheap and unit-testable in isolation.
    """
    q = np.asarray(quats_xyzw, dtype=np.float32)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    out[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[..., 0, 1] = 2.0 * (xy - wz)
    out[..., 0, 2] = 2.0 * (xz + wy)
    out[..., 1, 0] = 2.0 * (xy + wz)
    out[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[..., 1, 2] = 2.0 * (yz - wx)
    out[..., 2, 0] = 2.0 * (xz - wy)
    out[..., 2, 1] = 2.0 * (yz + wx)
    out[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


def quat_fd_omega(
    q_prev_xyzw: np.ndarray, q_curr_xyzw: np.ndarray, dt: float
) -> np.ndarray:
    """World-frame angular velocity from quaternion finite-difference.

    Computes ``omega = 2 * imag(q_curr (x) q_prev^-1) / dt``, with
    shortest-path correction (negate ``q_curr`` if the dot product is
    negative so the differential rotation is below 180 deg). Returns
    ``(..., 3)`` float32. Inputs are XYZW. ``dt`` must be positive.

    This is the standard small-angle FD formula and is exact in the
    limit ``dt -> 0``. Over a 1/120 s substep it differs from the
    instantaneous ``omega`` by ``0.5 * alpha * dt`` with ``alpha`` the
    body angular acceleration; for SMPL motion under PD control that's
    sub-mrad/s noise.
    """
    qp = np.asarray(q_prev_xyzw, dtype=np.float32)
    qc = np.asarray(q_curr_xyzw, dtype=np.float32).copy()

    # Shortest-path: ensure dot >= 0 so the differential rotation is
    # the minimal one. Without this, a quaternion sign flip across
    # the integration step would produce a 2*pi/dt spike.
    dot = (qp * qc).sum(axis=-1, keepdims=True)
    qc = np.where(dot < 0, -qc, qc)

    # qd = q_curr (x) conj(q_prev). With XYZW layout (vec, scalar):
    #   qd_w = wc*wp + xc*xp + yc*yp + zc*zp
    #   qd_xyz = vc*wp - vp*wc - cross(vp, vc)
    vp, wp = qp[..., :3], qp[..., 3:4]
    vc, wc = qc[..., :3], qc[..., 3:4]
    qd_xyz = vc * wp - vp * wc - np.cross(vp, vc)

    return (2.0 / float(dt) * qd_xyz).astype(np.float32)


def finite_difference_velocities(
    body_pos_prev: np.ndarray,
    body_rot_prev_xyzw: np.ndarray,
    body_pos_curr: np.ndarray,
    body_rot_curr_xyzw: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-body FD of link-origin velocities and angular velocities.

    Returns ``(body_lin_vel_at_link_origin, body_ang_vel)`` each
    ``(L, 3)`` float32 in world frame. Linear velocity is the simple
    secant ``(p_curr - p_prev) / dt``; angular velocity is the
    quaternion FD (see :func:`quat_fd_omega`). Both are averaged
    over ``[t - dt, t]`` rather than instantaneous at ``t``.

    The linear value is at the **link origin** (where ``body_pos`` is
    measured); pass it through :func:`apply_com_correction` together
    with the body's rotation matrix and ``omega`` to get the COM
    velocity that matches the trainer's ``rigid_body_vel`` field.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}")
    lin = ((body_pos_curr - body_pos_prev) / float(dt)).astype(np.float32)
    ang = quat_fd_omega(body_rot_prev_xyzw, body_rot_curr_xyzw, dt)
    return lin, ang


def apply_com_correction(
    body_lin_vel_at_link: np.ndarray,
    body_ang_vel: np.ndarray,
    body_rot_xyzw: np.ndarray,
    com_offset_local: np.ndarray,
) -> np.ndarray:
    """Convert link-origin linear velocity to COM linear velocity.

    Adds the rigid-body transport term ``omega x (R . com_offset_local)``,
    matching IsaacLab's convention (``body_lin_vel_w`` is at COM,
    ``body_pos_w`` is at link origin — confirmed by the
    actor-frame-vs-COM-frame note in IsaacLab's articulation docs and
    bit-exact against captured ``debug_output.json``).

    All inputs are ``(L, 3)`` (or ``(L, 4)`` for quats) float32.
    ``omega`` is invariant under reference-point shift, so this helper
    only touches the linear component.
    """
    R = quat_xyzw_to_mat(body_rot_xyzw)              # (L, 3, 3)
    com_world = np.einsum("lij,lj->li", R, com_offset_local)  # (L, 3)
    return (
        body_lin_vel_at_link.astype(np.float32)
        + np.cross(body_ang_vel, com_world).astype(np.float32)
    )
