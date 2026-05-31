"""Thin wrapper around an ``ovphysx.PhysX`` instance for the SMPL articulation.

The wrapper:

- Loads the runtime USD (single articulation, single environment).
- Creates the small set of tensor bindings the policy loop needs.
- Exposes ``read_state`` / ``set_dof_targets`` / ``set_root_state`` /
  ``set_dof_positions`` / ``step`` returning numpy arrays in body- and
  DOF-order matching ProtoMotions' ``kinematic_info``.

Tensor types reference (``ovphysx.TensorType`` enum, ovphysx >= 0.3):

- ``TensorType.ARTICULATION_ROOT_POSE``        (A, 7)  pos.xyz + quat.xyzw
- ``TensorType.ARTICULATION_ROOT_VELOCITY``    (A, 6)  lin.xyz + ang.xyz
- ``TensorType.ARTICULATION_LINK_POSE``        (A, L, 7) — read only
- ``TensorType.ARTICULATION_LINK_VELOCITY``    (A, L, 6) — read only,
  used as a cold-start fallback for body velocities only.
- ``TensorType.ARTICULATION_DOF_POSITION``     (A, D)
- ``TensorType.ARTICULATION_DOF_VELOCITY``     (A, D)
- ``TensorType.ARTICULATION_DOF_POSITION_TARGET`` (A, D) — write target
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np


if TYPE_CHECKING:
    from .robot_chain import RobotChain


log = logging.getLogger(__name__)


@dataclass
class PhysxState:
    """One policy-tick snapshot of the SMPL articulation.

    All arrays are 2D ``(num_bodies, ...)`` or ``(num_dofs,)`` for a single env.
    Quaternions are XYZW (PhysX convention, matches w_last=True in ProtoMotions).
    """

    body_pos: np.ndarray         # (L, 3)
    body_rot: np.ndarray         # (L, 4) XYZW
    body_lin_vel: np.ndarray     # (L, 3)
    body_ang_vel: np.ndarray     # (L, 3)
    root_pos: np.ndarray         # (3,)
    root_rot: np.ndarray         # (4,) XYZW
    root_lin_vel: np.ndarray     # (3,)
    root_ang_vel: np.ndarray     # (3,)
    dof_pos: np.ndarray          # (D,)
    dof_vel: np.ndarray          # (D,)


class PhysxWorld:
    """Single-articulation, single-env PhysX wrapper.

    Args:
        usd_path: Absolute path to the runtime composed USD stage.
        articulation_root_path: USD prim path of the SMPL articulation root
            (typically ``/World/Robot/bodies/Pelvis``).
        num_bodies: Expected link count (24 for SMPL). Used to validate the
            articulation binding shape.
        num_dofs: Expected DOF count (69 for SMPL).
        dt_physics: Physics sub-step ``dt`` (e.g. 1/120 s).
        use_gpu: If True, request GPU mode; ovphysx falls back to CPU when
            no compatible GPU is available.
    """

    def __init__(
        self,
        usd_path: Path | str,
        articulation_root_path: str,
        num_bodies: int,
        num_dofs: int,
        dt_physics: float = 1.0 / 120.0,
        use_gpu: bool = False,
        expected_body_names: Optional[Sequence[str]] = None,
        expected_dof_names: Optional[Sequence[str]] = None,
        robot_chain: "Optional[RobotChain]" = None,
    ) -> None:
        self._usd_path = str(Path(usd_path).resolve())
        self._art_root = articulation_root_path
        self._num_bodies = num_bodies
        self._num_dofs = num_dofs
        self.dt_physics = float(dt_physics)
        self._sim_time = 0.0
        self._use_gpu = use_gpu
        self._expected_body_names = (
            list(expected_body_names) if expected_body_names else None
        )
        self._expected_dof_names = (
            list(expected_dof_names) if expected_dof_names else None
        )
        # Optional COM-offset table. When set, ``read_state`` computes
        # per-body world-frame velocities by finite-differencing the
        # link pose across the last physics substep (in :meth:`step`)
        # and then applying the COM correction
        # ``v_at_COM = v_at_L + omega x (R . com_offset)``. This matches
        # the trainer's IsaacLab body-velocity convention bit-for-bit
        # on captured frames; ovphysx's ``ARTICULATION_LINK_VELOCITY``
        # for non-root bodies disagrees in non-trivial ways and breaks
        # obs / policy / PD parity.
        self._robot_chain = robot_chain
        # FD bookkeeping. ``_prev_link_pose_for_fd`` is the link-pose
        # snapshot saved just before the *last* substep of the most
        # recent :meth:`step`; ``_fd_dt`` is the substep ``dt`` used
        # for that FD. Both are reset to ``None`` after a teleport
        # (since the FD window would straddle the teleport and produce
        # spurious velocities).
        self._prev_link_pose_for_fd: Optional[np.ndarray] = None
        self._fd_dt: float = 0.0

        # Reorder maps: ``physx_index = self._body_perm[expected_index]``.
        # If PhysX reports body/dof names in the same order as the expected
        # kinematic_info ordering, these are identity permutations.
        self._body_perm: Optional[np.ndarray] = None
        self._body_inv: Optional[np.ndarray] = None
        self._dof_perm: Optional[np.ndarray] = None
        self._dof_inv: Optional[np.ndarray] = None

        # Lazy-imported to keep ``ppp`` importable without ovphysx in CI/tests.
        import ovphysx

        self._ovphysx = ovphysx
        self._TT = ovphysx.TensorType
        device = "gpu" if use_gpu else "cpu"
        self._physx = ovphysx.PhysX(device=device)
        log.info(
            "ovphysx version: %s, device=%s, "
            "physics substep target = %.4f s (%.1f Hz).",
            getattr(ovphysx, "__version__", "?"),
            device,
            self.dt_physics,
            1.0 / self.dt_physics,
        )

        log.info("Loading runtime USD: %s", self._usd_path)
        self._usd_handle, _ = self._physx.add_usd(self._usd_path)

        # Build bindings. We use ``prim_paths=[art_root]`` for articulation
        # bindings so the binding deterministically targets *this* articulation.
        self._bind_root_pose = self._physx.create_tensor_binding(
            prim_paths=[self._art_root],
            tensor_type=self._TT.ARTICULATION_ROOT_POSE,
        )
        self._bind_root_vel = self._physx.create_tensor_binding(
            prim_paths=[self._art_root],
            tensor_type=self._TT.ARTICULATION_ROOT_VELOCITY,
        )
        self._bind_link_pose = self._physx.create_tensor_binding(
            prim_paths=[self._art_root],
            tensor_type=self._TT.ARTICULATION_LINK_POSE,
        )
        self._bind_link_vel = self._physx.create_tensor_binding(
            prim_paths=[self._art_root],
            tensor_type=self._TT.ARTICULATION_LINK_VELOCITY,
        )
        self._bind_dof_pos = self._physx.create_tensor_binding(
            prim_paths=[self._art_root],
            tensor_type=self._TT.ARTICULATION_DOF_POSITION,
        )
        self._bind_dof_vel = self._physx.create_tensor_binding(
            prim_paths=[self._art_root],
            tensor_type=self._TT.ARTICULATION_DOF_VELOCITY,
        )
        self._bind_dof_target = self._physx.create_tensor_binding(
            prim_paths=[self._art_root],
            tensor_type=self._TT.ARTICULATION_DOF_POSITION_TARGET,
        )

        # GPU warmup so first read does not stall the policy loop.
        if use_gpu:
            try:
                self._physx.warmup_gpu()
            except Exception as e:  # pragma: no cover - hardware dependent
                log.warning("warmup_gpu failed (falling back transparently): %s", e)

        # Validate shapes after a single zero-dt step to populate buffers.
        self._physx.step(0.0, 0.0)
        self._physx.wait_all()
        self._validate_shapes()
        self._build_reorder_maps()

        # Pre-allocate staging buffers (NumPy, contiguous, float32 / int32).
        self._buf_root_pose = np.zeros(self._bind_root_pose.shape, dtype=np.float32)
        self._buf_root_vel = np.zeros(self._bind_root_vel.shape, dtype=np.float32)
        self._buf_link_pose = np.zeros(self._bind_link_pose.shape, dtype=np.float32)
        self._buf_link_vel = np.zeros(self._bind_link_vel.shape, dtype=np.float32)
        self._buf_dof_pos = np.zeros(self._bind_dof_pos.shape, dtype=np.float32)
        self._buf_dof_vel = np.zeros(self._bind_dof_vel.shape, dtype=np.float32)
        self._buf_dof_target = np.zeros(self._bind_dof_target.shape, dtype=np.float32)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        for b in (
            getattr(self, "_bind_root_pose", None),
            getattr(self, "_bind_root_vel", None),
            getattr(self, "_bind_link_pose", None),
            getattr(self, "_bind_link_vel", None),
            getattr(self, "_bind_dof_pos", None),
            getattr(self, "_bind_dof_vel", None),
            getattr(self, "_bind_dof_target", None),
        ):
            if b is not None:
                try:
                    b.destroy()
                except Exception:
                    pass
        if getattr(self, "_physx", None) is not None:
            try:
                self._physx.release()
            except Exception:
                pass
            self._physx = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_shapes(self) -> None:
        link_shape = self._bind_link_pose.shape
        dof_shape = self._bind_dof_pos.shape
        if link_shape[0] != 1:
            raise RuntimeError(
                f"Expected 1 articulation at {self._art_root}, "
                f"got link_pose shape {link_shape}"
            )
        if link_shape[1] != self._num_bodies:
            raise RuntimeError(
                f"Articulation reports {link_shape[1]} links but "
                f"kinematic_info expects {self._num_bodies}. The link order "
                "in PhysX must match the USDA prim order (resolved_configs "
                "body_names). Check that smpl_humanoid.usda was loaded "
                "untouched."
            )
        if dof_shape[1] != self._num_dofs:
            raise RuntimeError(
                f"Articulation reports {dof_shape[1]} DOFs but "
                f"kinematic_info expects {self._num_dofs}."
            )
        log.info(
            "Articulation OK: %d links, %d DOFs (one env).",
            link_shape[1],
            dof_shape[1],
        )

    def _build_reorder_maps(self) -> None:
        """Resolve PhysX body/DOF order vs expected ``kinematic_info`` order.

        ``ovphysx.TensorBinding`` exposes ``body_names`` and ``dof_names``
        properties that describe the order PhysX uses for the bound
        articulation. If the order differs from the trained ``kinematic_info``
        ordering we build a permutation that read_state/set_dof_targets/etc.
        use to translate between the two.
        """
        try:
            physx_body_names = list(self._bind_link_pose.body_names)
            physx_dof_names = list(self._bind_dof_pos.dof_names)
        except Exception as e:  # pragma: no cover - older builds
            log.warning(
                "TensorBinding.body_names/dof_names unavailable (%s); "
                "assuming identity ordering.",
                e,
            )
            return

        if self._expected_body_names:
            if sorted(self._expected_body_names) != sorted(physx_body_names):
                raise RuntimeError(
                    "PhysX body names do not match kinematic_info.\n"
                    f"  expected: {self._expected_body_names}\n"
                    f"  got     : {physx_body_names}"
                )
            perm = [physx_body_names.index(n) for n in self._expected_body_names]
            self._body_perm = np.asarray(perm, dtype=np.int64)
            if not np.array_equal(self._body_perm, np.arange(len(perm))):
                log.warning(
                    "PhysX body order differs from kinematic_info; "
                    "applying permutation."
                )
                inv = np.empty_like(self._body_perm)
                inv[self._body_perm] = np.arange(self._body_perm.size)
                self._body_inv = inv
            else:
                self._body_perm = None  # identity -> skip work

        if self._expected_dof_names:
            if sorted(self._expected_dof_names) != sorted(physx_dof_names):
                raise RuntimeError(
                    "PhysX DOF names do not match kinematic_info.\n"
                    f"  expected: {self._expected_dof_names}\n"
                    f"  got     : {physx_dof_names}"
                )
            perm = [physx_dof_names.index(n) for n in self._expected_dof_names]
            self._dof_perm = np.asarray(perm, dtype=np.int64)
            if not np.array_equal(self._dof_perm, np.arange(len(perm))):
                log.warning(
                    "PhysX DOF order differs from kinematic_info; "
                    "applying permutation."
                )
                inv = np.empty_like(self._dof_perm)
                inv[self._dof_perm] = np.arange(self._dof_perm.size)
                self._dof_inv = inv
            else:
                self._dof_perm = None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read_state(self) -> PhysxState:
        """Read the full state of the articulation as a ``PhysxState``.

        Body velocities are derived by finite-differencing the link pose
        across the most recent physics substep and applying the COM
        correction ``v_at_COM = v_at_L + omega x (R . com_offset)`` —
        see :func:`ppp.robot_chain.apply_com_correction`. This matches
        the trainer's IsaacLab/IsaacGym body-velocity convention
        bit-for-bit; ovphysx's ``ARTICULATION_LINK_VELOCITY`` binding
        for non-root bodies does not, and using it breaks
        obs / policy / PD parity.

        On the very first read after a teleport (no FD window
        available), we fall back to the link-velocity binding so the
        first observation isn't NaN. After one ``step`` the FD path
        kicks in.
        """
        self._bind_root_pose.read(self._buf_root_pose)
        self._bind_root_vel.read(self._buf_root_vel)
        self._bind_link_pose.read(self._buf_link_pose)
        # link-velocity binding still read so the cold-start /
        # post-teleport fallback path works.
        need_link_vel_fallback = (
            self._robot_chain is None or self._prev_link_pose_for_fd is None
        )
        if need_link_vel_fallback:
            self._bind_link_vel.read(self._buf_link_vel)
        self._bind_dof_pos.read(self._buf_dof_pos)
        self._bind_dof_vel.read(self._buf_dof_vel)

        # Shapes: root (1,7); link (1,L,7); dof (1,D)
        root_pose = self._buf_root_pose[0]
        root_vel = self._buf_root_vel[0]
        link_pose = self._buf_link_pose[0]
        dof_pos = self._buf_dof_pos[0]
        dof_vel = self._buf_dof_vel[0]

        if self._body_perm is not None:
            link_pose = link_pose[self._body_perm]
        if self._dof_perm is not None:
            dof_pos = dof_pos[self._dof_perm]
            dof_vel = dof_vel[self._dof_perm]

        body_pos = link_pose[:, 0:3].copy()
        body_rot = link_pose[:, 3:7].copy()
        root_pos_arr = root_pose[0:3].copy()
        root_rot_arr = root_pose[3:7].copy()
        root_lin_vel = root_vel[0:3].copy()
        root_ang_vel = root_vel[3:6].copy()

        body_lin_vel, body_ang_vel = self._compute_body_velocities(
            body_pos=body_pos,
            body_rot=body_rot,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
        )

        return PhysxState(
            body_pos=body_pos,
            body_rot=body_rot,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
            root_pos=root_pos_arr,
            root_rot=root_rot_arr,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos.copy(),
            dof_vel=dof_vel.copy(),
        )

    def _compute_body_velocities(
        self,
        body_pos: np.ndarray,
        body_rot: np.ndarray,
        root_lin_vel: np.ndarray,
        root_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-body world-frame velocities (FD + COM correction)."""
        if self._robot_chain is not None and self._prev_link_pose_for_fd is not None:
            from .robot_chain import (
                apply_com_correction,
                finite_difference_velocities,
            )

            prev_pose = self._prev_link_pose_for_fd
            if self._body_perm is not None:
                prev_pose = prev_pose[self._body_perm]

            body_lin_vel_at_L, body_ang_vel = finite_difference_velocities(
                body_pos_prev=prev_pose[:, 0:3],
                body_rot_prev_xyzw=prev_pose[:, 3:7],
                body_pos_curr=body_pos,
                body_rot_curr_xyzw=body_rot,
                dt=self._fd_dt,
            )
            body_lin_vel = apply_com_correction(
                body_lin_vel_at_link=body_lin_vel_at_L,
                body_ang_vel=body_ang_vel,
                body_rot_xyzw=body_rot,
                com_offset_local=self._robot_chain.com_offset_local,
            )
            # The articulation root's own velocity binding is at COM
            # already (matches trainer convention) and round-trips at
            # fp32 through ``set_root_state``. Replace the FD-based
            # estimates for body 0 with the binding so teleport
            # diagnostics keep their fp32 round-trip and the chain
            # propagation upstream of the FD window doesn't bleed
            # into the root.
            body_lin_vel[0] = root_lin_vel
            body_ang_vel[0] = root_ang_vel
            return body_lin_vel, body_ang_vel

        # Cold-start / post-teleport fallback.
        link_vel = self._buf_link_vel[0]
        if self._body_perm is not None:
            link_vel = link_vel[self._body_perm]
        return link_vel[:, 0:3].copy(), link_vel[:, 3:6].copy()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def set_dof_targets(self, targets_69: Sequence[float] | np.ndarray) -> None:
        """Write position-control targets for all DOFs (units: radians).

        Input is in the trained ``kinematic_info`` DOF order; we permute to
        PhysX order on write.
        """
        arr = np.asarray(targets_69, dtype=np.float32).reshape(self._num_dofs)
        if self._dof_inv is not None:
            arr = arr[self._dof_inv]
        self._buf_dof_target[0] = arr
        self._bind_dof_target.write(self._buf_dof_target)

    def set_dof_positions(
        self,
        positions: Sequence[float] | np.ndarray,
        velocities: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Teleport joint positions (and optionally velocities) — used for reset."""
        pos_arr = np.asarray(positions, dtype=np.float32).reshape(self._num_dofs)
        if self._dof_inv is not None:
            pos_arr = pos_arr[self._dof_inv]
        self._buf_dof_pos[0] = pos_arr
        self._bind_dof_pos.write(self._buf_dof_pos)

        if velocities is None:
            self._buf_dof_vel.fill(0.0)
        else:
            vel_arr = np.asarray(velocities, dtype=np.float32).reshape(self._num_dofs)
            if self._dof_inv is not None:
                vel_arr = vel_arr[self._dof_inv]
            self._buf_dof_vel[0] = vel_arr
        self._bind_dof_vel.write(self._buf_dof_vel)
        # Teleport invalidates the FD window — the next ``read_state``
        # would otherwise differentiate the current pose against a
        # link pose from before the teleport, producing huge spurious
        # velocities. Force the fallback path until the next ``step``.
        self._prev_link_pose_for_fd = None
        self._fd_dt = 0.0

    def set_root_state(
        self,
        pos: Sequence[float] | np.ndarray,
        quat_xyzw: Sequence[float] | np.ndarray,
        lin_vel: Sequence[float] | np.ndarray | None = None,
        ang_vel: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Teleport the articulation root pose and (optionally) velocity."""
        pos = np.asarray(pos, dtype=np.float32).reshape(3)
        quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
        self._buf_root_pose[0, 0:3] = pos
        self._buf_root_pose[0, 3:7] = quat
        self._bind_root_pose.write(self._buf_root_pose)

        if lin_vel is None:
            self._buf_root_vel[0, 0:3] = 0.0
        else:
            self._buf_root_vel[0, 0:3] = np.asarray(lin_vel, dtype=np.float32).reshape(
                3
            )
        if ang_vel is None:
            self._buf_root_vel[0, 3:6] = 0.0
        else:
            self._buf_root_vel[0, 3:6] = np.asarray(ang_vel, dtype=np.float32).reshape(
                3
            )
        self._bind_root_vel.write(self._buf_root_vel)
        # See ``set_dof_positions``: a root teleport invalidates the
        # FD window for body velocities.
        self._prev_link_pose_for_fd = None
        self._fd_dt = 0.0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """Advance physics by exactly ``dt`` seconds of sim time.

        ``dt`` is the actual elapsed time we want the integrator to
        cover — same units as ``ovphysx.PhysX.step(dt, sim_time)``.
        Internally we split ``dt`` into ``ceil(dt / self.dt_physics)``
        equal substeps so the PhysX integrator never sees a step bigger
        than its trained substep size (``1 / physics_fps``, by default
        1/120 s). This is the right knob for variable-dt simulation:

        - ``dt = 1/30 s``  -> 4 substeps of 1/120 s  (matches training).
        - ``dt = 1/60 s``  -> 2 substeps of 1/120 s  (half-speed playback).
        - ``dt = 1/15 s``  -> 8 substeps of 1/120 s  (2x playback).
        - ``dt < 1/120 s`` -> 1 substep of dt.

        Small ``dt`` is fine. Very large ``dt`` would just queue up more
        substeps; callers should cap upstream if they want to bound the
        worst-case work per call.

        Side effect: snapshots the link pose just before the *last*
        substep so :meth:`read_state` can finite-difference body
        velocities over a single ``dt_physics`` window (matching the
        trainer's body-velocity convention via the COM correction in
        :mod:`ppp.robot_chain`). FD over a single substep keeps the
        avg-vs-instantaneous bias under ``0.5 * a * dt_physics`` —
        sub-mrad/s for typical SMPL motion.
        """
        if dt <= 0.0:
            return
        n = max(1, int(round(dt / self.dt_physics)))
        h = dt / n
        for k in range(n):
            if k == n - 1 and self._robot_chain is not None:
                # Snapshot link pose right before the final substep so
                # we have an FD window of exactly ``h`` seconds when
                # ``read_state`` runs after this call.
                self._bind_link_pose.read(self._buf_link_pose)
                self._prev_link_pose_for_fd = self._buf_link_pose[0].copy()
                self._fd_dt = float(h)
            self._physx.step(h, self._sim_time)
            self._sim_time += h
        self._physx.wait_all()

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def num_dofs(self) -> int:
        return self._num_dofs

    @property
    def link_pose_buffer(self) -> np.ndarray:
        """Latest cached link poses ``(L, 7)``. Filled by ``read_state``."""
        return self._buf_link_pose[0]

    @property
    def physx_body_names(self) -> List[str]:
        """Body names in PhysX's native order (may differ from kinematic_info)."""
        try:
            return list(self._bind_link_pose.body_names)
        except Exception:  # pragma: no cover
            return []

    @property
    def physx_dof_names(self) -> List[str]:
        """DOF names in PhysX's native order."""
        try:
            return list(self._bind_dof_pos.dof_names)
        except Exception:  # pragma: no cover
            return []
