"""Compose a runtime USD scene for the SMPL mimic inference app.

The scene references ProtoMotions' existing ``smpl_humanoid.usda`` (placed
under ``/World/Robot`` so all bodies live at ``/World/Robot/bodies/<name>``)
and ``checkerboard_ground.usda`` (under ``/World/Ground``, used purely as a
visual). It also adds:

- ``/World/physicsScene`` — a ``PhysicsScene`` prim with Z-down gravity
  and ``PhysxSceneAPI`` pinning the solver to ``TGS`` to match the
  IsaacGym/IsaacLab training setup (matches every ``ovphysx`` sample
  under ``ovphysx/samples/data/``, plus ProtoMotions' default
  ``IsaacGymPhysXParams.solver_type = 1``).
- ``/World/GroundCollider`` — a ``Plane`` prim with ``PhysicsCollisionAPI``
  + ``PhysxCollisionAPI`` applied (``axis = "Z"``, ``purpose = "guide"``,
  ``contactOffset = 0.001 m``). This is the canonical ovphysx static-ground
  pattern from ``ovphysx/samples/data/boxes_falling_on_groundplane.usda``;
  it gives us an invisible, infinite Z=0 collision plane without authoring
  any rigid body APIs. The 1 mm contact offset is tuned to roughly match
  the IsaacGym terrain mesh contact offset captured in OmniPVD
  (~0.00136 m); the PhysX default of 0.02 m would have feet "land"
  ~2 cm above the visual ground, which detunes the trained policy.
  The upstream ``checkerboard_ground.usda`` cannot be used as a collider
  in ovphysx because it applies ``PhysicsRigidBodyAPI`` on the root (with
  a triangle mesh child, which is invalid for a dynamic body) *and*
  explicitly sets ``physics:collisionEnabled = 0`` on the mesh. Disabling
  the rigid body via ``physics:rigidBodyEnabled = 0`` was tried first but
  doesn't activate descendant colliders in ovphysx — the character still
  falls through. Authoring a separate static plane bypasses the
  upstream's physics setup entirely.
- ``/World/Camera`` (the camera xformable) and ``/Render/Camera`` (the
  ``RenderProduct`` referencing it) — the layout ovrtx requires.

In addition, this composer authors a set of per-body / per-joint USDA
overrides that bake the runtime PhysX configuration ProtoMotions sets
in code (and that an OmniPVD capture confirmed the trained SMPL policy
was exposed to). Without these, ovphysx falls back to PhysX defaults
that differ visibly from the IsaacGym/IsaacLab setup the policy was
trained on:

- Per-body ``physxRigidBody`` caps: ``maxAngularVelocity = 1000`` deg/s
  (≈ 17.45 rad/s, matches the OmniPVD capture exactly), ``maxLinearVelocity
  = 1000`` m/s, ``maxDepenetrationVelocity = 1.0`` m/s,
  ``enableGyroscopicForces = 1``. These are *not* in
  ``smpl_humanoid.usda``; they're set at runtime by
  ``protomotions/simulator/{isaacgym,isaaclab}/...``.
- Per-joint drive caps: ``drive:rotX/Y/Z:physics:maxForce = 500`` N·m
  (the upstream USDA authors FLT_MAX, ProtoMotions overrides to 500 in
  ``robot_configs/smpl.py::effort_limit``) and ``physxJoint:maxJointVelocity
  = 100`` rad/s (``robot_configs/smpl.py::velocity_limit``). With FLT_MAX
  the PD drive can apply unbounded torque to chase action targets, far
  outside the training distribution.
- Articulation solver iteration counts on the Pelvis (the articulation
  root): ``physxArticulation:solverPositionIterationCount = 4``,
  ``physxArticulation:solverVelocityIterationCount = 4`` — matches the
  OmniPVD capture (``positionIterations = 4``, ``velocityIterations = 4``).

The runtime file is written as plain USDA (no USD Python dep required).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


_PROTOMOTIONS_ASSETS = (
    Path("C:/Git/ProtoMotions/protomotions/data/assets").resolve()
)
_SMPL_HUMANOID_USDA = _PROTOMOTIONS_ASSETS / "usd" / "smpl_humanoid.usda"
_CHECKERBOARD_GROUND_USDA = (
    _PROTOMOTIONS_ASSETS / "checkerboard" / "checkerboard_ground.usda"
)


# Paths inside the composed runtime stage.
ROBOT_PRIM = "/World/Robot"
ROBOT_BODIES_ROOT = "/World/Robot/bodies"
GROUND_PRIM = "/World/Ground"
CAMERA_PRIM = "/Render/Camera"
SMPL_BODY_MATERIAL_PRIM = "/World/PhysicsMaterials/SmplBodyMaterial"
# Per-body marker prims for the optional ``--draw-mimic-pose`` overlay
# (see ``write_runtime_scene(mimic_target_body_names=...)``). Each marker
# is a pure-visual ``Xform`` + child ``Sphere`` with no physics APIs, so
# ovphysx skips them entirely while ovrtx renders them as floating dots
# at the motion-reference body positions.
MIMIC_TARGETS_ROOT = "/World/MimicTargets"


# SMPL humanoid collider layout (matches the upstream
# ``protomotions/.../smpl_humanoid.usda``): 24 body Xforms, each with a
# single collider mesh under ``collisions/_geom_<idx>`` where ``idx`` is
# the body's position in this list (so Pelvis→_geom_0, R_Hand→_geom_23).
# Used to author per-collider physics material bindings in the runtime
# scene without relying on ovphysx implementing USD's ancestor-resolution
# for material bindings.
_SMPL_BODY_COLLIDER_GEOMS: Tuple[Tuple[str, int], ...] = (
    ("Pelvis", 0),
    ("L_Hip", 1), ("L_Knee", 2), ("L_Ankle", 3), ("L_Toe", 4),
    ("R_Hip", 5), ("R_Knee", 6), ("R_Ankle", 7), ("R_Toe", 8),
    ("Torso", 9), ("Spine", 10), ("Chest", 11), ("Neck", 12), ("Head", 13),
    ("L_Thorax", 14), ("L_Shoulder", 15), ("L_Elbow", 16),
    ("L_Wrist", 17), ("L_Hand", 18),
    ("R_Thorax", 19), ("R_Shoulder", 20), ("R_Elbow", 21),
    ("R_Wrist", 22), ("R_Hand", 23),
)


# SMPL humanoid joints (everything except the Pelvis, which is the
# articulation root and has no inbound joint). Order is the same as the
# upstream ``smpl_humanoid.usda`` ``def "joints"`` block.
_SMPL_JOINT_NAMES: Tuple[str, ...] = (
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
)


@dataclass(frozen=True)
class JointGains:
    """Per-joint physical PD gains and limits, one entry per SMPL joint.

    The trainer's ``robot.control.control_info`` is keyed by per-DOF
    name (``L_Hip_x``, ``L_Hip_y``, ``L_Hip_z``, ...). Group regexes
    set the same value across the three axes of any given joint, so
    we collapse to one ``stiffness`` / ``damping`` / ``effort`` /
    ``velocity`` per joint when authoring the USD overrides; if a
    future config splits per-axis, this dataclass would gain
    ``stiffness_x/y/z`` etc.

    ``drive_type`` is the USD drive token. ``"force"`` is the raw
    torque PD (gain in N·m/rad) and is what the trainer actually
    runs on IsaacLab — ``ImplicitActuatorCfg`` hands the gains
    straight to PhysX without overriding the USD's authored drive
    type, and the upstream ``smpl_humanoid.usda`` authors ``"force"``.
    ``"acceleration"`` is a mass-normalised mode where the gain is
    multiplied by each joint's effective inertia before integration
    — same number on the dial, very different effective torque on
    heavy joints (e.g. ~5× on a hip), and the policy will not balance
    under it because it was trained against force-mode response.
    """

    stiffness: float
    damping: float
    effort_limit: float
    velocity_limit: float
    drive_type: str = "force"


def per_joint_gains_from_config(
    dof_names: Sequence[str],
    pd_stiffness: Sequence[float],
    pd_damping: Sequence[float],
    pd_effort_limit: Sequence[float],
    pd_velocity_limit: Sequence[float],
    drive_type: str,
) -> "dict[str, JointGains]":
    """Collapse per-DOF arrays to per-joint :class:`JointGains`.

    Verifies that the three DOFs of each SMPL joint share the same
    gains (they always do for the SMPL group-regex tables — kp/kd
    depend on joint group, not axis). If that ever stops being true
    upstream, this raises so the next caller knows to widen
    :class:`JointGains` rather than silently dropping two of the three
    axis values.
    """
    by_joint: dict[str, JointGains] = {}
    for joint in _SMPL_JOINT_NAMES:
        try:
            ix = dof_names.index(f"{joint}_x")
            iy = dof_names.index(f"{joint}_y")
            iz = dof_names.index(f"{joint}_z")
        except ValueError as e:
            raise RuntimeError(
                f"DOF for joint {joint!r} not found in dof_names: {e}"
            ) from None
        kps = (pd_stiffness[ix], pd_stiffness[iy], pd_stiffness[iz])
        kds = (pd_damping[ix], pd_damping[iy], pd_damping[iz])
        effs = (pd_effort_limit[ix], pd_effort_limit[iy], pd_effort_limit[iz])
        vels = (
            pd_velocity_limit[ix],
            pd_velocity_limit[iy],
            pd_velocity_limit[iz],
        )
        if not (kps[0] == kps[1] == kps[2] and kds[0] == kds[1] == kds[2]):
            raise RuntimeError(
                f"Joint {joint!r} has per-axis kp/kd asymmetry "
                f"(kp={kps}, kd={kds}); JointGains needs widening."
            )
        by_joint[joint] = JointGains(
            stiffness=float(kps[0]),
            damping=float(kds[0]),
            effort_limit=float(max(effs)),
            velocity_limit=float(max(vels)),
            drive_type=drive_type,
        )
    return by_joint


# Per-body PhysX caps captured by OmniPVD against the IsaacGym SMPL
# motion-tracker training run, cross-checked against
# ``protomotions/robot_configs/smpl.py``. Authored in USDA so ovphysx
# picks them up directly at load time; no extra Python knob needed.
#
# Units: ``physxRigidBody:maxAngularVelocity`` is in **deg/s** per the
# OmniPhysics USD schema (1000 deg/s = 17.453293 rad/s, the value PVD
# captures on every SMPL link). ``maxLinearVelocity`` is in m/s,
# ``maxDepenetrationVelocity`` is in m/s. ``enableGyroscopicForces`` is
# off by default in PhysX but PVD shows ``eENABLE_GYROSCOPIC_FORCES`` set
# on every SMPL link, so we explicitly enable it here.
_BODY_MAX_ANGULAR_VELOCITY_DEG_PER_SEC = 1000.0
_BODY_MAX_LINEAR_VELOCITY_M_PER_SEC = 1000.0
_BODY_MAX_DEPENETRATION_VELOCITY_M_PER_SEC = 1.0

# Per-joint runtime caps. ProtoMotions sets these on every joint of the
# SMPL humanoid (see ``robot_configs/smpl.py::ControlInfo`` defaults
# applied across all joint groups: every group has ``effort_limit=500``
# and ``velocity_limit=100``). The upstream ``smpl_humanoid.usda``
# authors ``drive:rotX/Y/Z:physics:maxForce = FLT_MAX`` instead, which
# means PhysX will apply unbounded torque chasing PD targets — far
# outside the training distribution.
_JOINT_DRIVE_MAX_FORCE_NM = 500.0
_JOINT_MAX_VELOCITY_RAD_PER_SEC = 100.0

# Articulation solver iterations as captured by OmniPVD on the SMPL
# articulation (``positionIterations = 4``, ``velocityIterations = 4``).
# The IsaacGym/IsaacLab default for velocity iterations is 0/1 depending
# on path, but the actual PhysX scene exposed to the trained policy used
# 4/4 — pin to that.
_ARTICULATION_POSITION_ITERATIONS = 4
_ARTICULATION_VELOCITY_ITERATIONS = 4


def _body_rigid_body_caps_block(indent: str, *, is_articulation_root: bool) -> str:
    """USDA snippet authoring per-body ``physxRigidBody:*`` caps.

    Pelvis additionally gets ``physxArticulation:*`` solver-iteration
    overrides — it's the articulation root and the only place those
    attributes actually take effect.
    """
    lines = [
        f"{indent}float physxRigidBody:maxAngularVelocity = "
        f"{_BODY_MAX_ANGULAR_VELOCITY_DEG_PER_SEC}\n",
        f"{indent}float physxRigidBody:maxLinearVelocity = "
        f"{_BODY_MAX_LINEAR_VELOCITY_M_PER_SEC}\n",
        f"{indent}float physxRigidBody:maxDepenetrationVelocity = "
        f"{_BODY_MAX_DEPENETRATION_VELOCITY_M_PER_SEC}\n",
        f"{indent}bool physxRigidBody:enableGyroscopicForces = 1\n",
    ]
    if is_articulation_root:
        lines.append(
            f"{indent}int physxArticulation:solverPositionIterationCount = "
            f"{_ARTICULATION_POSITION_ITERATIONS}\n"
        )
        lines.append(
            f"{indent}int physxArticulation:solverVelocityIterationCount = "
            f"{_ARTICULATION_VELOCITY_ITERATIONS}\n"
        )
    return "".join(lines)


def _smpl_body_overs(material_prim_path: str) -> str:
    """Author per-body overs: rigid-body caps + per-collider material binding.

    Returns the USDA text for an ``over "bodies"`` block (to be nested
    inside ``def "Robot"``) that walks into each of the 24 SMPL body
    Xforms and:

    1. Authors ``physxRigidBody:*`` velocity / depenetration caps so
       ovphysx matches the IsaacGym/IsaacLab values the policy was
       trained against (and that OmniPVD captured).
    2. For the Pelvis (articulation root), additionally pins the
       articulation solver-iteration counts.
    3. Walks into ``collisions/_geom_<idx>`` and prepends
       ``MaterialBindingAPI`` + the physics-purpose binding to the given
       material — bound on the geom itself (not the body Xform) so
       ovphysx picks up the material directly when iterating colliders,
       no reliance on ancestor binding lookup.
    """
    body_blocks: list[str] = []
    indent = " " * 8
    for body_name, geom_idx in _SMPL_BODY_COLLIDER_GEOMS:
        caps = _body_rigid_body_caps_block(
            indent + "    ", is_articulation_root=(body_name == "Pelvis")
        )
        body_blocks.append(
            f'{indent}over "{body_name}"\n'
            f"{indent}{{\n"
            f"{caps}"
            f'{indent}    over "collisions"\n'
            f"{indent}    {{\n"
            f'{indent}        over "_geom_{geom_idx}" (\n'
            f'{indent}            prepend apiSchemas = ["MaterialBindingAPI"]\n'
            f"{indent}        )\n"
            f"{indent}        {{\n"
            f"{indent}            rel material:binding:physics = "
            f"<{material_prim_path}> (\n"
            f'{indent}                bindMaterialAs = "weakerThanDescendants"\n'
            f"{indent}            )\n"
            f"{indent}        }}\n"
            f"{indent}    }}\n"
            f"{indent}}}\n"
        )
    return (
        '    over "bodies"\n'
        "    {\n"
        + "".join(body_blocks)
        + "    }\n"
    )


def _smpl_joint_overs(
    joint_gains: Optional[dict[str, JointGains]] = None,
) -> str:
    """Author per-joint overs: drive type, kp, kd, maxForce, maxJointVelocity.

    Returns the USDA text for an ``over "joints"`` block (to be nested
    inside ``def "Robot"`` next to ``over "bodies"``) that walks into
    every SMPL joint Xform and authors the full PD drive
    configuration:

    - ``drive:rot{X,Y,Z}:physics:type`` — ``"force"`` by default,
      matching what the trainer actually runs. IsaacLab's
      ``ImplicitActuatorCfg`` hands ``stiffness`` / ``damping`` to
      PhysX without overriding the USD's drive type, and the upstream
      ``smpl_humanoid.usda`` ships ``"force"`` — so the trainer
      effectively runs force-mode PD (gain in N·m/rad). Switching to
      ``"acceleration"`` makes PhysX multiply the gain by each
      joint's effective inertia (mass-normalised PD); the SAME number
      then yields ~5× the torque on a hip, the policy fails to
      balance because it was trained against force-mode dynamics, and
      even ``--bypass-policy`` (motion DOFs piped straight to the PD)
      falls.
    - ``drive:rot{X,Y,Z}:physics:stiffness`` / ``damping`` — pulled
      from the resolved checkpoint so the gains track whatever the
      trainer used. Falls back to the SMPL group table baked in
      ``config_loader.SMPL_GROUP_GAINS`` when the .pt blob doesn't
      carry ``robot.control.control_info``.
    - ``drive:rot{X,Y,Z}:physics:maxForce`` — the trainer's
      ``effort_limit`` (500 N·m for SMPL); upstream USDA authors
      FLT_MAX which would let the PD apply unbounded torque.
    - ``physxJoint:maxJointVelocity`` — the trainer's
      ``velocity_limit`` (100 rad/s for SMPL); not in upstream USDA.

    If ``joint_gains`` is ``None`` we keep the legacy behaviour
    (only maxForce + maxJointVelocity; drive type stays at upstream
    USDA's ``"force"``, which is the trainer-equivalent value) so
    old callers still work.
    """
    joint_blocks: list[str] = []
    indent = " " * 8
    for joint_name in _SMPL_JOINT_NAMES:
        if joint_gains is not None:
            g = joint_gains[joint_name]
            kp_lines = "".join(
                f'{indent}    float drive:rot{ax}:physics:stiffness = '
                f"{g.stiffness}\n"
                f'{indent}    float drive:rot{ax}:physics:damping = '
                f"{g.damping}\n"
                f'{indent}    float drive:rot{ax}:physics:maxForce = '
                f"{g.effort_limit}\n"
                f'{indent}    uniform token drive:rot{ax}:physics:type = '
                f'"{g.drive_type}"\n'
                for ax in ("X", "Y", "Z")
            )
            max_vel = g.velocity_limit
        else:
            kp_lines = (
                f'{indent}    float drive:rotX:physics:maxForce = '
                f"{_JOINT_DRIVE_MAX_FORCE_NM}\n"
                f'{indent}    float drive:rotY:physics:maxForce = '
                f"{_JOINT_DRIVE_MAX_FORCE_NM}\n"
                f'{indent}    float drive:rotZ:physics:maxForce = '
                f"{_JOINT_DRIVE_MAX_FORCE_NM}\n"
            )
            max_vel = _JOINT_MAX_VELOCITY_RAD_PER_SEC
        joint_blocks.append(
            f'{indent}over "{joint_name}"\n'
            f"{indent}{{\n"
            f"{kp_lines}"
            f'{indent}    float physxJoint:maxJointVelocity = {max_vel}\n'
            f"{indent}}}\n"
        )
    return (
        '    over "joints"\n'
        "    {\n"
        + "".join(joint_blocks)
        + "    }\n"
    )


# Default radius for the mimic-target overlay markers. Roughly half the
# scale of a SMPL hand collider so the markers read as "joint dots" next
# to the simulated character without occluding it.
_MIMIC_TARGET_SPHERE_RADIUS_M = 0.04


def _mimic_target_markers_block(
    body_names: Sequence[str],
    radius: float = _MIMIC_TARGET_SPHERE_RADIUS_M,
) -> str:
    """USDA snippet authoring the ``/World/MimicTargets`` overlay.

    For each body name we author an ``Xform`` (writable via ovrtx's
    ``omni:xform`` attribute, same mechanism the real character bodies
    use) containing a single ``Sphere`` with a bright ``displayColor``.
    The Xforms ship with their transform set so the spheres start *well*
    below the ground; the parent process is expected to push real
    transforms each policy tick. Until it does, the markers stay
    invisible rather than piling at the origin.

    No physics APIs are applied, so ovphysx ignores the subtree.
    """
    indent = " " * 4
    inner = " " * 8
    deep = " " * 12
    blocks: list[str] = []
    for name in body_names:
        blocks.append(
            f'{inner}def Xform "{name}"\n'
            f"{inner}{{\n"
            f"{deep}matrix4d xformOp:transform = ( "
            f"(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), "
            f"(0, 0, -1000, 1) )\n"
            f'{deep}uniform token[] xformOpOrder = ["xformOp:transform"]\n'
            f'{deep}def Sphere "shape"\n'
            f"{deep}{{\n"
            f"{deep}    double radius = {radius}\n"
            f"{deep}    color3f[] primvars:displayColor = [(1.0, 0.25, 0.05)]\n"
            f"{deep}    float[] primvars:displayOpacity = [0.85]\n"
            f"{deep}}}\n"
            f"{inner}}}\n"
        )
    return (
        f'{indent}def Scope "MimicTargets"\n'
        f"{indent}{{\n"
        + "".join(blocks)
        + f"{indent}}}\n"
    )


@dataclass
class SceneAssets:
    """Resolved paths to the assets that go into the runtime USD."""

    runtime_usd: Path
    smpl_humanoid_usda: Path
    checkerboard_ground_usda: Path


def _usda_asset_path(p: Path) -> str:
    """Format an absolute path as a USD ``@<path>@`` asset reference.

    On Windows we need forward slashes inside USDA asset paths.
    """
    return p.resolve().as_posix()


def write_runtime_scene(
    out_path: Path | str,
    smpl_humanoid_usda: Path | str | None = None,
    checkerboard_ground_usda: Path | str | None = None,
    camera_translate: Tuple[float, float, float] = (3.5, -3.5, 1.5),
    camera_look_at_z: float = 1.0,
    render_resolution: Tuple[int, int] = (960, 540),
    mimic_target_body_names: Optional[Sequence[str]] = None,
    joint_gains: Optional[dict[str, JointGains]] = None,
) -> SceneAssets:
    """Write a self-contained USDA referencing SMPL, the ground, light and camera.

    The render side uses the ovrtx-expected layout:

    * ``/World/Camera``      — the actual ``Camera`` prim (xformable).
    * ``/Render/Camera``     — a ``RenderProduct`` that references
      ``/World/Camera`` and declares an ``LdrColor`` render var. This is
      the prim path passed to ``Renderer.step(render_products=...)``.

    Args:
        out_path: Destination for the runtime ``.usda`` file.
        smpl_humanoid_usda: Override path to ProtoMotions' ``smpl_humanoid.usda``.
        checkerboard_ground_usda: Override path to the ground USDA.
        camera_translate: Camera position in world space (meters, Z-up).
        camera_look_at_z: Approximate Z to keep the humanoid roughly centered.
        render_resolution: ``(width, height)`` for the RenderProduct.
        mimic_target_body_names: When provided, author a ``MimicTargets``
            scope under ``/World`` containing one sphere marker per body
            name (used by ``--draw-mimic-pose`` to overlay the motion
            reference). The order here defines the prim paths returned by
            :func:`mimic_target_prim_paths`. Authored only when truthy so
            the default scene stays unchanged.
        joint_gains: Optional per-joint :class:`JointGains` table. When
            provided, every joint over emits explicit
            ``stiffness`` / ``damping`` / ``maxForce`` / drive-type tokens
            so PPP no longer inherits the upstream USDA's ``"force"``
            drive type. Build with
            :func:`per_joint_gains_from_config`. ``None`` (legacy
            behaviour) leaves the upstream gains and only authors max
            caps.

    Returns:
        ``SceneAssets`` with resolved absolute paths.
    """
    out_path = Path(out_path).resolve()
    smpl_path = Path(
        smpl_humanoid_usda or _SMPL_HUMANOID_USDA
    ).resolve()
    ground_path = Path(
        checkerboard_ground_usda or _CHECKERBOARD_GROUND_USDA
    ).resolve()

    if not smpl_path.exists():
        raise FileNotFoundError(f"SMPL USDA not found: {smpl_path}")
    if not ground_path.exists():
        raise FileNotFoundError(f"Ground USDA not found: {ground_path}")

    cam_x, cam_y, cam_z = camera_translate
    res_w, res_h = render_resolution
    smpl_body_overs = _smpl_body_overs(SMPL_BODY_MATERIAL_PRIM)
    smpl_joint_overs = _smpl_joint_overs(joint_gains=joint_gains)
    mimic_targets_block = (
        _mimic_target_markers_block(mimic_target_body_names)
        if mimic_target_body_names
        else ""
    )

    body_text = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    # ``PhysxSceneAPI`` lets us pin the solver to TGS — ProtoMotions'
    # default (``IsaacGymPhysXParams.solver_type = 1``) and what OmniPVD
    # captured (``omni:pvd:solverType = "eTGS"``). The PhysX library
    # default is PGS, which is noticeably less stable under the stiff PD
    # gains the SMPL motion-tracker uses (stiffness ∈ [300, 1000]).
    def PhysicsScene "physicsScene" (
        prepend apiSchemas = ["PhysxSceneAPI"]
    )
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
        uniform token physxScene:solverType = "TGS"
        # Cap scene-wide solver iterations to match IsaacLab's
        # ``PhysxCfg(max_position_iteration_count=4,
        # max_velocity_iteration_count=4)`` from the resolved
        # checkpoint (sim.physx.num_position_iterations / num_velocity_iterations
        # = 4). The PhysX default cap is 255; the per-articulation
        # count we author on Pelvis (also 4/4) governs the actual
        # iteration count, but the scene cap is what IsaacLab sets, so
        # we mirror it to avoid any silent ovphysx-side clamp.
        int physxScene:maxPositionIterationCount = 4
        int physxScene:maxVelocityIterationCount = 4
        # ``bounce_threshold_velocity`` from the trainer's PhysxCfg. The
        # PhysX default is 0; IsaacLab/IsaacGym pin this to 0.2 m/s so
        # contacts below 0.2 m/s relative velocity don't bounce. Without
        # it, foot-ground micro-contacts at low closing speeds can add
        # spurious restitution noise that the policy wasn't trained
        # against.
        float physxScene:bounceThreshold = 0.2
    }}

    def "Robot" (
        prepend references = @{_usda_asset_path(smpl_path)}@
    )
    {{
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

{smpl_body_overs}{smpl_joint_overs}    }}

    # Visual-only reference to the upstream checkerboard ground. We strip
    # the upstream's PhysicsRigidBodyAPI (applied on the root) and clamp
    # ``rigidBodyEnabled = 0`` as a belt-and-suspenders so ovphysx ignores
    # this subtree entirely. The actual ground collider is the ``Plane``
    # below.
    def "Ground" (
        prepend references = @{_usda_asset_path(ground_path)}@
        delete apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {{
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        bool physics:rigidBodyEnabled = 0
    }}

    # Physics materials. ``PhysicsMaterialAPI`` is the standard USD physics
    # schema that carries friction / restitution / density. We bind these
    # to colliders via ``rel material:binding:physics`` (the ``physics``
    # purpose used by ``omni.physx`` and ovphysx — see
    # ``ovphysx/.../physicsUtils.py::add_physics_material_to_prim``).
    # PhysX combines per-contact friction by averaging the two bodies'
    # materials by default, so the final coefficient at a humanoid-ground
    # contact is ``(1.0 + 0.5) / 2 = 0.75`` with the values authored here.
    def Scope "PhysicsMaterials"
    {{
        def Material "GroundMaterial" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = 1.0
            float physics:dynamicFriction = 1.0
            float physics:restitution = 0.0
        }}

        def Material "SmplBodyMaterial" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            # IsaacLab leaves robot shape friction at the engine
            # default (~1.0; see protomotions/simulator/isaaclab/utils/
            # scene.py — no per-body material override) so PPP matches
            # that. The earlier 0.5/0.5 came from a contact tuning
            # round before integrator parity was prioritised; combined
            # with the ground's 1.0/1.0 it yielded an effective 0.75
            # friction (PhysX averages by default), which is below
            # what the trainer saw.
            float physics:staticFriction = 1.0
            float physics:dynamicFriction = 1.0
            float physics:restitution = 0.0
        }}
    }}

    # Static ground collider. Matches the canonical ovphysx pattern in
    # ``ovphysx/samples/data/boxes_falling_on_groundplane.usda``:
    # a ``Plane`` prim with ``PhysicsCollisionAPI`` (no rigid body) is
    # treated as an infinite static collider. ``purpose = "guide"`` keeps
    # it invisible — the checkerboard above provides the visual.
    #
    # ``PhysxCollisionAPI`` carries the contact / rest offsets. The
    # IsaacLab checkpoint's ``sim.physx.contact_offset = 0.02`` is what
    # the policy was trained against (applied to both robot colliders
    # and terrain by ``isaaclab/utils/scene.py``), so we mirror that
    # value here. An earlier OmniPVD-driven build pinned this to 0.001 m
    # to match an IsaacGym terrain-mesh offset (~0.00136 m); for the
    # IsaacLab checkpoint shipped with PPP today, 0.02 m is correct.
    # If a future IsaacGym checkpoint regresses this, switch back to
    # 0.001 m and gate on the loaded ``simulator`` type in
    # ``config_loader``.
    def Plane "GroundCollider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        uniform token axis = "Z"
        uniform token purpose = "guide"
        bool physics:collisionEnabled = 1
        float physxCollision:contactOffset = 0.02
        float physxCollision:restOffset = 0
        rel material:binding:physics = </World/PhysicsMaterials/GroundMaterial> (
            bindMaterialAs = "weakerThanDescendants"
        )
    }}

    def DistantLight "Sun"
    {{
        float inputs:intensity = 3000
        float inputs:angle = 1.0
        double3 xformOp:rotateXYZ = (-45, 0, 30)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}

{mimic_targets_block}
    def Camera "Camera"
    {{
        float focalLength = 24
        float focusDistance = 4
        float fStop = 0
        float horizontalAperture = 20.955
        float verticalAperture = 15.2908
        float2 clippingRange = (0.1, 1000)
        double3 xformOp:translate = ({cam_x}, {cam_y}, {cam_z})
        double3 xformOp:rotateXYZ = (75, 0, 45)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}
}}

def Scope "Render"
{{
    def RenderProduct "Camera"
    {{
        rel camera = </World/Camera>
        int2 resolution = ({res_w}, {res_h})
        rel orderedVars = <LdrColor>

        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}
    }}
}}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body_text, encoding="utf-8")

    return SceneAssets(
        runtime_usd=out_path,
        smpl_humanoid_usda=smpl_path,
        checkerboard_ground_usda=ground_path,
    )


def body_prim_paths(body_names) -> list[str]:
    """Return the full USD prim path for each robot body name."""
    return [f"{ROBOT_BODIES_ROOT}/{name}" for name in body_names]


def mimic_target_prim_paths(body_names) -> list[str]:
    """Return the per-body marker prim paths under ``/World/MimicTargets``.

    The list aligns 1:1 with the body order used by ``body_prim_paths``;
    callers feed the parallel-indexed motion-reference positions into
    ``RtxViewer.push_mimic_target_positions``.
    """
    return [f"{MIMIC_TARGETS_ROOT}/{name}" for name in body_names]
