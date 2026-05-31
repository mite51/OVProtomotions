"""Load the ONNX sidecar YAML (``unified_pipeline.yaml`` and friends).

The new ProtoMotions export (``deployment/export_bm_tracker_onnx.py``) bakes
the observation computation *into* the ONNX graph. Together with the ONNX
file it writes a rich YAML sidecar that fully describes the deployment
contract: input names + shapes, output names, joint / body orderings,
PD gains, control timing, and the future-step indices the policy was
trained against. PPP's inference loop is driven entirely from that YAML
— no more rummaging through the trainer's ``resolved_configs_inference.pt``
blob and reconstructing fields by hand.

See ``docs/onnx_input_migration.md`` for the full new-input rundown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# SMPL fallbacks
#
# The YAML sidecar carries everything PPP needs at runtime, but a few
# fields (effort_limit, velocity_limit) are sometimes ``null`` in the
# dump. We fall back to the canonical SMPL group table from
# ``protomotions/robot_configs/smpl.py::SMPLRobotConfig.control``.
# ----------------------------------------------------------------------
SMPL_BODY_NAMES: List[str] = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

SMPL_PARENT_INDICES: List[int] = [
    -1,  0,  1,  2,  3,  0,  5,  6,  7,  0,
     9, 10, 11, 12, 11, 14, 15, 16, 17, 11,
    19, 20, 21, 22,
]

SMPL_DOF_NAMES: List[str] = [
    f"{joint}_{axis}"
    for joint in [
        "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
        "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
        "Torso", "Spine", "Chest", "Neck", "Head",
        "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
        "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
    ]
    for axis in ("x", "y", "z")
]


# (regex-fragment-list, stiffness, damping, effort_limit, velocity_limit)
# Used only as a fallback when the YAML doesn't carry per-DOF gains;
# the unified-pipeline export always ships explicit per-DOF arrays so
# this is rarely exercised in practice.
_SMPL_GROUP_GAINS = (
    (("Hip", "Knee", "Ankle"),                     800.0, 80.0,  500.0, 100.0),
    (("Toe",),                                     500.0, 50.0,  500.0, 100.0),
    (("Torso", "Spine", "Chest"),                 1000.0, 100.0, 500.0, 100.0),
    (("Neck", "Head"),                             500.0, 50.0,  500.0, 100.0),
    (("Thorax", "Shoulder", "Elbow"),              500.0, 50.0,  500.0, 100.0),
    (("Wrist", "Hand"),                            300.0, 30.0,  500.0, 100.0),
)


def _smpl_default_gain_table() -> Tuple[List[float], List[float], List[float], List[float]]:
    kp: List[float] = []
    kd: List[float] = []
    eff: List[float] = []
    vel: List[float] = []
    for dof in SMPL_DOF_NAMES:
        joint = dof.rsplit("_", 1)[0]
        for tags, k_p, k_d, e, v in _SMPL_GROUP_GAINS:
            if any(tag in joint for tag in tags):
                kp.append(k_p)
                kd.append(k_d)
                eff.append(e)
                vel.append(v)
                break
        else:
            raise RuntimeError(
                f"DOF {dof!r} did not match any SMPL gain group. "
                "Update _SMPL_GROUP_GAINS in config_loader.py."
            )
    return kp, kd, eff, vel


_SMPL_KP_DEFAULT, _SMPL_KD_DEFAULT, _SMPL_EFFORT_DEFAULT, _SMPL_VELOCITY_DEFAULT = (
    _smpl_default_gain_table()
)


# ----------------------------------------------------------------------
# Resolved config
# ----------------------------------------------------------------------
@dataclass
class PolicyInputSpec:
    """One entry from ``policy_inputs`` in the YAML sidecar.

    Maps the ONNX input ``name`` (as ONNX rewrote it during export, e.g.
    ``current_rigid_body_pos``) back to the dotted context ``key`` it was
    sourced from (``current.rigid_body_pos``) plus the shape and kind
    metadata needed to fill it at inference time.
    """

    name: str          # ONNX input name (after sanitization)
    key: str           # Dotted context key (current.rigid_body_pos, ...)
    shape: List[int]   # Full shape including the leading batch dim
    kind: Optional[str] = None
    output_key: Optional[str] = None  # For historical inputs that feed back


@dataclass
class PolicyOutputSpec:
    """One entry from ``policy_outputs`` in the YAML sidecar."""

    name: str          # ONNX output name (actions, joint_pos_targets, ...)
    key: str
    kind: str
    shape: List[int]
    joint_names: Optional[List[str]] = None


@dataclass
class ResolvedConfig:
    """Inference-time configuration assembled from the ONNX sidecar YAML."""

    # Joint / body schema.
    body_names: List[str] = field(default_factory=lambda: list(SMPL_BODY_NAMES))
    dof_names: List[str] = field(default_factory=lambda: list(SMPL_DOF_NAMES))
    parent_indices: List[int] = field(
        default_factory=lambda: list(SMPL_PARENT_INDICES)
    )

    # Per-DOF PD gains. With the unified-pipeline export the policy
    # also emits per-step ``stiffness_targets`` / ``damping_targets``
    # but PPP keeps the gains static (USD-authored) and uses the
    # defaults from the YAML as the single source of truth — see
    # CHANGELOG for the dynamic-gain caveat.
    pd_stiffness: List[float] = field(
        default_factory=lambda: list(_SMPL_KP_DEFAULT)
    )
    pd_damping: List[float] = field(
        default_factory=lambda: list(_SMPL_KD_DEFAULT)
    )
    pd_effort_limit: List[float] = field(
        default_factory=lambda: list(_SMPL_EFFORT_DEFAULT)
    )
    pd_velocity_limit: List[float] = field(
        default_factory=lambda: list(_SMPL_VELOCITY_DEFAULT)
    )

    # USD ``drive:rot*:physics:type`` token. ``"force"`` matches what
    # the trainer actually runs on IsaacLab.
    #
    # The previous default here was ``"acceleration"`` with a comment
    # claiming it mirrored IsaacLab's ``ImplicitActuatorCfg``. That
    # was wrong: per the IsaacLab actuator docs
    # (https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#implicit-actuator)
    # ``ImplicitActuator`` "does not perform its own computations on
    # the joint action that needs to be applied to the simulation" —
    # the articulation just writes ``stiffness`` / ``damping`` into
    # the PhysX PD and inherits the drive type from whatever the USD
    # authors. The upstream ``smpl_humanoid.usda`` authors
    # ``"force"`` (see the historical note in assets.py — "the
    # upstream USDA's `force` drives"), so the trainer's effective
    # drive type is ``"force"``.
    #
    # ``"acceleration"`` makes PhysX multiply the gain by each joint's
    # effective inertia before integrating: stiffness 800 on a hip
    # with ~5 kg·m² of reflected leg inertia becomes effective
    # torque-stiffness 4000 N·m/rad, ~5× what the policy trained
    # against. That mismatch is enough to make even ``--bypass-policy``
    # (motion's own DOFs piped straight to the PD) lose balance — the
    # PD overshoots, oscillates, and the character falls. Diagnosed
    # against trainer-vs-PPP parity dumps where outputs matched
    # bit-exact and motion lookup matched to fp32 noise, but the
    # character still couldn't hold the pose.
    drive_type: str = "force"

    # Spawn lift. Not carried in the YAML; pinned to ProtoMotions'
    # canonical 5 cm character clearance.
    ref_respawn_offset_z: float = 0.05

    # Control timing (from YAML ``timing.*``).
    control_dt: float = 1.0 / 30.0
    physics_dt: float = 1.0 / 120.0
    decimation: int = 4

    # Mimic future look-ahead. ``future_step_indices`` is the canonical
    # source — a list of integer offsets in units of ``control_dt``
    # (e.g. ``[1]`` means "one control step ahead"). ``future_steps``
    # is kept as a derived count for downstream consumers that just
    # want the length.
    future_step_indices: List[int] = field(default_factory=lambda: [1])

    # ONNX-side schema, mirrors the YAML's ``policy_inputs`` /
    # ``policy_outputs`` blocks. Sorted in YAML order so callers can
    # iterate them deterministically.
    policy_inputs: List[PolicyInputSpec] = field(default_factory=list)
    policy_outputs: List[PolicyOutputSpec] = field(default_factory=list)

    # Path to the YAML the config was loaded from (None when defaults).
    yaml_path: Optional[Path] = None
    # Path to the ONNX file the YAML was discovered for (None when
    # the YAML was loaded explicitly).
    onnx_path: Optional[Path] = None

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def num_dofs(self) -> int:
        return len(self.dof_names)

    @property
    def future_steps(self) -> int:
        return len(self.future_step_indices)

    @property
    def policy_fps(self) -> float:
        return 1.0 / self.control_dt

    @property
    def physics_fps(self) -> float:
        return 1.0 / self.physics_dt

    @property
    def dt_physics(self) -> float:
        """Substep ``dt`` (seconds) — alias for ``physics_dt``."""
        return self.physics_dt

    def get_input_spec(self, name: str) -> Optional[PolicyInputSpec]:
        for spec in self.policy_inputs:
            if spec.name == name:
                return spec
        return None

    def get_input_spec_by_key(self, key: str) -> Optional[PolicyInputSpec]:
        for spec in self.policy_inputs:
            if spec.key == key:
                return spec
        return None


# ----------------------------------------------------------------------
# YAML discovery / loading
# ----------------------------------------------------------------------
def discover_yaml_sidecar(onnx_path: Path | str) -> Optional[Path]:
    """Find the unified-pipeline YAML next to an ONNX file.

    Search order:

    1. ``<onnx_stem>.yaml`` (the canonical export-script naming).
    2. ``<onnx_parent>/unified_pipeline.yaml`` (fallback name the
       upstream export script uses when the model file was renamed).
    3. Any single ``*.yaml`` file in the same directory.

    Returns ``None`` if no candidate is found; the caller decides
    whether to fall back to the built-in defaults.
    """
    onnx_path = Path(onnx_path).resolve()
    candidates: List[Path] = []
    stem_yaml = onnx_path.with_suffix(".yaml")
    if stem_yaml.exists():
        candidates.append(stem_yaml)
    pipeline_yaml = onnx_path.parent / "unified_pipeline.yaml"
    if pipeline_yaml.exists() and pipeline_yaml not in candidates:
        candidates.append(pipeline_yaml)
    if not candidates:
        siblings = sorted(onnx_path.parent.glob("*.yaml"))
        if len(siblings) == 1:
            candidates.append(siblings[0])

    if not candidates:
        return None
    return candidates[0]


def _coerce_int_list(v: Any) -> List[int]:
    if v is None:
        return []
    if hasattr(v, "tolist"):
        v = v.tolist()
    return [int(x) for x in v]


def _coerce_float_list(v: Any, n: int, default: List[float]) -> List[float]:
    if v is None:
        return list(default)
    if hasattr(v, "tolist"):
        v = v.tolist()
    out = [float(x) for x in v]
    if len(out) != n:
        log.warning(
            "Per-DOF list has %d entries, expected %d; padding with defaults.",
            len(out),
            n,
        )
        if len(out) < n:
            out = out + list(default[len(out):])
        else:
            out = out[:n]
    return out


def _parse_input_spec(entry: Dict[str, Any]) -> PolicyInputSpec:
    return PolicyInputSpec(
        name=str(entry["name"]),
        key=str(entry.get("key", entry["name"])),
        shape=[int(x) for x in entry.get("shape", [])],
        kind=entry.get("kind"),
        output_key=entry.get("output_key"),
    )


def _parse_output_spec(entry: Dict[str, Any]) -> PolicyOutputSpec:
    return PolicyOutputSpec(
        name=str(entry["name"]),
        key=str(entry.get("key", entry["name"])),
        kind=str(entry.get("kind", "")),
        shape=[int(x) for x in entry.get("shape", [])],
        joint_names=(
            list(entry["joint_names"]) if entry.get("joint_names") else None
        ),
    )


def load_config_from_yaml(
    yaml_path: Path | str,
    onnx_path: Optional[Path | str] = None,
) -> ResolvedConfig:
    """Parse a unified-pipeline YAML sidecar into a :class:`ResolvedConfig`.

    The YAML schema is the one produced by
    ``deployment/export_bm_tracker_onnx.py::_build_yaml``. Unknown fields
    are ignored so PPP keeps working when the export script grows new
    keys.
    """
    import yaml

    yaml_path = Path(yaml_path).resolve()
    log.info("Loading model config from %s", yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        blob: Dict[str, Any] = yaml.safe_load(f) or {}

    cfg = ResolvedConfig(
        yaml_path=yaml_path,
        onnx_path=Path(onnx_path).resolve() if onnx_path is not None else None,
    )

    body_names = blob.get("body_names") or (blob.get("robot") or {}).get("body_names")
    joint_names = blob.get("joint_names") or (blob.get("robot") or {}).get("joint_names")
    if body_names:
        cfg.body_names = [str(n) for n in body_names]
    if joint_names:
        cfg.dof_names = [str(n) for n in joint_names]

    num_dofs = len(cfg.dof_names)

    # PD gains: prefer the deployment-contract ``control`` block, fall
    # back to the top-level ``default_joint_*`` arrays, then the SMPL
    # group fallback.
    control = blob.get("control") or {}
    stiffness = (
        control.get("stiffness")
        if control.get("stiffness") is not None
        else blob.get("default_joint_stiffness")
    )
    damping = (
        control.get("damping")
        if control.get("damping") is not None
        else blob.get("default_joint_damping")
    )
    effort = control.get("effort_limits")
    velocity = control.get("velocity_limits")

    cfg.pd_stiffness = _coerce_float_list(stiffness, num_dofs, _SMPL_KP_DEFAULT)
    cfg.pd_damping = _coerce_float_list(damping, num_dofs, _SMPL_KD_DEFAULT)
    cfg.pd_effort_limit = _coerce_float_list(
        effort, num_dofs, _SMPL_EFFORT_DEFAULT
    )
    cfg.pd_velocity_limit = _coerce_float_list(
        velocity, num_dofs, _SMPL_VELOCITY_DEFAULT
    )

    timing = blob.get("timing") or {}
    if timing.get("control_dt") is not None:
        cfg.control_dt = float(timing["control_dt"])
    elif blob.get("dt") is not None:
        cfg.control_dt = float(blob["dt"])
    if timing.get("physics_dt") is not None:
        cfg.physics_dt = float(timing["physics_dt"])
    if timing.get("decimation") is not None:
        cfg.decimation = int(timing["decimation"])

    motion = blob.get("motion") or {}
    fsi = _coerce_int_list(motion.get("future_step_indices"))
    if fsi:
        cfg.future_step_indices = fsi

    cfg.policy_inputs = [
        _parse_input_spec(e) for e in (blob.get("policy_inputs") or [])
    ]
    cfg.policy_outputs = [
        _parse_output_spec(e) for e in (blob.get("policy_outputs") or [])
    ]

    log.info(
        "Resolved config: %d bodies, %d DOFs, control_dt=%.4fs (%.1f Hz), "
        "physics_dt=%.4fs (%.1f Hz), decimation=%d, future_step_indices=%s, "
        "%d policy inputs / %d policy outputs.",
        cfg.num_bodies,
        cfg.num_dofs,
        cfg.control_dt,
        cfg.policy_fps,
        cfg.physics_dt,
        cfg.physics_fps,
        cfg.decimation,
        cfg.future_step_indices,
        len(cfg.policy_inputs),
        len(cfg.policy_outputs),
    )
    return cfg


def load_config_for_onnx(
    onnx_path: Path | str,
    explicit_yaml: Optional[Path | str] = None,
) -> ResolvedConfig:
    """Load the model config for an ONNX file, auto-discovering the sidecar.

    If ``explicit_yaml`` is provided, that path is used unconditionally.
    Otherwise we look next to the ONNX file using
    :func:`discover_yaml_sidecar`. If nothing matches we raise — the
    unified-pipeline ONNX is unusable without its schema.
    """
    if explicit_yaml is not None:
        return load_config_from_yaml(explicit_yaml, onnx_path=onnx_path)

    yaml_path = discover_yaml_sidecar(onnx_path)
    if yaml_path is None:
        raise FileNotFoundError(
            f"No YAML sidecar found next to {onnx_path!s}. Expected one of "
            "<stem>.yaml, unified_pipeline.yaml, or a single *.yaml in the "
            "same folder. Pass --yaml to override."
        )
    return load_config_from_yaml(yaml_path, onnx_path=onnx_path)
