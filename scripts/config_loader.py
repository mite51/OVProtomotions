"""Load ``resolved_configs_inference.pt`` (or fall back to hard-coded SMPL).

The motion-tracker checkpoint directory ships:

- ``last.ckpt``
- ``resolved_configs_inference.pt`` (a pickled ``EnvConfig`` plus
  ``AgentConfig``)
- ``resolved_configs_inference.yaml`` (human-readable copy)

We only need a small slice of the env config: body names, DOF names, action
config (offset, scale), and a few timings (num_state_history_steps,
future_steps). All defaults match the SMPL motion-tracker resolved config so
the app can still boot if the .pt is missing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


log = logging.getLogger(__name__)


SMPL_BODY_NAMES: List[str] = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

# SMPL kinematic tree (parent body index per body, -1 for the articulation
# root). Hard-coded mirror of ``robot.kinematic_info.parent_indices`` from
# ``resolved_configs_inference.{yaml,pt}`` for the motion-tracker SMPL
# config; used as the fallback when the .pt blob is missing or doesn't
# carry parent indices. Verified bit-for-bit against the .yaml.
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


# Per-joint-group PD gain table from
# ``protomotions/robot_configs/smpl.py::SMPLRobotConfig.control``. The
# trainer applies these via ``override_control_info`` regex matching on
# DOF names (one entry per joint × {x, y, z}); we mirror the regexes as
# a flat per-DOF table so PPP boots correctly when the .pt blob is
# missing or doesn't carry ``robot.control.control_info``. The yaml
# dump confirms these values land verbatim in the resolved config.
_SMPL_GROUP_GAINS = (
    # (regex-fragment-list, stiffness, damping, effort_limit, velocity_limit)
    (("Hip", "Knee", "Ankle"),                     800.0, 80.0,  500.0, 100.0),
    (("Toe",),                                     500.0, 50.0,  500.0, 100.0),
    (("Torso", "Spine", "Chest"),                 1000.0, 100.0, 500.0, 100.0),
    (("Neck", "Head"),                             500.0, 50.0,  500.0, 100.0),
    (("Thorax", "Shoulder", "Elbow"),              500.0, 50.0,  500.0, 100.0),
    (("Wrist", "Hand"),                            300.0, 30.0,  500.0, 100.0),
)


def _smpl_default_gain_table() -> tuple[List[float], List[float], List[float], List[float]]:
    """Build per-DOF (kp, kd, effort, velocity) lists in :data:`SMPL_DOF_NAMES` order.

    Walks each joint name, finds the matching SMPL group, and emits the
    group's gains for every axis suffix. Used as the fallback when the
    checkpoint .pt doesn't carry ``robot.control.control_info``.
    """
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


@dataclass
class ResolvedConfig:
    """Subset of fields we actually use during inference."""

    body_names: List[str] = field(default_factory=lambda: list(SMPL_BODY_NAMES))
    dof_names: List[str] = field(default_factory=lambda: list(SMPL_DOF_NAMES))
    # Per-body parent index (length == ``num_bodies``); root is ``-1``.
    # Used by :class:`ppp.robot_chain.RobotChain` to walk the chain when
    # computing body velocities from joint state. Defaults to the SMPL
    # tree so PPP boots correctly when the checkpoint .pt is missing.
    parent_indices: List[int] = field(
        default_factory=lambda: list(SMPL_PARENT_INDICES)
    )

    pd_action_offset: List[float] = field(default_factory=lambda: [0.0] * 69)
    pd_action_scale: List[float] = field(default_factory=lambda: [math.pi] * 69)
    action_transform: str = "tanh"
    clamp_value: float = 1.0

    # Per-DOF physical PD gains and limits, mirroring the trainer's
    # ``robot.control.control_info`` dict (resolved via
    # ``override_control_info`` regex matching at training time). The
    # checkpoint yaml carries these per-DOF; we read them out of the
    # .pt file and fall back to the SMPL group table if absent. PPP
    # writes them into the runtime USD as
    # ``drive:rot{X,Y,Z}:physics:stiffness/damping/maxForce`` and
    # ``physxJoint:maxJointVelocity`` so ovphysx's built-in articulation
    # PD matches the trainer's solver-implicit PD numerically.
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

    # USD ``drive:rot*:physics:type`` token. ``"acceleration"`` mirrors
    # IsaacLab's ``ImplicitActuatorCfg`` (the trainer's actuator model
    # for this checkpoint): the solver mass-normalises the torque so a
    # given (kp, kd) produces the same closed-loop dynamics regardless
    # of inertia. ``"force"`` is the upstream USDA default and behaves
    # like a raw torque drive — same numbers, different response. See
    # CHANGELOG / phase-A audit for why this matters.
    drive_type: str = "acceleration"

    num_state_history_steps: int = 2
    future_steps: int = 1
    ref_respawn_offset_z: float = 0.05

    # Simulator timing (training defaults for IsaacLab; the policy was trained
    # on a 120Hz physics step with decimation=4 → 30Hz policy tick).
    #
    # ``substeps`` is the IsaacGym-style "PhysX substeps per fps step" knob
    # (``IsaacGymSimParams.substeps`` defaults to 2). The trained policy's
    # PhysX integrator advances at ``1 / (physics_fps * substeps)`` seconds
    # per substep — for the canonical IsaacGym SMPL config that's
    # ``1 / (60 * 2) = 1/120 s``, even though ``fps`` itself is 60. PPP
    # exposes this via :pyattr:`dt_physics` and uses it as the per-substep
    # ``dt`` argument to ``ovphysx.PhysX.step``. Defaults to ``substeps=1``
    # so the IsaacLab-style configs (``fps=120``, no substeps) still
    # integrate at 120Hz.
    physics_fps: int = 120
    decimation: int = 4
    substeps: int = 1

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def num_dofs(self) -> int:
        return len(self.dof_names)

    @property
    def policy_fps(self) -> float:
        return self.physics_fps / self.decimation

    @property
    def dt_physics(self) -> float:
        """Substep ``dt`` (seconds) the trained PhysX integrator used.

        ``1 / (physics_fps * substeps)``. For an IsaacLab SMPL config
        (``fps=120``, ``substeps=1``) this is ``1/120 s``. For an
        IsaacGym SMPL config (``fps=60``, ``substeps=2``) it's *also*
        ``1/120 s`` — the substeps multiplier is what makes the two
        training paths share an integrator rate despite different
        nominal ``fps``.
        """
        return 1.0 / (self.physics_fps * max(1, self.substeps))


def _maybe_list(v: Any, n: int, default: float) -> List[float]:
    if v is None:
        return [default] * n
    if hasattr(v, "tolist"):
        try:
            v = v.tolist()
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)] * n


def _dig(obj: Any, *keys: str, default: Any = None) -> Any:
    """Best-effort nested attribute/dict lookup."""
    cur = obj
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, default if k == keys[-1] else None)
        else:
            cur = getattr(cur, k, default if k == keys[-1] else None)
    return cur if cur is not None else default


def load_resolved_config(checkpoint_path: Optional[Path | str]) -> ResolvedConfig:
    """Try to load ``resolved_configs_inference.pt`` next to ``checkpoint_path``.

    Returns the default SMPL config if anything goes wrong (with a warning).
    """
    cfg = ResolvedConfig()
    if checkpoint_path is None:
        return cfg

    ckpt = Path(checkpoint_path).resolve()
    pt_path = ckpt.parent / "resolved_configs_inference.pt"
    if not pt_path.exists():
        log.warning(
            "%s not found — using built-in SMPL defaults. "
            "This should still match the smpl_policy_orig export.",
            pt_path,
        )
        return cfg

    try:
        import torch

        log.info("Loading resolved configs from %s", pt_path)
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:  # pragma: no cover - falls back gracefully
        log.warning("Failed to torch.load(%s): %s. Using defaults.", pt_path, e)
        return cfg

    # The pt blob is structured as
    #   {"robot": RobotConfig, "simulator": SimulatorConfig,
    #    "env": EnvConfig, "agent": PPOAgentConfig, ...}
    # We pull the relevant fields out of ``robot`` and ``env``.
    if isinstance(blob, dict):
        robot_cfg = blob.get("robot")
        env_cfg = blob.get("env")
        sim_cfg = blob.get("simulator")
    else:
        robot_cfg = getattr(blob, "robot", None)
        env_cfg = getattr(blob, "env", None)
        sim_cfg = getattr(blob, "simulator", None)

    if env_cfg is None:
        log.warning("resolved_configs_inference.pt missing 'env'; using defaults.")
        return cfg

    body_names = _dig(robot_cfg, "kinematic_info", "body_names")
    dof_names = _dig(robot_cfg, "kinematic_info", "dof_names")
    if body_names:
        cfg.body_names = list(body_names)
    if dof_names:
        cfg.dof_names = list(dof_names)

    parent_indices = _dig(robot_cfg, "kinematic_info", "parent_indices")
    if parent_indices is not None:
        try:
            if hasattr(parent_indices, "tolist"):
                parent_indices = parent_indices.tolist()
            cfg.parent_indices = [int(p) for p in parent_indices]
        except Exception:
            log.warning(
                "Failed to parse robot.kinematic_info.parent_indices; "
                "falling back to hard-coded SMPL tree."
            )

    # ``robot.control.control_info`` is a per-DOF dict with the keys
    # ``stiffness`` / ``damping`` / ``effort_limit`` / ``velocity_limit``
    # (and ``armature`` / ``friction``, which we don't currently
    # plumb). The checkpoint resolves the ``override_control_info``
    # regexes into a flat per-DOF dict, so we just look up each DOF
    # name. If the .pt carries a different control model (e.g. raw
    # torque) we leave the SMPL fallback in place — PPP only
    # supports BUILT_IN_PD for now.
    control_info = _dig(robot_cfg, "control", "control_info")
    if control_info is not None:
        try:
            kp_list: List[float] = []
            kd_list: List[float] = []
            eff_list: List[float] = []
            vel_list: List[float] = []
            for dof in cfg.dof_names:
                entry: Any = None
                if isinstance(control_info, dict):
                    entry = control_info.get(dof)
                else:
                    entry = getattr(control_info, dof, None)
                if entry is None:
                    raise KeyError(f"control_info missing entry for {dof!r}")
                kp_list.append(float(_dig(entry, "stiffness")))
                kd_list.append(float(_dig(entry, "damping")))
                eff_list.append(float(_dig(entry, "effort_limit")))
                vel_list.append(float(_dig(entry, "velocity_limit")))
            cfg.pd_stiffness = kp_list
            cfg.pd_damping = kd_list
            cfg.pd_effort_limit = eff_list
            cfg.pd_velocity_limit = vel_list
        except Exception as e:
            log.warning(
                "Failed to parse robot.control.control_info (%s); "
                "falling back to hard-coded SMPL group gains.",
                e,
            )

    num_dofs = len(cfg.dof_names)
    ac = _dig(env_cfg, "action_config", default={})
    cfg.pd_action_offset = _maybe_list(_dig(ac, "pd_action_offset"), num_dofs, 0.0)
    cfg.pd_action_scale = _maybe_list(_dig(ac, "pd_action_scale"), num_dofs, math.pi)
    transform = _dig(ac, "action_transform")
    if transform:
        cfg.action_transform = str(transform)
    clamp = _dig(ac, "clamp_value")
    if clamp is not None:
        try:
            cfg.clamp_value = float(clamp)
        except Exception:
            pass

    nshs = _dig(env_cfg, "num_state_history_steps")
    if nshs is not None:
        try:
            cfg.num_state_history_steps = int(nshs)
        except Exception:
            pass

    fs = _dig(env_cfg, "control_components", "mimic", "future_steps")
    if fs is None:
        # Try the mimic_obs / mimic_target_poses observation component path.
        fs = _dig(env_cfg, "observation_components", "mimic_target_poses", "future_steps")
    if fs is not None:
        try:
            cfg.future_steps = int(fs)
        except Exception:
            pass

    rro = _dig(env_cfg, "control_components", "mimic", "ref_respawn_offset")
    if rro is None:
        rro = _dig(env_cfg, "control_components", "mimic", "respawn_offset_z")
    if rro is not None:
        try:
            cfg.ref_respawn_offset_z = float(rro)
        except Exception:
            pass

    # SimulatorConfig nests fps/decimation under a ``sim`` (SimParams)
    # field — see ``protomotions/simulator/base_simulator/config.py``.
    # Earlier versions of this loader dug ``sim_cfg.fps`` directly,
    # which returns None against the canonical layout and silently fell
    # back to the 120/4 defaults. We try the nested location first and
    # fall through to the legacy flat lookup for forward compatibility.
    fps = _dig(sim_cfg, "sim", "fps")
    if fps is None:
        fps = _dig(sim_cfg, "fps")
    if fps is not None:
        try:
            cfg.physics_fps = int(fps)
        except Exception:
            pass
    dec = _dig(sim_cfg, "sim", "decimation")
    if dec is None:
        dec = _dig(sim_cfg, "decimation")
    if dec is not None:
        try:
            cfg.decimation = int(dec)
        except Exception:
            pass

    # IsaacGym/Genesis-style configs add a ``substeps`` field on top of
    # ``fps`` (PhysX-internal solver substeps inside a single 1/fps step
    # — see ``protomotions/simulator/isaacgym/config.py``). PPP threads
    # this through to ``PhysxWorld`` as the per-substep ``dt`` so the
    # actual integrator rate matches what the policy was trained on:
    # ``IsaacGym SMPL`` has ``fps=60, substeps=2`` ⇒ 1/120 s substep,
    # ``IsaacLab SMPL`` has ``fps=120, substeps=1`` ⇒ 1/120 s substep.
    # Without the multiplication PPP would silently integrate the
    # IsaacGym checkpoint at half the trained solver rate.
    substeps = _dig(sim_cfg, "sim", "substeps")
    if substeps is None:
        substeps = _dig(sim_cfg, "substeps")
    if substeps is not None:
        try:
            cfg.substeps = max(1, int(substeps))
        except Exception:
            pass

    log.info(
        "Resolved config: %d bodies, %d DOFs, fps=%d/decimation=%d/substeps=%d "
        "(policy=%.1f Hz, integrator=%.1f Hz), future_steps=%d, "
        "num_state_history=%d, action_transform=%s.",
        cfg.num_bodies,
        cfg.num_dofs,
        cfg.physics_fps,
        cfg.decimation,
        cfg.substeps,
        cfg.policy_fps,
        1.0 / cfg.dt_physics,
        cfg.future_steps,
        cfg.num_state_history_steps,
        cfg.action_transform,
    )
    return cfg
