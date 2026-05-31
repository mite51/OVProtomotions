"""Top-level application: compose the scene, drive the unified-pipeline
policy loop, render.

The new ProtoMotions export bakes obs computation + action processing
into a single ONNX graph (``unified_pipeline.onnx``), so each policy
tick is now just:

1. ``state = world.read_state()``                  (read PhysX)
2. ``future = motion.get_future(t, control_dt, step_indices=...)``
3. ``feed = obs_builder.build(state, future, ground_height)``
4. ``out = policy.run(feed)``                      (ONNX forward pass)
5. ``obs_builder.push_action(out.actions)``        (history buffer)
6. ``world.set_dof_targets(out.joint_pos_targets); world.step(control_dt)``
7. ``viewer.push_body_transforms(state.body_pos, state.body_rot)``
8. ``t += control_dt``; on motion end, ``reset()``.

No more host-side obs functions, ``pi * tanh(mu)`` post-processing, or
graph-patching to expose ``raw_mu`` — the ONNX outputs ``actions`` and
``joint_pos_targets`` directly. See ``docs/onnx_input_migration.md``
for the full migration rundown.

The outer ``run()`` loop still turns variable wall ``dt`` into a
constant policy cadence with a simple accumulator: each iter,
``sim_time_accum += wall_dt * --time-scale`` (capped), then drain the
accumulator in ``dt_policy``-sized chunks by calling ``policy_tick()``.
``viewer.render`` + ``viewer.poll_events`` run once per outer iter so
the renderer and keyboard stay responsive even when no policy tick
fires.

Reset writes the articulation root pose and DOF positions/velocities to
the motion's first frame and zeros the action history. The
``cfg.ref_respawn_offset_z`` (5 cm for SMPL) is added *only* to the
character's spawn root; the tracking-error reference uses the un-lifted
motion-local pose, matching ProtoMotions' convention.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Default paths (overridable on the CLI).
# ----------------------------------------------------------------------
DEFAULT_ONNX = str(
    Path(__file__).resolve().parent.parent / "models" / "model.onnx"
)
DEFAULT_MOTION_FILE = (
    "C:/Git/ProtoMotions/data/yaml_files/ACCAD/body_jab_left.motion"
)


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ppp-inference",
        description=(
            "Run the ProtoMotions SMPL motion-tracker policy on "
            "ovphysx + ovrtx using the unified-pipeline ONNX export."
        ),
    )
    p.add_argument(
        "--onnx",
        default=DEFAULT_ONNX,
        help=(
            "Path to the unified-pipeline ``model.onnx``. The matching "
            "YAML sidecar (joint / body schema, PD gains, timing, future "
            "step indices) is auto-discovered next to it; use --yaml to "
            "override."
        ),
    )
    p.add_argument(
        "--yaml",
        default=None,
        help=(
            "Optional path to the unified-pipeline YAML sidecar. When "
            "omitted, PPP looks for <onnx_stem>.yaml, then "
            "unified_pipeline.yaml, then any single *.yaml in the same "
            "directory as --onnx."
        ),
    )
    p.add_argument(
        "--motion-file",
        default=DEFAULT_MOTION_FILE,
        help="Path to .motion file to play back.",
    )
    p.add_argument(
        "--runtime-usd",
        default=str(Path(__file__).resolve().parent.parent / "_runtime_scene.usda"),
        help="Where to write the composed runtime USD scene.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for torch math (motion lib) and onnxruntime.",
    )
    p.add_argument(
        "--gpu-physics",
        action="store_true",
        help="Use GPU mode for ovphysx (otherwise CPU).",
    )
    p.add_argument(
        "--no-render",
        action="store_true",
        help="Skip ovrtx rendering (useful for smoke tests without RTX driver).",
    )
    p.add_argument(
        "--window",
        default="1280x720",
        help="Pygame window size, formatted as WxH.",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="If >0, stop after this many sim-seconds.",
    )
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help=(
            "Sim seconds per wall second. Each outer iter we accumulate "
            "``wall_dt * --time-scale`` and drain it in fixed dt_policy "
            "chunks (the trained control cadence); see InferenceApp.run. "
            "1.0 = real-time playback (default), 0.5 = slow motion, "
            "2.0 = fast forward. 0 disables wall-pacing and runs exactly "
            "one policy tick per outer iter."
        ),
    )
    p.add_argument(
        "--tracking-log-every",
        type=int,
        default=30,
        help=(
            "Log mean per-body tracking error every N policy ticks (default "
            "30 = once per simulated second at 30 Hz). Set to 0 to disable."
        ),
    )
    p.add_argument(
        "--draw-mimic-pose",
        action="store_true",
        help=(
            "Overlay the motion-reference body positions as bright spheres "
            "in the renderer window. Useful for eyeballing where the policy "
            "thinks each joint should be vs where PhysX actually puts it "
            "(this is the same quantity the tracking-error metric measures)."
        ),
    )
    p.add_argument(
        "--control-dt",
        type=float,
        default=None,
        help=(
            "Override the per-policy-tick wall time, in seconds. Falls "
            "back to the YAML's ``timing.control_dt`` when unset. The "
            "trained control cadence is hard-baked into the policy via "
            "the ``historical_actions`` rotation rate and the "
            "``mimic_future_*`` look-ahead, so PPP *must* tick at the "
            "exact cadence the policy was trained at — if the exported "
            "YAML lies (e.g. exporter hardcoded MuJoCo's 50 Hz when "
            "training was actually 30 Hz on IsaacLab), this flag is the "
            "knob to correct it without re-exporting the model. The "
            "physics substep count is derived from ``control_dt / "
            "physics_dt`` so this also changes how many physics "
            "substeps run per policy tick."
        ),
    )
    p.add_argument(
        "--physics-dt",
        type=float,
        default=None,
        help=(
            "Override the inner PhysX substep size, in seconds. Falls "
            "back to the YAML's ``timing.physics_dt`` when unset. "
            "Independent of ``--control-dt``; the world auto-substeps "
            "by ``ceil(control_dt / physics_dt)``."
        ),
    )
    p.add_argument(
        "--drive-type",
        type=str,
        choices=("force", "acceleration"),
        default=None,
        help=(
            "Override the per-joint USD drive token authored into the "
            "runtime scene. Defaults to ``cfg.drive_type`` (currently "
            "``\"force\"``, matching what the trainer's "
            "``ImplicitActuatorCfg`` actually runs against — see "
            "``scripts/config_loader.py`` docstring). Use "
            "``--drive-type acceleration`` to A/B against the old "
            "mass-normalised mode if you suspect a regression; expect "
            "instability under bypass-policy because the SMPL "
            "checkpoint was not trained for that effective torque."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _parse_window(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid --window value: {s!r}")


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
class InferenceApp:
    def __init__(self, args: argparse.Namespace) -> None:
        from .assets import (
            ROBOT_BODIES_ROOT,
            body_prim_paths,
            mimic_target_prim_paths,
            per_joint_gains_from_config,
            write_runtime_scene,
        )
        from .config_loader import load_config_for_onnx
        from .motion import MotionPlayer
        from .obs_builder import ObsBuilder
        from .policy import OnnxPolicy

        self.args = args

        self.cfg = load_config_for_onnx(args.onnx, explicit_yaml=args.yaml)

        # Apply CLI timing overrides *before* any consumer of cfg
        # (``PhysxWorld(dt_physics=...)`` and ``self.dt_policy``) reads
        # them. The trainer's exporter has historically baked the
        # MuJoCo deployment cadence (50 Hz / 1 kHz) into the YAML even
        # when the policy was actually trained against a 30 Hz / 600 Hz
        # IsaacLab loop, and the resulting cadence mismatch silently
        # breaks ``historical_actions`` / ``mimic_future_*`` semantics
        # in a way that looks like a balance-policy failure. We surface
        # both knobs so the user can correct the YAML at run time
        # without re-exporting.
        if args.control_dt is not None:
            if args.control_dt <= 0.0:
                raise ValueError(
                    f"--control-dt must be positive, got {args.control_dt}"
                )
            old = self.cfg.control_dt
            self.cfg.control_dt = float(args.control_dt)
            log.warning(
                "CLI override: control_dt %.6fs (%.1f Hz) -> %.6fs (%.1f Hz).",
                old, 1.0 / old, self.cfg.control_dt, self.cfg.policy_fps,
            )
        if args.physics_dt is not None:
            if args.physics_dt <= 0.0:
                raise ValueError(
                    f"--physics-dt must be positive, got {args.physics_dt}"
                )
            old = self.cfg.physics_dt
            self.cfg.physics_dt = float(args.physics_dt)
            log.warning(
                "CLI override: physics_dt %.6fs (%.1f Hz) -> %.6fs (%.1f Hz).",
                old, 1.0 / old, self.cfg.physics_dt, self.cfg.physics_fps,
            )
        if args.drive_type is not None:
            old = self.cfg.drive_type
            self.cfg.drive_type = str(args.drive_type)
            log.warning(
                "CLI override: drive_type %r -> %r (re-authoring USD drives).",
                old, self.cfg.drive_type,
            )

        # Sanity: warn if control_dt isn't an integer multiple of
        # physics_dt — the world will round substep count to the
        # nearest int, so a non-integer ratio means each tick covers
        # slightly more/less than ``control_dt`` of sim time. This is
        # usually benign but worth surfacing so a bad combo isn't
        # silent.
        ratio = self.cfg.control_dt / self.cfg.physics_dt
        if abs(round(ratio) - ratio) > 1e-3:
            log.warning(
                "control_dt / physics_dt = %.4f is not (near) integer; "
                "PhysxWorld.step will round substep count and the "
                "effective per-tick sim time will drift from "
                "control_dt by %.6fs.",
                ratio,
                abs(round(ratio) - ratio) * self.cfg.physics_dt,
            )

        log.info(
            "Robot: %d bodies, %d DOFs (policy fps=%.1f, future indices=%s).",
            self.cfg.num_bodies,
            self.cfg.num_dofs,
            self.cfg.policy_fps,
            self.cfg.future_step_indices,
        )

        # Pump per-DOF PD gains and the ``cfg.drive_type`` token
        # (default ``"force"``, matching the trainer's effective PD —
        # see ``scripts/config_loader.py``) into the runtime USD so
        # the ovphysx joints use the same gains and drive mode the
        # policy was trained against. With the unified-pipeline
        # export the policy *can* emit per-step ``stiffness_targets``
        # / ``damping_targets`` outputs, but PPP keeps the gains
        # static (USD-authored, matched to the YAML defaults) and
        # only logs a warning if the policy ever drifts from them —
        # see ``_check_pd_drift`` below.
        joint_gains = per_joint_gains_from_config(
            dof_names=self.cfg.dof_names,
            pd_stiffness=self.cfg.pd_stiffness,
            pd_damping=self.cfg.pd_damping,
            pd_effort_limit=self.cfg.pd_effort_limit,
            pd_velocity_limit=self.cfg.pd_velocity_limit,
            drive_type=self.cfg.drive_type,
        )
        scene = write_runtime_scene(
            args.runtime_usd,
            mimic_target_body_names=(
                self.cfg.body_names if args.draw_mimic_pose else None
            ),
            joint_gains=joint_gains,
        )
        log.info("Runtime USD written: %s", scene.runtime_usd)

        self.body_paths = body_prim_paths(self.cfg.body_names)
        self.mimic_target_paths = (
            mimic_target_prim_paths(self.cfg.body_names)
            if args.draw_mimic_pose
            else None
        )
        self.articulation_root_path = f"{ROBOT_BODIES_ROOT}/{self.cfg.body_names[0]}"

        from .physx_world import PhysxWorld
        from .robot_chain import RobotChain

        # Body-velocity helper. ovphysx's
        # ``ARTICULATION_LINK_VELOCITY`` for non-root bodies diverges
        # from the trainer's IsaacLab/IsaacGym body-velocity
        # convention on identical state. PPP finite-differences the
        # link pose across the last physics substep and applies the
        # COM correction to recover that convention bit-for-bit.
        robot_chain = RobotChain.from_body_names(self.cfg.body_names)

        self.world = PhysxWorld(
            usd_path=scene.runtime_usd,
            articulation_root_path=self.articulation_root_path,
            num_bodies=self.cfg.num_bodies,
            num_dofs=self.cfg.num_dofs,
            dt_physics=self.cfg.dt_physics,
            use_gpu=args.gpu_physics,
            expected_body_names=self.cfg.body_names,
            expected_dof_names=self.cfg.dof_names,
            robot_chain=robot_chain,
        )

        # MotionPlayer returns motion-local body positions (no
        # ``ref_respawn_offset_z`` lift). We spawn at the motion's
        # frame-0 pelvis XY/Z so the trainer's
        # ``get_spawn_to_ref_pose_offset_with_terrain_height_correction``
        # collapses to the identity offset and the raw motion frames
        # can be fed straight into ``mimic.future_*`` (the policy was
        # trained against the same un-translated frames in this
        # alignment).
        self.motion = MotionPlayer(
            args.motion_file,
            device=args.device,
        )
        self.dt_policy = float(self.cfg.control_dt)
        log.info("dt_policy = %.4fs (%.1f Hz).", self.dt_policy, self.cfg.policy_fps)

        self.obs_builder = ObsBuilder(self.cfg)
        self.policy = OnnxPolicy(args.onnx)

        # Cache the per-DOF default gains as float32 numpy arrays so
        # the ``stiffness_targets`` / ``damping_targets`` drift check
        # in ``policy_tick`` doesn't reallocate every step.
        self._default_stiffness = np.asarray(self.cfg.pd_stiffness, dtype=np.float32)
        self._default_damping = np.asarray(self.cfg.pd_damping, dtype=np.float32)
        self._warned_stiffness_drift = False
        self._warned_damping_drift = False

        # Renderer (optional). ``ovrtx`` and ``ovphysx`` cannot coexist in
        # the same Python process — see ppp/remote_renderer.py for the gory
        # details. The ``RemoteViewer`` spawns a child Python interpreter
        # that owns ovrtx + pygame and never touches ovphysx; the parent
        # ships per-tick body transforms to it over a multiprocessing queue.
        self.viewer = None
        if args.no_render:
            log.info("Renderer disabled by --no-render.")
        else:
            try:
                from .remote_renderer import RemoteViewer
                w, h = _parse_window(args.window)
                log.info(
                    "Spawning ovrtx renderer subprocess (first run compiles "
                    "shaders — this can take 30-90s)."
                )
                self.viewer = RemoteViewer(
                    usd_path=scene.runtime_usd,
                    body_prim_paths=self.body_paths,
                    window_size=(w, h),
                    log_level=args.log_level,
                    mimic_target_prim_paths=self.mimic_target_paths,
                )
            except Exception as e:
                log.exception(
                    "Remote renderer disabled (%s). If ovrtx is missing, install with:\n"
                    "  pip install https://pypi.nvidia.com/ovrtx/ovrtx-0.3.0.312915-py3-none-win_amd64.whl",
                    e,
                )
                self.viewer = None

        self.t = 0.0
        self._running = True
        self._step_count = 0
        self._tracking_err_sum = 0.0
        self._tracking_err_count = 0

    # ------------------------------------------------------------------
    def _motion_frame_to_numpy(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pull a single motion state at ``t`` and convert to numpy arrays.

        Returns ``(root_pos, root_rot, root_lin_vel, root_ang_vel, dof_pos, dof_vel)``.
        """
        sample = self.motion.get_state(t)
        body_pos = sample.rigid_body_pos[0, 0].detach().cpu().numpy()
        body_rot = sample.rigid_body_rot[0, 0].detach().cpu().numpy()
        body_lin = sample.rigid_body_vel[0, 0].detach().cpu().numpy()
        body_ang = sample.rigid_body_ang_vel[0, 0].detach().cpu().numpy()
        dof_pos = sample.dof_pos[0].detach().cpu().numpy()
        dof_vel = sample.dof_vel[0].detach().cpu().numpy()
        return body_pos, body_rot, body_lin, body_ang, dof_pos, dof_vel

    def reset(self) -> None:
        """Reset to motion frame 0 and zero the action history.

        Spawns the articulation at ``motion[t=0]`` plus the
        ``ref_respawn_offset_z`` lift on Z (the 5 cm clearance training
        uses to keep the character from instantiating clipped into the
        floor). The lift is added *only* here, not to the obs ref pose
        or the tracking reference — ProtoMotions does the same. DOF
        positions AND velocities are taken from the motion so walking
        motions start with the correct stride momentum.
        """
        log.info("Reset: teleporting articulation to motion[t=0].")
        self.t = 0.0
        self.obs_builder.reset_history()
        self._tracking_err_sum = 0.0
        self._tracking_err_count = 0

        root_pos, root_rot, root_lin, root_ang, dof_pos, dof_vel = (
            self._motion_frame_to_numpy(0.0)
        )

        # Lift the spawn root by ``ref_respawn_offset_z`` (5 cm for the
        # SMPL motion-tracker); see ProtoMotions
        # ``_compute_respawn_offset`` where the same value is added to
        # ``respawn_root_offset[:, 2]`` on top of the terrain height.
        root_pos = root_pos.copy()
        root_pos[2] += float(self.cfg.ref_respawn_offset_z)

        self.world.set_root_state(
            pos=root_pos,
            quat_xyzw=root_rot,
            lin_vel=root_lin,
            ang_vel=root_ang,
        )
        # Pass DOF velocities from the motion. ``set_dof_positions``
        # zeros velocities when the second arg is omitted, which on
        # locomotion motions silently kills the legs' stride momentum at
        # frame 0.
        self.world.set_dof_positions(dof_pos, dof_vel)
        # Hold the new pose with zero targets for one physics substep
        # so the solver converges on the teleported state before the
        # policy takes over.
        self.world.set_dof_targets(dof_pos)
        self.world.step(self.world.dt_physics)

    # ------------------------------------------------------------------
    def policy_tick(self) -> None:
        """One full unified-pipeline tick at ``self.dt_policy`` seconds.

        Pipeline:

        1. ``state = world.read_state()``  (PhysX -> numpy snapshot).
        2. ``future = motion.get_future(t, dt_policy, step_indices=...)``.
        3. ``feed = obs_builder.build(state, future, ground_height)``.
        4. ``out = policy.run(feed)`` (ONNX forward pass; returns
           ``actions``, ``joint_pos_targets``, optional stiffness /
           damping targets).
        5. ``obs_builder.push_action(out.actions)`` (history buffer
           stores raw ``actions`` for next step's
           ``historical_actions`` input).
        6. ``world.set_dof_targets(out.joint_pos_targets)``.
        7. ``world.step(dt_policy)``.
        8. Push body transforms to the renderer.
        9. ``t += dt_policy``; on motion end, ``reset()``.

        The trained control cadence is encoded in ``control_dt``
        (read from the YAML's ``timing.control_dt``); the
        ``historical_actions`` rotation, the ``mimic_future_*``
        look-ahead, and the PD target hold-time are all locked to it.
        """
        dt = self.dt_policy

        state = self.world.read_state()
        future = self.motion.get_future(
            self.t,
            dt,
            step_indices=self.cfg.future_step_indices,
        )
        feed = self.obs_builder.build(state, future, ground_height=0.0)

        out = self.policy.run(feed)
        self.obs_builder.push_action(out.actions)

        if not hasattr(self, "_logged_action_stats"):
            log.info(
                "Action stats: actions range=[%.3f, %.3f] (abs.mean=%.3f), "
                "targets range=[%.3f, %.3f] (abs.mean=%.3f)",
                float(np.min(out.actions)),
                float(np.max(out.actions)),
                float(np.mean(np.abs(out.actions))),
                float(np.min(out.joint_pos_targets)),
                float(np.max(out.joint_pos_targets)),
                float(np.mean(np.abs(out.joint_pos_targets))),
            )
            self._logged_action_stats = True

        # The policy *can* emit per-step ``stiffness_targets`` /
        # ``damping_targets`` (the unified-pipeline export wires the
        # control config's per-DOF gains into separate ONNX outputs).
        # PPP keeps the gains static (USD-authored from the YAML
        # defaults at startup) because ovphysx has no per-step
        # PD-gain binding; we just log a warning the first time the
        # policy drifts from the defaults so the operator knows.
        self._check_pd_drift(out)

        self.world.set_dof_targets(out.joint_pos_targets[0])
        self.world.step(dt)

        # Motion-reference body positions at the current tick — used for
        # the tracking-error diagnostic. ``self.t`` hasn't been
        # incremented yet, matching the input frame we built above.
        ref_body_pos = self._motion_ref_body_pos(self.t)
        self._accumulate_tracking_error(state.body_pos, ref_body_pos)

        if self.viewer is not None:
            self.viewer.push_body_transforms(state.body_pos, state.body_rot)
            self.viewer.update_follow_camera(state.root_pos)
            if self.mimic_target_paths is not None:
                self.viewer.push_mimic_target_positions(ref_body_pos)

        self.t += dt
        if self.t >= self.motion.length:
            log.info("Motion finished, looping.")
            self.reset()
        self._step_count += 1
        self._maybe_log_tracking_error()

    # ------------------------------------------------------------------
    def _check_pd_drift(self, out) -> None:
        """Warn (once) if the policy's stiffness/damping diverge from defaults.

        With the unified-pipeline export the actor doesn't actually
        change the PD gains for a tracker config — the
        ``ActionExportModule`` just rebroadcasts ``control_info`` into
        the outputs. But if a future training run switches to a
        learned-gain head, PPP's USD-authored static gains would
        silently override the policy's intent. Catching that early
        beats debugging it via tracking error.
        """
        if out.stiffness_targets is not None and not self._warned_stiffness_drift:
            diff = float(
                np.max(np.abs(out.stiffness_targets[0] - self._default_stiffness))
            )
            if diff > 1e-3:
                log.warning(
                    "Policy stiffness_targets drift from USD-authored "
                    "defaults: max |delta| = %.4f. PPP keeps gains static; "
                    "ignore if expected, otherwise the policy may need "
                    "per-step PD updates wired through ovphysx.",
                    diff,
                )
                self._warned_stiffness_drift = True
        if out.damping_targets is not None and not self._warned_damping_drift:
            diff = float(
                np.max(np.abs(out.damping_targets[0] - self._default_damping))
            )
            if diff > 1e-3:
                log.warning(
                    "Policy damping_targets drift from USD-authored "
                    "defaults: max |delta| = %.4f.",
                    diff,
                )
                self._warned_damping_drift = True

    # ------------------------------------------------------------------
    def _motion_ref_body_pos(self, t: float) -> np.ndarray:
        """Per-body world positions of the motion reference at ``t``.

        Returns the *un-lifted* motion-local positions — the same frame
        the policy sees as ``mimic_future_pos``. The 0.05 m
        ``ref_respawn_offset_z`` is **not** applied here; ProtoMotions
        also leaves it out of the mimic targets and the gt-error
        metric (only the character's spawn root is lifted).

        Returned as a ``(num_bodies, 3)`` numpy float32 array.
        """
        return self.motion.get_state(t).rigid_body_pos[0].detach().cpu().numpy().copy()

    def _accumulate_tracking_error(
        self,
        sim_body_pos: np.ndarray,
        ref_body_pos: np.ndarray,
    ) -> None:
        """Update running tracking error using the motion ref at ``self.t``.

        Tracking error here is *per-body L2 distance, averaged over bodies*
        — the same quantity ProtoMotions logs as ``gt_err``. Computing it
        in world coordinates (no heading alignment) gives a strict upper
        bound; if the heading drifts the value goes up. For SMPL a healthy
        tracker runs in the 0.05-0.15 m range; >0.5 m means the agent is
        either falling over or off-policy.
        """
        err = float(np.linalg.norm(sim_body_pos - ref_body_pos, axis=-1).mean())
        self._tracking_err_sum += err
        self._tracking_err_count += 1

    def _maybe_log_tracking_error(self) -> None:
        every = self.args.tracking_log_every
        if every <= 0 or self._tracking_err_count < every:
            return
        avg = self._tracking_err_sum / self._tracking_err_count
        log.info(
            "Tracking err (mean per-body L2, last %d ticks): %.3f m  [t=%.2fs / %.2fs]",
            self._tracking_err_count,
            avg,
            self.t,
            self.motion.length,
        )
        self._tracking_err_sum = 0.0
        self._tracking_err_count = 0

    # ------------------------------------------------------------------
    def run(self) -> int:
        """Drive the policy loop with a fixed control-rate cadence.

        The trained policy is locked to ``self.dt_policy`` (read from
        the YAML's ``timing.control_dt``): ``historical_actions``,
        ``mimic_future_*`` look-ahead, time-to-target, and the PD
        target hold time all encode that interval. Calling
        ``policy_tick()`` at variable wall cadence drives the buffers
        and look-ahead far off-distribution.

        Instead of stepping by wall ``dt``, we accumulate wall-derived
        sim time and drain it in ``dt_policy``-sized chunks:

        1. Measure ``wall_dt = perf_counter() - last_iter``.
        2. ``sim_time_accum += wall_dt * --time-scale`` (clamped per
           iter so a stall can't dump 50 ticks on resume).
        3. ``while sim_time_accum >= dt_policy``: ``policy_tick()``;
           ``sim_time_accum -= dt_policy``.
        4. ``viewer.render(...)`` and ``viewer.poll_events()`` once per
           outer iter so the renderer / input stays responsive even
           when no policy tick fires that iter.
        """
        time_scale = float(self.args.time_scale)
        wall_driven = time_scale > 0.0
        max_iter_sim_dt = 5.0 * self.dt_policy
        max_ticks_per_iter = 5

        if wall_driven:
            log.info(
                "Time scaling: %.3f sim-sec per wall-sec (fixed policy "
                "cadence %.1f Hz; per-iter sim-dt clamp %.1f ms; up to "
                "%d ticks per outer iter).",
                time_scale,
                1.0 / self.dt_policy,
                max_iter_sim_dt * 1000.0,
                max_ticks_per_iter,
            )
        else:
            log.info(
                "Time scaling disabled (--time-scale 0) — exactly one "
                "policy tick of %.1f ms per outer iter, regardless of "
                "wall clock.",
                self.dt_policy * 1000.0,
            )

        self.reset()
        start = time.perf_counter()
        last_iter_wall = start
        sim_time_accum = 0.0

        log.info("Starting policy loop. Press Q to quit, R to reset, O to toggle follow.")
        try:
            while self._running:
                now = time.perf_counter()
                wall_dt = now - last_iter_wall
                last_iter_wall = now

                if wall_driven:
                    sim_time_accum += min(wall_dt * time_scale, max_iter_sim_dt)
                else:
                    sim_time_accum = self.dt_policy

                ticks_this_iter = 0
                while (
                    sim_time_accum >= self.dt_policy
                    and ticks_this_iter < max_ticks_per_iter
                ):
                    self.policy_tick()
                    sim_time_accum -= self.dt_policy
                    ticks_this_iter += 1

                if ticks_this_iter >= max_ticks_per_iter:
                    sim_time_accum = 0.0

                if self.viewer is not None:
                    self.viewer.render(self.dt_policy)
                    actions = self.viewer.poll_events()
                    if actions["quit"]:
                        self._running = False
                    if actions["reset"]:
                        self.reset()
                        sim_time_accum = 0.0
                        last_iter_wall = time.perf_counter()

                if (
                    self.args.max_seconds > 0
                    and (now - start) >= self.args.max_seconds
                ):
                    log.info("--max-seconds reached, stopping.")
                    break
        except KeyboardInterrupt:
            log.info("Interrupted.")
        finally:
            self.close()
        return 0

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        try:
            self.world.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
def cli_main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    required_paths = [("--onnx", args.onnx), ("--motion-file", args.motion_file)]
    if args.yaml is not None:
        required_paths.append(("--yaml", args.yaml))
    for label, p in required_paths:
        if not Path(p).exists():
            log.error("%s does not exist: %s", label, p)
            return 2

    app = InferenceApp(args)
    return app.run()


if __name__ == "__main__":
    sys.exit(cli_main())
