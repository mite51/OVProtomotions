"""Top-level application: compose the scene, drive the policy loop, render.

Per *policy tick* (always exactly ``dt = self.dt_policy = 1/30 s`` — the
trained cadence — independent of how busy the wall loop is):

1. ``state = world.read_state()``                  (read PhysX)
2. ``future = motion.get_future(t, dt_policy, n=1)``  (motion-local ref)
3. ``obs = obs_builder.build(state, future, ...)`` (3 ONNX inputs)
4. ``raw_mu, tanh_mu = policy.run_action_outputs(obs)``  (pre/post tanh)
5. ``obs_builder.push_action(tanh_mu)``           (history stores the
   post-tanh ``mean_action``; obs reads it back with a 2-step lag, matching
   protomotions' ``state_history_buffer`` rotate-then-obs ordering)
6. ``targets = action_proc.process(tanh_mu)``     (pi * tanh(mu) for SMPL)
7. ``world.set_dof_targets(targets); world.step(dt_policy)``  (auto-substeps)
8. ``viewer.push_body_transforms(state.body_pos, state.body_rot)``
9. ``t += dt_policy``; on motion end, reset to ``t=0``.

The outer ``run()`` loop turns variable wall ``dt`` into a constant
policy cadence with a simple accumulator: each iter, ``sim_time_accum
+= wall_dt * --time-scale`` (capped), then drain the accumulator in
``dt_policy``-sized chunks by calling ``policy_tick()``. ``viewer.render``
+ ``viewer.poll_events`` run once per outer iter so the renderer and
keyboard stay responsive even when no policy tick fires.

Reset writes the articulation root pose and DOF positions/velocities to
the motion's first frame and zeros the action history. The
``cfg.ref_respawn_offset_z`` (5 cm for SMPL) is added *only* to the
character's spawn root; the tracking-error reference uses the un-lifted
motion-local pose, matching protomotions' convention (its
``get_spawn_to_ref_pose_offset_with_terrain_height_correction`` returns
the spawn-XY translation plus terrain height, but never the +0.05 m
character lift).
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
DEFAULT_CHECKPOINT = (
    "C:/Git/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt"
)
DEFAULT_MOTION_FILE = (
    "C:/Git/ProtoMotions/data/yaml_files/ACCAD/body_jab_left.motion"
)
DEFAULT_ONNX = (
    "C:/Git/ProtoMotions/onnx/smpl_policy_orig/model.onnx"
)


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ppp-inference",
        description=(
            "Run the ProtoMotions SMPL motion-tracker policy on ovphysx + ovrtx."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to last.ckpt (used to find resolved_configs_inference.pt).",
    )
    p.add_argument(
        "--motion-file",
        default=DEFAULT_MOTION_FILE,
        help="Path to .motion file to play back.",
    )
    p.add_argument(
        "--onnx",
        default=DEFAULT_ONNX,
        help="Path to model.onnx (defaults to smpl_policy_orig/model.onnx).",
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
        help="Device used for torch math (obs builders) and onnxruntime.",
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
            "chunks (the trained 30 Hz cadence); see InferenceApp.run. "
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
        from .config_loader import load_resolved_config
        from .motion import MotionPlayer
        from .obs_builder import ObsBuilder
        from .policy import OnnxPolicy
        from .action import ActionConfig, ActionProcessor

        self.args = args

        self.cfg = load_resolved_config(args.checkpoint)
        log.info(
            "Robot: %d bodies, %d DOFs (policy fps=%.1f).",
            self.cfg.num_bodies,
            self.cfg.num_dofs,
            self.cfg.policy_fps,
        )

        # Pump per-DOF PD gains and the IsaacLab-style ``"acceleration"``
        # drive type from the resolved checkpoint into the runtime USD,
        # instead of inheriting the upstream ``smpl_humanoid.usda``
        # ``"force"`` drives. Same numerical kp/kd, but
        # mass-normalisation matches the trainer's ``ImplicitActuator``.
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
        # convention on identical state. The trainer reports body
        # velocities at each body's center of mass, while ``body_pos``
        # is at the link origin — see ``robot_chain.py``. PPP
        # finite-differences the link pose across the last physics
        # substep and applies the COM correction to recover that
        # convention bit-for-bit.
        robot_chain = RobotChain.from_body_names(self.cfg.body_names)

        # ``cfg.dt_physics`` is ``1 / (fps * substeps)`` so we hit the
        # same per-substep solver rate the policy was trained at on
        # both the IsaacLab path (fps=120, substeps=1) and the IsaacGym
        # path (fps=60, substeps=2) — see ``config_loader.py``.
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
        # ``ref_respawn_offset_z`` lift). This matches what protomotions
        # feeds the obs builder via
        # ``get_spawn_to_ref_pose_offset_with_terrain_height_correction``,
        # which on flat ground returns ``(respawn_root_offset.xy,
        # terrain_z)`` — the spawn-XY translation plus terrain height,
        # but NOT the 0.05 m character lift. The 0.05 m only applies to
        # the *character's spawn position* at reset (see ``reset()``),
        # never to the tracking reference.
        self.motion = MotionPlayer(
            args.motion_file,
            device=args.device,
        )
        self.dt_policy = 1.0 / self.cfg.policy_fps
        log.info("dt_policy = %.4fs (%.1f Hz).", self.dt_policy, self.cfg.policy_fps)

        self.obs_builder = ObsBuilder(
            num_bodies=self.cfg.num_bodies,
            num_dofs=self.cfg.num_dofs,
            action_dim=self.cfg.num_dofs,
            device=args.device,
            future_steps=self.cfg.future_steps,
        )

        self.policy = OnnxPolicy(args.onnx)

        action_cfg = ActionConfig(
            pd_action_offset=self.cfg.pd_action_offset,
            pd_action_scale=self.cfg.pd_action_scale,
            apply_tanh=(self.cfg.action_transform != "tanh"),
            clamp_value=self.cfg.clamp_value,
        )
        self.action_proc = ActionProcessor(action_cfg)

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
        or the tracking reference — protomotions does the same. DOF
        positions AND velocities are taken from the motion so walking
        motions start with the correct stride momentum; previously the
        DOF velocities were silently zeroed, putting the very first
        policy obs out of distribution and biasing the gait toward
        in-place stepping (visible on Walk_B4 as 7+ m of Y drift).

        Action history convention. ``reset_history()`` zeroes both
        slots of the action history buffer. This intentionally matches
        what the trainer does on a clean env reset (see
        ``protomotions/envs/base_env/env.py::_reset_state_history`` —
        ``actions=None`` zeros every action slot in
        :class:`StateHistoryBuffer`).
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
        # SMPL motion-tracker); see protomotions
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
        # policy takes over. We step a single substep-sized dt (= the
        # configured ``dt_physics``); since the character is held by
        # the teleported targets this isn't sensitive to the exact
        # value, we just need a small > 0 dt to settle the solver.
        self.world.set_dof_targets(dof_pos)
        self.world.step(self.world.dt_physics)

    # ------------------------------------------------------------------
    def policy_tick(self) -> None:
        """One full policy tick at exactly ``self.dt_policy`` seconds.

        Pipeline (everything below uses ``dt = self.dt_policy`` — the
        trained 1/30 s cadence — independent of how often ``run()`` calls
        this method or how busy the wall loop is):

        1. ``state = world.read_state()``  (PhysX -> numpy snapshot).
        2. ``future = motion.get_future(t, dt_policy, n=future_steps)``.
        3. ``obs = obs_builder.build(state, future, future_dt=dt_policy)``.
        4. ``raw_mu, tanh_mu = policy.run_action_outputs(obs)``.
        5. ``obs_builder.push_action(tanh_mu)``  (history stores the
           post-tanh ``mean_action`` and the obs reads it back with a
           2-step lag — see ``ObsBuilder.push_action`` for the full
           contract; matches ``state_history_buffer.rotate_and_update``
           followed by ``get_obs`` in protomotions ``base_env/env.py``).
        6. ``targets = action_proc.process(tanh_mu)``  (pi * tanh(mu)).
        7. ``world.set_dof_targets(targets); world.step(dt_policy)``.
        8. Push body transforms to the renderer for the next frame.
        9. ``t += dt_policy``; on motion end, ``reset()``.

        Fixed-cadence ticking matters because the trained policy's
        ``historical_previous_actions`` buffer, the ``mimic_target_poses``
        look-ahead distance, and the PD target hold-time are all encoded
        for 1/30 s. Earlier versions of this loop used the wall-driven
        ``sim_dt`` here, which at high outer-loop rates fed the policy a
        future ~ 5 ms ahead (target body deltas collapse toward zero) and
        rotated the previous-action buffer every few ms instead of every
        33 ms — far off-distribution. See ``run()`` for the accumulator
        that turns variable wall ``dt`` into a constant policy cadence.
        """
        dt = self.dt_policy

        state = self.world.read_state()
        future = self.motion.get_future(self.t, dt, n=self.cfg.future_steps)
        obs = self.obs_builder.build(
            state,
            future,
            motion_time=self.t,
            motion_length=self.motion.length,
            future_dt=dt,
        )

        # ProtoMotions inference passes ``mean_action`` (= ONNX ``tanh``
        # output = ``tanh(mu_model_output)``) directly to ``env.step``. The
        # state-history buffer therefore stores the *post-tanh* value, and
        # the PD target is ``pi * mean_action`` (no second tanh). Verified
        # against the captured ``debug_output.json``: ``actions`` field
        # equals the ONNX ``tanh`` output to fp32, ``pd_targets = pi *
        # actions`` to fp32, and ``historical_previous_actions[N] ==
        # actions[N-2]`` for every captured frame.
        #
        # ``raw_mu`` (the pre-tanh ``mu_model`` output, exposed via the
        # in-memory graph patch in ``OnnxPolicy``) is kept around for
        # diagnostic logging only — it is *not* fed into the obs.
        raw_mu, tanh_mu = self.policy.run_action_outputs(obs.as_dict())  # (1, 69) each
        self.obs_builder.push_action(tanh_mu)
        targets = self.action_proc.process(tanh_mu)  # (1, 69)

        if not hasattr(self, "_logged_action_stats"):
            log.info(
                "Action stats: raw_mu range=[%.3f, %.3f] (abs.mean=%.3f), "
                "targets range=[%.3f, %.3f] (abs.mean=%.3f)",
                float(np.min(raw_mu)),
                float(np.max(raw_mu)),
                float(np.mean(np.abs(raw_mu))),
                float(np.min(targets)),
                float(np.max(targets)),
                float(np.mean(np.abs(targets))),
            )
            self._logged_action_stats = True

        self.world.set_dof_targets(targets[0])
        self.world.step(dt)

        # Motion-reference body positions at the current tick — used for
        # the tracking-error diagnostic. ``self.t`` hasn't been
        # incremented yet, matching the obs frame we built above.
        ref_body_pos = self._motion_ref_body_pos(self.t)

        # Tracking-error diagnostic: mean L2 distance between sim body
        # positions and the motion reference at *this* tick. Computed
        # against the same ``self.t`` the obs were built for, so it's
        # directly comparable across runs/simulators.
        self._accumulate_tracking_error(state.body_pos, ref_body_pos)

        # Stage fresh transforms for the renderer (queue put is
        # non-blocking; the renderer subprocess drops stale frames on
        # overflow). The actual ``viewer.render`` call happens in
        # ``run()`` once per outer iter, so the renderer's framerate is
        # decoupled from the policy's tick rate.
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
    def _motion_ref_body_pos(self, t: float) -> np.ndarray:
        """Per-body world positions of the motion reference at ``t``.

        Returns the *un-lifted* motion-local positions — the same frame
        the obs builder feeds the policy as ``mimic_ref_pos``. The 0.05 m
        ``ref_respawn_offset_z`` is **not** applied here; protomotions
        also leaves it out of ``mimic_target_poses`` and the gt-error
        metric (only the character's spawn root is lifted). Exposing
        this same convention gives callers — the tracking-error metric
        and the ``--draw-mimic-pose`` overlay — apples-to-apples deltas
        against the policy's tracking target.

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
        """Drive the policy loop with a fixed 30 Hz policy cadence.

        The trained policy is locked to a 1/30 s tick (``self.dt_policy``):
        ``historical_previous_actions``, ``mimic_target_poses``
        look-ahead, time-to-target, and the PD-target hold time all
        encode that interval. Calling ``policy_tick()`` at variable wall
        cadence — as earlier versions did — drives the buffers and
        look-ahead far off-distribution and tracking collapses within
        a second.

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

        Pacing semantics:

        - ``--time-scale 1.0`` (default): real-time playback. At any
          wall rate the policy still ticks once per 33.3 ms of sim
          time; extra wall iters just feed events / render.
        - ``--time-scale 0.5`` / ``2.0``: slow-motion / fast-forward,
          still at fixed policy cadence.
        - ``--time-scale 0``: legacy "ignore wall clock" — exactly one
          policy tick per outer iter.
        """
        time_scale = float(self.args.time_scale)
        wall_driven = time_scale > 0.0
        # Cap per-iter accumulated sim time at 5 policy ticks (~165 ms).
        # Without this, a stall (debugger pause, GC, shader compile,
        # initial frame) would queue dozens of policy ticks on resume,
        # blowing through the motion timeline in one outer iter.
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

                # Drain the accumulator at the trained policy cadence.
                ticks_this_iter = 0
                while (
                    sim_time_accum >= self.dt_policy
                    and ticks_this_iter < max_ticks_per_iter
                ):
                    self.policy_tick()
                    sim_time_accum -= self.dt_policy
                    ticks_this_iter += 1

                # Discard any leftover ticks beyond the per-iter cap so
                # transient stalls don't snowball into a backlog.
                if ticks_this_iter >= max_ticks_per_iter:
                    sim_time_accum = 0.0

                # Render + event polling every iter, decoupled from the
                # policy tick rate. ``viewer.render`` is a no-op until at
                # least one ``push_body_transforms`` has staged data
                # (i.e. one policy tick has fired since reset).
                if self.viewer is not None:
                    self.viewer.render(self.dt_policy)
                    actions = self.viewer.poll_events()
                    if actions["quit"]:
                        self._running = False
                    if actions["reset"]:
                        self.reset()
                        sim_time_accum = 0.0
                        # After reset, the next iter's wall_dt would
                        # span all of `reset()`'s wall time and dump a
                        # giant chunk into the accumulator — re-baseline.
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

    for label, p in (
        ("--checkpoint", args.checkpoint),
        ("--motion-file", args.motion_file),
        ("--onnx", args.onnx),
    ):
        if not Path(p).exists():
            log.error("%s does not exist: %s", label, p)
            return 2

    app = InferenceApp(args)
    return app.run()


if __name__ == "__main__":
    sys.exit(cli_main())
