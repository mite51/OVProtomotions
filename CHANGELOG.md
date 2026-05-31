# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Restored the mimic-pose overlay under the new flag name
  `--draw-mimic-pose` (previously `--debug-mimic-pose`). When set,
  `ppp/app.py` re-authors the `/World/MimicTargets` scope in the
  runtime USD, plumbs prim paths through `RemoteViewer`/`RtxViewer`,
  and pushes per-tick motion-reference body positions to the
  spheres via `push_mimic_target_positions`. The flag is the only
  knob; all related plumbing (`MIMIC_TARGETS_ROOT`,
  `mimic_target_prim_paths`, the `_mimic_target_markers_block`
  USDA snippet) is back in place.

### Removed

- Stripped the project down to a single inference loop. Removed
  every diagnostic, replay-parity, and alternate-mode option so
  `run_inference.py` exposes only the flags actually needed for
  motion playback (`--checkpoint`, `--motion-file`, `--onnx`,
  `--runtime-usd`, `--device`, `--gpu-physics`, `--no-render`,
  `--window`, `--max-seconds`, `--time-scale`,
  `--tracking-log-every`, `--log-level`).
  - **CLI flags gone.** `--body-velocity-mode`,
    `--controller-order`, `--dump-diagnostics`, and every
    `--replay-*` flag (`--replay-debug-log`, `--replay-mode`,
    `--replay-action-source`, `--replay-frames`,
    `--replay-no-xy-shift`, `--replay-state-tol`,
    `--replay-obs-tol`, `--replay-action-tol`,
    `--replay-poststep-tol`, `--replay-inject-historical`,
    `--replay-teleport-mode`, `--replay-override-body-vel`,
    `--replay-direct-pd`, `--replay-trace-tick`).
    `--debug-mimic-pose` was renamed to `--draw-mimic-pose`
    (same behaviour; see the matching `Added` entry).
  - **Files deleted.** `ppp/debug_replay.py` (replay-log loader
    and diff aggregator), `scripts/smoke_obs.py` (offline ONNX
    smoke test), and the empty `_smoke_runtime_scene.usda`.
  - **`ppp/physx_world.py` trimmed.** Dropped the
    `body_velocity_mode` constructor arg and the experimental
    `fk_chain` branch (FD+COM stays as the only body-velocity
    backend, with the existing post-teleport
    `ARTICULATION_LINK_VELOCITY` fallback), the
    `commit_writes` method, the `direct_pd_available` capability
    flag, `set_dof_actuation_forces`,
    `apply_direct_pd_torques`, and the
    `ARTICULATION_DOF_ACTUATION_FORCE` binding plus its staging
    buffer.
  - **`ppp/robot_chain.py` trimmed.** Removed
    `SMPL_PARENT_INDICES_LOCAL`, `fk_body_velocities`,
    `remove_com_correction`, and the `parent_indices` field on
    `RobotChain` (only the FK backend consumed them).
  - **`ppp/app.py` trimmed.** `InferenceApp` lost the
    `_run_replay`, `_teleport_from_frame`, `_diff_state`,
    `_diff_obs`, `_trace_state_vs_frame`,
    `_rotate_vec_by_quat_xyzw`, and `_maybe_dump_diagnostics`
    methods, the `_replay`, `_controller_order`, and
    `_pending_targets` state, and the deferred-controller branches
    in `policy_tick`.

### Changed

- `_run_replay` in `ppp/app.py` now writes DOF targets *after*
  `world.step(...)` instead of before. The action computed at tick
  `N` therefore drives tick `N+1`'s integrator, matching trainers
  that record state post-step and apply action `N` on the next
  `env.step`. Tick 0 steps with whatever drive state the
  post-teleport articulation came up with — fine in open-loop
  mode (we re-teleport before the next step) but a real semantic
  shift in closed-loop replay; the post-step `read_state`
  diagnostic now compares "state after step `N` driven by action
  `N-1`" against `next_frame`.

### Added

- Mimic vs Unity-port parity toolbox (see the
  `mimic_inference_parity_fixes_2` plan). All four items land
  together as new opt-in tools; PPP's default code paths are
  unchanged so existing replay parity numbers stay valid.
  - **Experimental FK body-velocity backend.** New
    `--body-velocity-mode {fd_com,fk_chain}` CLI flag (default
    `fd_com` — the existing finite-difference + center-of-mass
    correction PPP relied on to work around ovphysx's broken
    `ARTICULATION_LINK_VELOCITY` binding for non-root bodies).
    `fk_chain` reconstructs per-body linear/angular velocities by
    propagating `root_vel` + `dof_vel` down the SMPL kinematic tree
    in `ppp/robot_chain.py::fk_body_velocities`, applying the
    per-body COM correction at the end so the output matches
    `ProtoMotions`' body-COM convention. Picked the FK path because
    it is independent of the previous-substep pose cache, so it
    survives teleports and reset-without-step. Switching backends
    on the same PhysX state shows micro-level disagreement at the
    post-reset zero-velocity pose (≤1e-5 m/s, ≤1e-5 rad/s for all
    24 bodies on the captured smpl asset) which is consistent with
    both methods being correct at quasi-static states.
  - **Direct-PD path via `ARTICULATION_DOF_ACTUATION_FORCE`.**
    `PhysxWorld` now probes ovphysx for the actuation-force tensor
    at init and exposes `direct_pd_available`, `set_dof_actuation_forces`,
    and `apply_direct_pd_torques(targets, kp, kd, effort_limits, …)`.
    The last method computes IsaacLab-style explicit-PD torques —
    `clip(kp*(target - dof_pos) - kd*dof_vel, ±effort)` — and writes
    them as raw joint torques. To avoid double-counting against the
    USD position-target drive, it also parks
    `ARTICULATION_DOF_POSITION_TARGET` at the current `dof_pos` so
    the drive's own PD error is zero. Wired into the replay harness
    behind `--replay-direct-pd` so users can A/B the explicit-PD
    path against the position-target drive on captured frames.
    Confirmed working against ovphysx 0.3.7: a single forward step
    with this path runs without errors and produces non-trivial
    torques (max 172 N·m on the SMPL asset at t=0). On older
    ovphysx builds that don't expose the actuation-force tensor,
    `direct_pd_available` is False and `--replay-direct-pd`
    transparently falls back to the position-target drive with a
    warning.
  - **Unity-style split infer/apply controller order.** New
    `--controller-order {default,deferred}` CLI flag. `default` is
    the existing PPP ordering (read → obs → policy → apply → step
    within a single tick, matching the IsaacLab trainer). `deferred`
    mirrors the Unity `SMPLInferenceAgent` pattern where
    `RunOnePolicyStep` and `ApplyActions` are split: PPP now applies
    the *previous* tick's PD targets and steps before reading the
    next state, so the observation reflects the post-step world.
    Motion-time advance moves to the apply phase so obs and the
    future-pose target stay aligned. The first deferred tick has no
    pending action and skips the apply+step phase entirely (the
    post-reset settle step already holds the joints).
    `InferenceApp.reset()` clears the pending-action buffer so
    looping a motion under deferred mode behaves like a fresh
    start.
  - **Compact startup diagnostics dump.** New `ppp/diagnostics.py`
    and `--dump-diagnostics PATH` CLI flag. After construction and
    the initial reset / replay teleport, PPP now optionally logs
    and JSON-dumps the answers to the four most-asked
    port-vs-trainer questions: (1) the full robot dim block
    (bodies / DOFs / policy fps / physics dt / future / history /
    drive type / action transform / clamp / ref lift), (2) body and
    DOF permutations between `kinematic_info` and what ovphysx
    reports, (3) the per-DOF gain/limit/scale table
    (canonical_idx → name → physx_idx → kp/kd/effort/vel_lim/scale/
    offset), and (4) action-unit ranges plus per-body FK-vs-FD
    velocity deltas on the live PhysX state. `PhysxWorld` now
    exposes `body_perm`, `dof_perm`, `body_velocity_mode`, and
    `direct_pd_available` as read-only properties so the dump can
    reflect *runtime* state, not just the resolved config.
    Diagnostics failures are logged but never abort the run.

- Mimic inference parity pass against the IsaacLab trainer (see the
  `mimic_inference_parity_fixes` plan). Four phases land together:
  - **Phase A — PD drive type.** `ppp/config_loader.py` now reads
    per-DOF `pd_stiffness` / `pd_damping` / `pd_effort_limit` /
    `pd_velocity_limit` from the resolved `robot.control.control_info`
    in the .pt blob (with a SMPL group-table fallback exposed via
    `_SMPL_GROUP_GAINS`). `ppp/assets.py::write_runtime_scene` accepts
    a `joint_gains: dict[str, JointGains]` table and emits the full
    `drive:rot{X,Y,Z}:physics:{type,stiffness,damping,maxForce}`
    block per joint; `app.py` builds the table via
    `per_joint_gains_from_config(...)` so the runtime USD's drive
    type is `"acceleration"` (mirrors IsaacLab's `ImplicitActuatorCfg`)
    instead of inheriting the upstream `"force"`. Net effect on the
    20-tick `commit + log-action` replay: `post_dof_pos` worst is
    unchanged at 0.371 rad — ovphysx silently treats the
    `acceleration` token as `force` in this build, so the change is
    captured but doesn't move the integrator. The USD authoring is
    correct against the schema and matches what the trainer would
    have authored, so it stays in.
  - **Phase B — Scene PhysX caps.** `physxScene:maxPositionIterationCount = 4`,
    `physxScene:maxVelocityIterationCount = 4`,
    `physxScene:bounceThreshold = 0.2` are now authored on the
    physicsScene (matches `sim.physx.num_position_iterations = 4`,
    `num_velocity_iterations = 4`, `bounce_threshold_velocity = 0.2`
    from the resolved checkpoint). `SmplBodyMaterial` friction was
    raised from 0.5 → 1.0 to match IsaacLab's default robot-shape
    friction. `GroundCollider:contactOffset` was raised from 0.001 m
    → 0.02 m to match IsaacLab's `sim.physx.contact_offset`; the
    earlier 0.001 m had been tuned to an IsaacGym-terrain-mesh
    OmniPVD capture that doesn't apply to this IsaacLab checkpoint.
    Net effect: replay `post_dof_pos` worst essentially unchanged
    (0.371 rad → 0.371 rad), live tracking error on
    `body_jab_left.motion` 0.287 m → 0.279 m (-3 %). ovphysx appears
    to clamp to per-articulation iteration caps and ignore the
    scene-level caps, but the authored values remain correct
    against the trainer config.
  - **Phase C — Frame-0 history seeding.** New
    `ObsBuilder.seed_history(slot1, slot0=None)` writes the captured
    `frame[0].historical_previous_actions` and (optionally)
    `frame[1].historical_previous_actions` into the action-history
    buffer. `_run_replay` calls it after `reset_history()` so the
    replay diagnostic reproduces the trainer's mid-episode warmup
    state across the first two ticks; `app.py.reset()` keeps the
    all-zero seed for live inference (which the trainer also uses
    on a clean env reset — the mid-episode capture is a debug
    artifact, not the steady-state behaviour). `_run_replay`'s
    per-tick `push_action` now respects `--replay-action-source`:
    when `=log`, the captured `frame.actions` is pushed instead of
    PPP's policy output so the buffer stays in lockstep with the
    trainer record (otherwise the tick-0 idx-213 obs residual got
    rotated into slot 1 every other tick and re-poisoned the obs).
    Net effect with `commit + log + override-body-vel + seed`:
    `obs.historical_previous_actions` 18/20 fails → 0/20,
    `policy_tanh_mu` 20/20 → 1/20, `pd_targets` 20/20 → 1/20. The
    remaining 1/20 is a frame-0 `obs_from_log` artifact (Phase D).
  - **Phase D — Frame-0 `body_rot_obs` artifact.** Diagnosed the
    residual `obs_from_log.max_coords_obs fails=1/20 worst_idx=213`
    as a trainer-side capture artifact in `debug_output.json`: at
    frame 0, the dumped `max_coords_obs` reflects the pre-reset
    sim's body-rot buffer (all 24 bodies' tan_norm collapses to
    `[1, 0, 0, 0, 0, 1]`, the identity), while the dumped
    `rigid_body_rot` field reflects the post-reset motion-ref
    pose. PPP's obs builder is bit-exact against the trainer math
    on frames 1+. No PPP code change required; the diagnostic now
    has a single known-flaky frame.

  Live closed-loop tracking error after the parity pass
  (`--max-seconds 5 --time-scale 0`, mean per-body L2):

  | motion                 | before | after  |
  |------------------------|--------|--------|
  | `body_jab_left.motion` | 0.287 m | 0.279 m |
  | `KB_WalkFwd1.motion`   | n/a    | 0.044 m |
  | `KB_Idle_1.motion`     | n/a    | 0.028 m |

  The `body_jab_left` punch motion is the worst-case (extreme R_Hand
  poses); two of the three sampled motions land well under the
  plan's 0.1 m acceptance bar. The remaining `body_jab_left` gap
  is dominated by deeper ovphysx-vs-IsaacLab integrator behaviour
  (joint armature handling, contact pipeline) outside the plan's
  scope.

### Fixed

- `PhysxWorld.read_state` now derives per-body world-frame velocities
  by finite-differencing the link pose across the last physics
  substep and then applying the COM correction
  `v_at_COM = v_at_L + ω × (R · com_offset_local)` instead of relying
  on `ARTICULATION_LINK_VELOCITY`. The trainer reports body
  velocities at each body's center of mass while `body_pos` is at
  the link origin (IsaacLab convention); the previous binding-based
  read disagreed on non-trivial chains and threw obs / policy / PD
  out of distribution (deepest links drifted to ~12 m/s and
  ~32 rad/s by tick 1 of `body_jab_left.motion`). The COM offset
  table for SMPL's 24 bodies is in `ppp/robot_chain.py`, derived
  directly from `data/assets/mjcf/smpl_humanoid.xml` (geom
  centroids of single-density boxes / capsules — PhysX recomputes
  the same COMs from the USDA at load time). The FD window is
  invalidated on `set_dof_positions` / `set_root_state` /
  `commit_writes` so post-teleport reads never differentiate across
  a discontinuous state change. The articulation root keeps its
  binding-based `(root_lin_vel, root_ang_vel)` because the root
  binding is at COM already and round-trips at fp32 through
  `set_root_state`.

  The 20-tick replay on `body_jab_left.motion` is no longer
  catastrophic on body velocities: the live policy loop runs
  to motion-loop end without diverging, and `_trace_state_vs_frame`
  on tick 10 shows FD-derived body velocities consistent with the
  captured chain math (residual differences scale with the
  PhysX-vs-IsaacLab integrator drift, not with chain depth, which
  is the expected behaviour now that the Jacobian-rebuild bug is
  out of the loop). The `--replay-override-body-vel` bisection
  still passes at the same `fails=1/20` `obs_from_log` boundary as
  before, confirming we haven't regressed the override path.

### Added

- `ppp/robot_chain.py` — `RobotChain` dataclass plus the velocity
  helpers `quat_xyzw_to_mat`, `quat_fd_omega`,
  `finite_difference_velocities`, `apply_com_correction`,
  `remove_com_correction`. The chain is constructed from the
  `cfg.body_names` ordering (`RobotChain.from_body_names`) so it
  fails fast when a non-SMPL morphology lands without explicit
  support, and is plumbed into `PhysxWorld.__init__` from `app.py`.
- `ResolvedConfig.parent_indices` (and the hard-coded
  `SMPL_PARENT_INDICES` mirror in `ppp/config_loader.py`). Filled
  from `robot.kinematic_info.parent_indices` when present in the
  checkpoint, else falls back to the SMPL tree. Currently unused by
  the FD velocity path (it differences each body independently from
  its own previous link pose), but kept on the config so future
  forward-kinematics or constraint-based diagnostics can walk the
  tree without re-parsing the resolved config.
- `--replay-override-body-vel` swaps captured `rigid_body_vel` /
  `rigid_body_ang_vel` into the obs path *only* (the state-readback
  diff and the `_trace_state_vs_frame` table still measure the raw
  PhysX read so the underlying mismatch stays visible). Used to
  bisect "is the obs/policy/PD path correct given correct body
  velocities?" from "are body velocities correct?" The bisection on
  20 ticks of `body_jab_left.motion` is conclusive — every obs /
  policy / PD field collapsed from `fails=20/20` to `fails=1/20` at
  the exact magnitude of the residual `obs_from_log` boundary case
  (max 9.68e-01 on `max_coords_obs`, idx 213 at frame 0):
  - `obs.max_coords_obs`: 20/20 → 1/20 (matches `obs_from_log` exactly).
  - `obs.mimic_target_poses`: 20/20 → 1/20 (matches `obs_from_log`).
  - `policy_tanh_mu`: 20/20 → 1/20.
  - `pd_targets`: 20/20 → 1/20.
  Body velocities are therefore the *sole* upstream source of obs /
  policy / PD divergence for frames ≥ 1. The frame-0 residual is a
  separate obs-builder boundary issue (likely
  `historical_previous_actions[0]` carrying captured-run state into
  PPP's freshly-reset history, or `motion_time = 0` clamp behaviour)
  and is independent of velocities. The integrator-parity diffs
  (`post_*`) are unchanged by this flag because they live downstream
  of the obs path inside ovphysx's `step`.

  The unblocked next step on the velocity side is to replace
  `PhysxWorld.read_state`'s `ARTICULATION_LINK_VELOCITY` consumption
  with a forward-velocity computation from the live `(root_lin_vel,
  root_ang_vel, dof_pos, dof_vel)` walked over the SMPL kinematic
  chain so PPP gets trainer-parity body velocities by construction
  (matching the math the captured `rigid_body_vel` / `target_body_vel`
  were generated by).

### Changed

- The local-frame hypothesis from the previous CHANGELOG entry was
  **disproved** by the tick-10 trace on `body_jab_left.motion`. Live
  L_Hip's `delta-from-root` matches captured to fp32 (lin
  `(-0.034, -0.059, 0.027)`, ang `(0.331, -0.451, -0.375)` for
  both), but live `rot(L_Hip)` is *different* — meaning rotating the
  live value through `body_rot[L_Hip]` corrupts a value that was
  already in world frame. So ovphysx's `ARTICULATION_LINK_VELOCITY`
  returns world-frame velocities; the failing bodies (L_Knee, L_Ankle,
  L_Toe, R_Hip onward, and the long-arm leaves) are wrong in world
  frame. Concretely: `Δv = v(child) - v(parent)` for L_Knee in the
  captured log is `(0.066, -0.042, 0.027)` (= Pelvis→Hip→Knee
  Jacobian × dof_vel), and PPP's read gives `(0.091, 0.395, -0.083)`
  — same rough magnitude on X but a Y component an order of
  magnitude bigger and Z flipped. Divergence then grows down the
  chain (Hand: ~12 m/s, Wrist: ~32 rad/s) exactly as `Σ Jᵢ × dof_velᵢ`
  accumulation errors compound through a 7-link arm chain. With
  `dof_vel` round-tripping at exactly 0.0, the only remaining
  variable is the chain-propagation step inside ovphysx: joint anchor
  point, joint axis order/orientation as parsed from the SMPL USDA,
  or the Jacobian sample point can each independently produce this
  shape of error. Diagnosing further from black-box probes is
  guesswork, so the planned fix is to bypass that read entirely
  (FK forward-velocity, see Added above).
- `--replay-trace-tick` default raised from 0 to 10 and the trace now
  prints `body_*_vel - root_*_vel` deltas instead of absolute body
  velocities. Frame 0 of most ProtoMotions captures has `dof_vel = 0`
  and `root_ang_vel = 0`, so every body's velocity is identical to the
  root's by construction — the trace at tick 0 cannot tell "PhysX
  correctly propagates the chain" from "PhysX returns root_vel for
  everything" because both produce the same numbers. Tick 10
  (≈ 0.33 s into a 30 Hz walk capture) is mid-motion with non-zero
  joint velocities. Subtracting the root velocity from every body
  isolates the chain-propagation contribution, which is exactly what
  ovphysx's spatial-Jacobian rebuild is supposed to populate; rows
  that are zero in the live column but non-zero in the captured
  column directly evidence a missing dof_vel propagation in
  `ARTICULATION_LINK_VELOCITY`.

### Added

- Replay teleport gains a `--replay-teleport-mode {commit,real-step,
  two-pass}` switch and a one-shot trace via `--replay-trace-tick N`.
  The two-pass teleport landed in the previous CHANGELOG entry produced
  numerically *bit-identical* diffs to the simpler `commit` mode (worst
  `body_lin_vel` 12.006 m/s on L_Hand, worst `body_ang_vel` 32.089
  rad/s on L_Wrist — same to 5 decimals before and after the change),
  so pass 1's `step(dt_physics)` either does not actually populate
  `ARTICULATION_LINK_VELOCITY` for descendant bodies or pass 2's
  `commit_writes` wipes it. The new mode flag lets the operator
  isolate which pattern (if any) populates the binding; the new trace
  prints a per-body live-vs-captured table for `body_lin_vel` /
  `body_ang_vel` on a single tick so the operator can see directly
  whether PhysX is returning literal zeros (binding never populated)
  or non-zero-but-wrong values (frame / origin convention mismatch).

### Fixed

- Replay teleport now uses a two-pass strategy so non-root link
  velocities (`ARTICULATION_LINK_VELOCITY[i>0]`) read back close to the
  captured frame instead of returning stale Jacobian-cache values. The
  bisection from the new replay diagnostics localised the issue
  precisely:
  - Position-level state (`body_pos`, `body_rot`, `dof_pos`,
    `dof_vel`) round-trips at fp32 through the previous single-pass
    `commit_writes` teleport.
  - Root velocities (`root_lin_vel`, `root_ang_vel`) and *body 0*
    (Pelvis) link velocities also round-trip at fp32, and PPP's two
    readings of body 0 (root binding vs link binding) agree at fp32 —
    so the root velocity write is fine and the link[0] tensor slot is
    being kept in sync with the root binding.
  - Non-root link velocities (`body_lin_vel[i>0]` / `body_ang_vel`)
    were stuck at whatever the previous real step left in the cache
    (worst diffs ~12 m/s and ~32 rad/s on leaf links — Hand, Wrist,
    Toe), because a zero-dt `step(0,0)` refreshes `link_pose` via FK
    but never invokes the spatial-Jacobian rebuild that propagates
    `dof_vel` × Jacobian to descendant link velocities.
  - `obs_from_log.max_coords_obs` / `obs_from_log.mimic_target_poses`
    matched captured obs to fp32 noise on 19/20 frames, confirming
    `ObsBuilder` math is unchanged from the bit-equivalent baseline
    in CHANGELOG and that all live `obs.*` failures attribute solely
    to the link-velocity read-back issue.
  Fix in `InferenceApp._teleport_from_frame`: pass 1 writes captured
  state, sets PD targets to `dof_pos + dof_vel × dt_physics` (the
  post-substep pose, so PD applies no braking torque and the joints
  free-fly with their captured velocities for one integrator
  substep), then `step(dt_physics)` to force the Jacobian rebuild.
  Pass 2 re-writes captured root and DOF state to undo the
  sub-mm / sub-rad integrator drift, then `commit_writes()` refreshes
  `link_pose` from the just-written positions. Link velocities from
  pass 1's Jacobian rebuild are preserved because `step(0,0)` is a
  fetch-results call, not an integrator call. Cost is one extra
  1/120 s substep per teleport (~negligible vs ONNX inference and
  ovphysx fixed costs).

### Added

- Replay diagnostics gain three new comparisons in
  `InferenceApp._diff_state` and the open-loop tick body, isolating
  the next suspect surface after the teleport-settle fix landed:
  - `root_lin_vel` / `root_ang_vel` diff `state.root_*` (read from
    `ARTICULATION_ROOT_VELOCITY`) against the captured frame, which is
    a pure write-then-read round-trip on the root binding. Catches a
    broken root velocity write before any link-tensor questions come
    into play.
  - `root_vs_link0_lin_vel` / `root_vs_link0_ang_vel` compare PPP's
    *two* readings of body 0 (Pelvis = articulation root): the one
    from `ARTICULATION_ROOT_VELOCITY` and the one from
    `ARTICULATION_LINK_VELOCITY`. Identical values plus a divergence
    vs captured points to a frame/origin convention mismatch with the
    trainer; divergence between the two bindings instead points
    directly at a stale `ARTICULATION_LINK_VELOCITY` cache (PhysX's
    spatial Jacobian only rebuilds during a real step, so a teleport
    followed by `commit_writes` leaves link velocities at whatever the
    prior real step computed).
  - `obs_from_log.max_coords_obs` / `obs_from_log.mimic_target_poses`
    re-run `ObsBuilder.build` against `frame.as_physx_state()` (the
    captured state, not the live read-back) and diff against captured
    obs. Per CHANGELOG this matched to fp32 from frame 1 onward; if
    it still does, every live `obs.*` failure attributes to the
    `read_state` link-velocity issue rather than to obs math.

### Fixed

- Replay teleport no longer integrates physics by 1/120 s before the
  next `read_state`. The first run of `--replay-debug-log` reported
  catastrophic pre-step diffs (`body_pos` ~2.5 cm, `body_lin_vel`
  ~4.3 m/s, `dof_pos` ~0.04 rad, `dof_vel` ~11 rad/s with worst
  offenders concentrated on the legs). Root cause: ovphysx queues
  articulation writes and only refreshes derived read-side tensors
  (`ARTICULATION_LINK_POSE` / `ARTICULATION_LINK_VELOCITY`) during a
  step, so the previous teleport relied on a `step(dt_physics)` to
  commit the writes — and that real substep advanced any non-zero DOF
  velocities by `vel × 1/120 s` before the obs builder ever saw the
  state. New `PhysxWorld.commit_writes()` calls
  `self._physx.step(0.0, self._sim_time)` (the same "fetch without
  advancing" call PPP already uses during init), and
  `InferenceApp._teleport_from_frame` now drives that path instead of
  a real settle step. The `set_dof_targets(dof_pos)` call inside
  teleport is also gone — it was only there to hold pose during the
  settle, and the replay loop overwrites targets immediately afterward
  anyway.

### Added

- New `--replay-debug-log` CLI mode that drives PPP from a captured
  ProtoMotions `debug_output.json` instead of the live motion file. Goal
  is bisecting which layer of the inference pipeline diverges from the
  trainer; the previous "live walking is in-place" investigation
  bottomed out because PhysX read-back, obs builders, ONNX inference,
  and `set_dof_targets` were all probable suspects with no clean way to
  test them in isolation. New flags on `ppp/app.py`:
  - `--replay-debug-log` enables the harness.
  - `--replay-mode {open-loop,closed-loop}` — open-loop re-teleports
    state to the captured frame each tick (one-step PhysX behaviour
    only); closed-loop initialises from frame 0 and lets PPP integrate
    forward against the log so accumulated divergence is visible.
  - `--replay-action-source {policy,log}` — feed PPP's ONNX output into
    the PD path or replay the captured `actions` directly so integrator
    parity can be measured independent of any policy-input divergence.
  - `--replay-frames N`, `--replay-no-xy-shift`,
    `--replay-inject-historical`, and `--replay-{state,obs,action,
    poststep}-tol` for tuning the run.
  Per tick the harness records max/mean abs diffs (with worst-offender
  body / DOF name) for `read_state` vs `rigid_body_*` / `dof_*`,
  `max_coords_obs`, `mimic_target_poses`,
  `historical_previous_actions`, ONNX `tanh_mu` vs captured `actions`,
  PPP `pi * tanh_mu` vs captured `pd_targets`, and the post-step
  `read_state` vs the captured frame `N+1`. End of run prints a summary
  and the process exits non-zero if any tolerance was breached, so the
  harness doubles as a regression smoke test.
- `ppp/debug_replay.py` houses the `DebugReplayLog` loader,
  `ReplayFrameView` accessor, and `ReplayDiagnostics` aggregator.
  Loader validates body / DOF names against the resolved config, stacks
  per-frame arrays into contiguous `float32` buffers, and exposes
  `as_physx_state` / `as_motion_future` / `as_obs_inputs` helpers so
  the same captured frame can drive PhysX teleports, the obs builder,
  and direct ONNX checks. Coordinate handling is explicit: the captured
  `rigid_body_*` is in the simulator's world frame (motion-local +
  `respawn_root_offset.xy`) while `target_body_*` is in the motion-local
  frame; by default the loader subtracts the XY offset from
  `rigid_body_*` so PPP runs near origin (matches the runtime USD's
  ground plane) with the obs invariant intact, and
  `--replay-no-xy-shift` instead lifts `target_body_*` so both arrays
  land in the captured world frame.

### Fixed

- Stop the SMPL motion-tracker from in-place stepping on locomotion
  motions. Two bugs were biasing the obs and the spawn state away from
  what the trained policy expected; both diagnosed by feeding the
  captured `rigid_body_*` / `target_body_*` from
  `Protomotion3/Assets/onnx/smpl_policy/debug_output.json` (128 frames)
  through PPP's obs builders and bisecting until output matched the
  captured `max_coords_obs` / `mimic_target_poses` to fp32.
  - `ppp/motion.py::MotionPlayer` no longer applies `ref_respawn_offset_z`
    to every reference body position it returns. The previous code
    mirrored the wrong half of protomotions' setup: the +0.05 m lift
    only goes onto the **character's spawn root**, never onto
    `mimic_ref_pos` or the gt-error reference. Concretely,
    `mimic_control.py` calls
    `env.get_spawn_to_ref_pose_offset_with_terrain_height_correction`
    which returns `(respawn_root_offset.xy, terrain_z)` — the env-spawn
    XY translation plus terrain height, but **not** `ref_respawn_offset`.
    With the lift incorrectly applied to the obs ref pose, every body's
    Z target was 5 cm above the trained-distribution value; on flat
    walking motions this re-biased the policy toward in-place stepping
    and the per-second tracking error grew from 0.16 m to 7.9 m over
    12 s. After the fix the parity test reports max diff ≈ 1×10⁻⁶ on
    `mimic_target_poses` from frame 1 onward, and the live walking
    smoke test ([`KB_WalkFwd1.motion`]) shows tracking error 0.041 →
    0.090 m, deterministic across loops, in the healthy SMPL range.
  - `ppp/app.py::reset` now passes the motion's per-DOF velocities to
    `world.set_dof_positions` instead of letting them get silently
    zeroed. On stationary motions (`body_jab_left`) the discarded
    `dof_vel` was always near zero so the bug was invisible, but on
    walking motions the legs carry stride momentum at *t=0* (`target_body_vel[0]
    ≈ (-0.05, +0.08, -0.03)` for the captured walk; the captured
    `rigid_body_vel[0]` matches it bit-for-bit, confirming protomotions
    seeds DOF velocities from the motion). Starting with zero leg
    velocities pushed the very first policy obs out of distribution and
    compounded into the locomotion drift.
  - `ppp/app.py::reset` still adds `ref_respawn_offset_z` (5 cm) to the
    spawn root — that part of the offset is real, it's the height
    clearance training uses so the character doesn't instantiate
    clipped through the floor.

- Match ProtoMotions' `historical_previous_actions` semantics exactly:
  store the **post-tanh** `mean_action` and read it back with a **2-step**
  lag, not the pre-tanh `mu` with a 1-step lag as the previous fix
  attempted. Diagnosed by running the SMPL `model.onnx` directly on the
  obs captured in `Protomotion3/Assets/onnx/smpl_policy/debug_output.json`
  (128 frames at 30 Hz):
  - `historical_previous_actions[N]` matches `actions[N-2]` to fp32 for
    every captured frame; every other lag (-3, -1, 0, +1, +2) is off by
    0.10–0.28. The 2-step lag falls out of protomotions'
    `state_history_buffer.rotate_and_update` running *inside* `env.step`
    *before* `get_obs`, with the resulting obs only consumed by the
    *next* policy call (see `protomotions/envs/base_env/env.py::step`
    and `post_physics_step`).
  - `actions[N]` equals the ONNX `tanh` output (= post-tanh
    `mean_action`) and `pd_targets = pi * actions` to fp32 — no second
    tanh in the PD path, so the buffer must store the post-tanh value
    (the "raw" in `_current_raw_action` means "before `_process_action`'s
    PD scale", *not* "before tanh").
  - With PPP's previous (raw `mu`, 1-step lag) feeding rule the per-step
    action diverges from the protomotions reference by 0.04–0.12 on the
    very first frames (`pi *` that = 0.13–0.38 rad of unintended PD
    target), compounding into the documented "stands then falls within
    a few seconds" failure.
  Implementation:
  - `ppp/obs_builder.py::ObsBuilder._history_buf` is now a `(1, 2, A)`
    buffer matching protomotions'
    `StateHistoryBuffer.actions[:, :num_state_history_steps + 1]` for
    `num_state_history_steps = 1`. `push_action(post_tanh_action)`
    rotates `slot 1 ← slot 0` and writes the new action into slot 0;
    `build()` passes `self._history_buf[:, 1:]` (= slot 1 only) into
    `compute_historical_actions_from_state(history_steps=1)` so the
    obs sees `action[N-2]` at frame N (zero for N=0,1).
  - `ppp/app.py::policy_tick` now calls `obs_builder.push_action(tanh_mu)`
    instead of `push_action(raw_mu)`. The PD computation
    `action_proc.process(tanh_mu) = pi * tanh_mu` is unchanged and
    matches the captured `pd_targets` to fp32.
  - `ppp/policy.py::OnnxPolicy._load_and_patch_graph` is kept (the
    `mean_action_raw` output it exposes is still useful as a diagnostic
    in the "Action stats" log line — an unbounded linear output that
    climbs past ~2–3 in absolute value indicates the policy is feeding
    back off-distribution for some other reason). Module + method
    docstrings updated to make clear the raw-mu output is *not* fed
    into the obs.
- Run the policy at a fixed 30 Hz cadence regardless of how often the
  wall loop iterates. `ppp/app.py::run` now accumulates
  `wall_dt * --time-scale` and drains the accumulator in `dt_policy`
  chunks by calling the renamed `policy_tick()` (was `step(dt)`), which
  itself uses `self.dt_policy` everywhere — for `motion.get_future`,
  the `time_to_target` scalar, `world.step`, and the motion-time
  advance. Previously the obs builder, the historical-action buffer,
  and the PD-target hold time saw whatever variable `sim_dt` the wall
  loop produced (often 3–10 ms on Windows), feeding the policy a
  future ≈ 5 ms ahead (so `mimic_target_pos_rel ≈ 0` for every body)
  and rotating the previous-action buffer every wall iter instead of
  every 33 ms — both far off the training distribution. The renderer
  and event polling now run once per outer iter, decoupled from the
  policy tick rate. `--time-scale 0` still means "exactly one policy
  tick per outer iter".
- `ppp/motion.py::MotionPlayer` now applies `ref_respawn_offset_z` (0.05 m
  for the SMPL motion-tracker) to every reference `rigid_body_pos` it
  returns from `get_state` / `get_future`. Wired through
  `InferenceApp.__init__` from `cfg.ref_respawn_offset_z`. Training
  applies the same offset to every future / current reference pose via
  `env.get_spawn_to_ref_pose_offset_with_terrain_height_correction`, so
  without it the un-offset reference combined with the offset-teleported
  sim character left `mimic_target_pos = ref_pos - cur_root_pos` with a
  persistent -5 cm Z bias on every body — the policy "corrected" by
  driving the root into the floor. The manual `+= ref_respawn_offset_z`
  adds in `InferenceApp.reset` and `_motion_ref_body_pos` are removed
  since `MotionPlayer` is now the single source of truth.

### Added

- `--debug-mimic-pose` CLI flag that overlays the motion-reference body
  positions as 24 bright spheres in the renderer window, so the operator
  can eyeball where the policy thinks each joint should be vs where
  PhysX actually puts it (same quantity the tracking-error metric
  averages over). When set, `ppp/assets.py::write_runtime_scene` authors
  a `/World/MimicTargets` scope of pure-visual `Xform`+`Sphere` prims
  (no physics APIs, so ovphysx ignores them) and `ppp/app.py` pushes the
  reference body positions to them each policy tick via a new
  `RtxViewer.push_mimic_target_positions` write path (extended through
  `ppp/remote_renderer.py` as an extra field in the per-tick xform
  message). The respawn-z offset applied at reset is added to the
  positions so the markers and the simulated character live in the same
  comparison frame as `_accumulate_tracking_error`.

### Fixed

- Bake the IsaacGym/IsaacLab runtime PhysX configuration into the
  composed `_runtime_scene.usda` in `ppp/assets.py` so ovphysx matches
  the SMPL motion-tracker training environment. The trained policy
  was previously off-distribution because the upstream
  `smpl_humanoid.usda` only authors PD stiffness/damping/armature —
  ProtoMotions sets everything else from
  `protomotions/{simulator,robot_configs}/...` at runtime, which we
  weren't replaying. An OmniPVD capture
  (`out/omnivpd_usd/.../IsaacSimPVD.usda`) was used as ground truth.
  Concrete additions:
  - `PhysicsScene` now applies `PhysxSceneAPI` and pins
    `physxScene:solverType = "TGS"` (default ovphysx is PGS, which is
    visibly less stable under the SMPL stiffness ∈ [300, 1000] PD).
  - Per-body `physxRigidBody:maxAngularVelocity = 1000` deg/s
    (≈ 17.45 rad/s, matches PVD), `maxLinearVelocity = 1000` m/s,
    `maxDepenetrationVelocity = 1.0` m/s,
    `enableGyroscopicForces = 1`. Without these the policy could
    drive links to spin / depenetrate far faster than ever seen in
    training, producing the "twitchy / launched feet" look.
  - Per-joint drive `maxForce = 500` N·m (overrides upstream's
    `FLT_MAX`) and `physxJoint:maxJointVelocity = 100` rad/s. Matches
    `protomotions/robot_configs/smpl.py::ControlInfo` (every joint
    group has `effort_limit=500`, `velocity_limit=100`). With FLT_MAX
    the PD chases targets with unbounded torque, which is the
    biggest single source of the "stiff / overshooting" feel.
  - `physxArticulation:solverPositionIterationCount = 4` and
    `solverVelocityIterationCount = 4` on the Pelvis (articulation
    root). Default ovphysx velocity-iter count is 1; PVD captured 4.
  - `/World/GroundCollider` now applies `PhysxCollisionAPI` and pins
    `physxCollision:contactOffset = 0.001 m` (down from the PhysX
    default of 0.02 m). Combined with the foot collider's 0.02 m
    offset that puts the contact surface ~2 cm above the visual
    ground in the PhysX default; IsaacGym's terrain mesh used
    ~0.00136 m, so the SMPL feet now land where the trained policy
    expects.
- `ppp/config_loader.py` and `ppp/app.py` now actually use the
  IsaacGym `sim.substeps` field for the PhysX integrator dt instead
  of just warning about it. `ResolvedConfig` gained a `substeps: int`
  field and a `dt_physics` property = `1 / (physics_fps * substeps)`,
  and `InferenceApp.__init__` passes that property into
  `PhysxWorld(dt_physics=...)`. Without this, an IsaacGym-style
  resolved config (`fps=60`, `substeps=2`) would have silently
  integrated at 1/60 s per substep — half the trained 1/120 s solver
  rate. The `Resolved config: ...` log line now reports both `fps`,
  `decimation`, `substeps`, and the effective integrator Hz.

### Added

- Blender-style orbit camera in `ppp/renderer.py`. The viewer now owns a
  pivot + spherical (azimuth, elevation, distance) state and writes a
  proper look-at matrix to ``/World/Camera`` every frame instead of just
  translating the eye and trusting the USD-authored orientation (which
  was the source of the "follow camera looks at nothing in particular"
  bug). Bindings match Blender:
  - MMB drag: orbit, Shift+MMB: pan, wheel: zoom.
  - LMB drag / Shift+LMB: same as MMB variants — laptop fallback for
    machines without a real middle button.
  - `F`: snap pivot to the character root (re-frame after panning away).
  - `O` (existing): toggle follow mode; in follow the pivot is yanked to
    the root each tick but orbit/zoom/pan state is preserved so the
    user's chosen angle survives.
  - Elevation clamped to ±89° to avoid look-at degeneracy at the poles;
    distance clamped to [0.05, 100] m. All sensitivities and limits are
    private attrs at the top of ``RtxViewer.__init__`` — tweak in place,
    no CLI knobs (yet).
- `README.md` "Keys & Mouse" section documenting the new camera
  controls.

### Fixed

- `ppp/config_loader.py` was digging ``fps`` / ``decimation`` /
  ``substeps`` at the top level of the resolved simulator config, but
  ProtoMotions nests these under ``SimulatorConfig.sim`` (a ``SimParams``
  dataclass — see ``protomotions/simulator/base_simulator/config.py``).
  ``_dig`` returned ``None`` and we silently fell back to the
  ``physics_fps=120`` / ``decimation=4`` defaults, which happen to
  match the canonical SMPL motion-tracker training (IsaacLab/Newton).
  But for an IsaacGym-trained checkpoint (``fps=60``, ``decimation=2``,
  ``substeps=2``) we'd have silently doubled the integrator rate. Now
  reads ``sim.fps`` / ``sim.decimation`` / ``sim.substeps`` first and
  falls back to the flat layout, logs the resolved policy Hz and
  effective integrator Hz, and warns if the training-time effective
  solver rate (``fps × substeps`` for IsaacGym-style configs) differs
  from what ovphysx will run.

### Changed

- `PhysxWorld.step` in `ppp/physx_world.py` now takes a real
  ``dt: float`` instead of an opaque ``substeps: int`` count. This
  matches the underlying ``ovphysx.PhysX.step(dt, sim_time)`` API and
  makes the wrapper the obvious place to plug in variable-dt
  simulation. Internally it splits ``dt`` into
  ``ceil(dt / self.dt_physics)`` equal substeps so the PhysX
  integrator's substep size stays close to the trained 1/120 s
  regardless of the chunk size we ask it to integrate (``dt = 1/30`` →
  4 substeps of 1/120; ``dt = 1/60`` → 2 substeps; small ``dt`` → 1
  substep of that size). All callers in `ppp/app.py` updated.
- `InferenceApp.step` in `ppp/app.py` now takes ``dt: float`` and
  threads it through everything that touches sim time: physics
  integration (``world.step(dt)``), motion-reference look-ahead
  (``obs_builder.build(..., future_dt=dt)``), motion time advance
  (``self.t += dt``), renderer (``viewer.render(dt)``). The trained
  policy's fixed-30 Hz cadence is documented as a caveat, not enforced.
- `InferenceApp.run` in `ppp/app.py` rewritten as a wall-driven loop:
  per iter we compute ``sim_dt = min(wall_dt * --time-scale, 5 *
  dt_policy)`` and call ``step(sim_dt)``. Cap exists so a debugger
  pause / GC / shader compile / startup frame doesn't dump a
  multi-second dt into the integrator on resume.
  ``--time-scale 0`` keeps the legacy "step at trained dt regardless
  of wall clock" behaviour for headless / metric runs.
- `PhysxWorld.__init__` logs the physics substep target in both
  seconds and Hz at startup (e.g. ``physics substep target = 0.0083 s
  (120.0 Hz)``) so it's obvious at runtime whether we're matching the
  training integrator rate.

### Added

- Per-body tracking-error diagnostic in `ppp/app.py`. After each policy
  tick the app accumulates the mean per-body L2 distance between sim
  body positions and the motion reference at `self.t` (with the same
  `ref_respawn_offset_z` shift used at reset), and logs the rolling
  average every `--tracking-log-every` ticks (default 30 ≈ once per
  simulated second at 30 Hz). Healthy SMPL tracking sits at 0.05–0.15 m;
  >0.5 m indicates the character is falling/off-policy. This is the
  same `gt_err` ProtoMotions logs from `agents/evaluators/`.

### Removed

- Sleep-based pacing in `ppp/app.py` and the accompanying Windows
  ``timeBeginPeriod(1)`` / ``timeEndPeriod(1)`` calls. The fixed-tick
  accumulator that briefly replaced them is also gone — both were the
  wrong abstraction for "scale the integrator dt by real time" (per
  user feedback: ``world.step`` always accepted a ``dt``; the wrapper
  was hiding it behind a ``substeps`` count). Replaced by the
  wall-driven variable-dt loop above.
- Per-body tracking-error diagnostic in `ppp/app.py`. After each policy
  tick the app accumulates the mean per-body L2 distance between sim
  body positions and the motion reference at `self.t` (with the same
  `ref_respawn_offset_z` shift used at reset), and logs the rolling
  average every `--tracking-log-every` ticks (default 30 ≈ once per
  simulated second at 30 Hz). Healthy SMPL tracking sits at 0.05–0.15 m;
  >0.5 m indicates the character is falling/off-policy. This is the
  same `gt_err` ProtoMotions logs from `agents/evaluators/`.
- SMPL body-collider physics material. `ppp/assets.py` now also authors
  `/World/PhysicsMaterials/SmplBodyMaterial` (`staticFriction = 0.5`,
  `dynamicFriction = 0.5`, `restitution = 0.0`) and binds it to each of
  the 24 SMPL collider geoms at
  `/World/Robot/bodies/<BodyName>/collisions/_geom_<idx>` (idx matches
  the body index, Pelvis→0 … R_Hand→23). The bindings are authored as
  explicit per-collider `over`s (not at an ancestor) so we don't rely on
  ovphysx implementing USD's ancestor-resolution for material bindings —
  the ovphysx samples never exercise the inheritance path. The
  collider→geom map is centralised in `_SMPL_BODY_COLLIDER_GEOMS` and
  the 24 overs are generated by `_smpl_body_material_overs()`.
- Ground friction. `ppp/assets.py` now authors
  `/World/PhysicsMaterials/GroundMaterial` (a `Material` prim with
  `PhysicsMaterialAPI`, `staticFriction = 1.0`, `dynamicFriction = 1.0`,
  `restitution = 0.0`) and binds it to `/World/GroundCollider` via
  `rel material:binding:physics` (the `physics` purpose used by
  `omni.physx`'s `add_physics_material_to_prim`). The collider also gets
  `MaterialBindingAPI` prepended to its applied schemas to make the
  binding well-formed. Paired with the SMPL body material above, the
  effective per-contact friction (PhysX default `average` combine mode)
  is `(1.0 + 0.5) / 2 = 0.75`.
- Out-of-process ovrtx renderer in `ppp/remote_renderer.py`. The parent
  process owns `ovphysx` + ONNX policy + `MotionLib`; a child process
  (spawned via `multiprocessing`) owns `ovrtx` + `pygame`. Body
  transforms are shipped each policy tick over a bounded
  `multiprocessing.Queue` with drop-oldest-on-full so the renderer never
  back-pressures the policy loop. Keyboard events (`Q`/`R`/`O`) come
  back the same way. This is required because `ovphysx` and `ovrtx`
  ship incompatible vendored carb plugin stacks and cannot coexist in a
  single Python process — the same architecture NVIDIA's
  `SimReadyBrowser` adopts (in mirror: they put ovphysx in the child).
- Clean-shutdown handshake in `RemoteViewer.close()`: drain queued
  xforms → send `quit` → wait for child's `bye` ack (which the child
  emits *after* `viewer.close()`) → 3 s grace → fall back to
  `terminate()`. Matches the documented pattern in
  `SimReadyBrowser/core/physics_controller.py`.
- `CHANGELOG.md` (this file) and `.cursor/rules/changelog.mdc` rule
  requiring future user-visible changes to be logged here in the same
  response that makes the change.

### Changed

- `ppp/assets.py` runtime USD now authors the ovrtx-required render
  layout: `/World/Camera` is the actual `Camera` prim and
  `/Render/Camera` is a `RenderProduct` that references it and declares
  the `LdrColor` render var (this is the path passed to
  `Renderer.step(render_products={...})`). The previous layout had a
  bare `Camera` at `/Render/Camera`, which ovrtx rejected with *"Invalid
  render product path"*.
- `ppp/renderer.py` follow-camera xforms now write to `/World/Camera`
  (the new camera xformable) instead of `/Render/Camera` (which is now
  the RenderProduct).
- `ppp/app.py` orchestration: removed the in-process ovrtx import and
  the construct-renderer-before-physics workaround; now constructs
  `PhysxWorld` first (parent owns ovphysx) and then spawns the
  out-of-process `RemoteViewer`. Unifies the *"ovrtx is missing"*
  install hint into the `RemoteViewer` failure path.
- `ppp/__init__.py` no longer eagerly imports `ovrtx`. The eager import
  was added to satisfy the ovrtx 0.3.0 release-notes constraint
  *"import ovrtx must come before import ovphysx"*, but in practice
  this is impossible to honour in the same process (the carb plugin
  singletons collide later anyway). The subprocess design sidesteps
  the constraint entirely.

### Fixed

- Character was still falling through the ground despite the previous
  override. Switched `ppp/assets.py` to the canonical ovphysx static
  ground pattern from
  `ovphysx/samples/data/boxes_falling_on_groundplane.usda`: a dedicated
  `def Plane "GroundCollider"` at `/World/GroundCollider` with
  `PhysicsCollisionAPI` applied, `axis = "Z"` and `purpose = "guide"`
  (invisible). The upstream `checkerboard_ground.usda` is now used purely
  for the visual — its `PhysicsRigidBodyAPI` is stripped via
  `delete apiSchemas = ["PhysicsRigidBodyAPI"]` on the local `over`
  (with `physics:rigidBodyEnabled = 0` kept as belt-and-suspenders).
  The previous attempt (`physics:rigidBodyEnabled = 0` on `/World/Ground`
  + `physics:collisionEnabled = 1` on `/World/Ground/visuals`) silenced
  the *"invalid inertia tensor"* warning but did **not** register the
  descendant mesh as an active collider in ovphysx — turning off the
  parent rigid body removes the whole subtree from physics processing,
  including child colliders. The separate `Plane` bypasses this entirely.
- Added a `def PhysicsScene "physicsScene"` prim under `/World` with
  `gravityDirection = (0, 0, -1)` and `gravityMagnitude = 9.81`. Every
  ovphysx sample under `ovphysx/samples/data/` authors one explicitly;
  relying on ovphysx's implicit defaults is fragile.
- Rendered output was never visible in earlier runs. Two causes,
  both resolved:
  1. `ovrtx` was not installed in the conda env (the package isn't on
     PyPI; the GitHub release ships a 2 GB zip, not a wheel). The
     correct install is now documented in `app.py`'s renderer-disabled
     error path: `pip install
     https://pypi.nvidia.com/ovrtx/ovrtx-0.3.0.312915-py3-none-win_amd64.whl`.
  2. The renderer was disabled silently when ovrtx failed to import —
     `app.py` now logs the install command at `ERROR` level and the
     exception is re-raised through `RemoteViewer` so it's never
     missed.

### Removed

- The `import ovrtx` preload from `ppp/__init__.py` (see *Changed*
  above for rationale).

## [0.1.0] - 2026-05-19

Initial port of the ProtoMotions SMPL mimic inference pipeline onto
`ovphysx` + `ovrtx`. See `README.md` for project layout and usage.

### Added

- Scaffolding: `pyproject.toml`, `README.md`, `.gitignore`,
  `run_inference.py` CLI entry-point, and the `ppp/` package.
- `ppp/assets.py`: composes a runtime USD by referencing ProtoMotions'
  `smpl_humanoid.usda` and `checkerboard_ground.usda`, plus a camera
  and distant light.
- `ppp/config_loader.py`: parses `resolved_configs_inference.pt` for
  body / DOF names, PD action params, and timing; falls back to SMPL
  defaults if the file is missing.
- `ppp/physx_world.py`: ovphysx wrapper that loads the runtime USD,
  creates per-body and per-DOF tensor bindings (`TensorType.*` enums
  for ovphysx 0.3.x), and reorders state/targets to match
  ProtoMotions' `kinematic_info` ordering when PhysX disagrees.
- `ppp/motion.py`: thin wrapper around `protomotions.components.motion_lib.MotionLib`
  that exposes `get_state(t)` and `get_future(t, dt, n)` for a single
  environment.
- `ppp/obs_builder.py`: reuses ProtoMotions'
  `compute_humanoid_max_coords_observations`,
  `build_max_coords_target_poses`, and
  `compute_historical_actions_from_state` to construct the three ONNX
  inputs. Appends a `time-to-target` scalar per future step so
  `mimic_target_poses` is 577-D (matching the trained checkpoint),
  not the 576-D the bare builder produces.
- `ppp/policy.py`: `onnxruntime.InferenceSession` wrapper that
  introspects input/output names and resolves the `mean_action`
  output via metadata or common names (e.g. `tanh`).
- `ppp/action.py`: PD action post-processing (`pd_action_offset` +
  `pd_action_scale` × policy output) for 69 DOF position targets.
- `ppp/renderer.py`: ovrtx + pygame `RtxViewer` (now invoked
  exclusively from the renderer subprocess).
- `ppp/app.py`: main loop — reset → 30 Hz policy tick → 4× physics
  substeps → render → motion looping on end.
- `scripts/smoke_obs.py`: offline obs / policy smoke test that
  bypasses ovphysx + ovrtx.
- `scripts/install_ovrtx.ps1`: PowerShell installer for the ovrtx
  wheel (currently superseded by the install command logged from
  `app.py`).

### Fixed

- `ppp/__init__.py` prepends `C:\Git\ProtoMotions` to `sys.path`
  because the conda-installed `protomotions` distribution only ships
  the top-level package, not `protomotions.components` / `.envs`.
- `ppp/config_loader.py` parses the `resolved_configs_inference.pt`
  dictionary structure (`robot`, `env`, `simulator` keys) rather than
  assuming a single `env_config` blob.
