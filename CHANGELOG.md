# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Re-targeted PPP at ProtoMotions' unified-pipeline ONNX export and
overhauled the runtime PhysX setup to match the IsaacGym/IsaacLab
training environment. This is one cumulative entry vs `[0.1.0]`; the
per-step engineering narrative lives in git history.

### Changed

- **Migrated to the unified-pipeline ONNX export.** PPP consumes
  `unified_pipeline.onnx` from
  `deployment/export_bm_tracker_onnx.py`, which bakes observation
  computation *and* PD action processing into the ONNX graph. Host
  code no longer rebuilds `max_coords_obs` / `mimic_target_poses` /
  `historical_previous_actions` or post-processes `pi * tanh(mu)`;
  it feeds raw context tensors (`current.rigid_body_*`,
  `mimic.future_*`, `historical.actions`, `ground_heights`) and
  reads `joint_pos_targets` straight out of the ONNX.
  `scripts/obs_builder.py` is now a YAML-driven raw-tensor
  assembler keyed by the sidecar's `policy_inputs`;
  `scripts/policy.py` is a thin wrapper returning a typed
  `PolicyOutputs`; `scripts/action.py` is deleted;
  `scripts/motion.py::MotionPlayer.get_future` accepts an explicit
  `step_indices` argument matching the YAML's
  `motion.future_step_indices`. See `docs/onnx_input_migration.md`.
- **Dropped `--checkpoint`; only `--onnx` is required.** The YAML
  sidecar shipped next to the ONNX carries the full deployment
  contract (joint / body schema, PD gains, control timing,
  future-step indices). PPP auto-discovers it by looking for
  `<onnx_stem>.yaml`, then `unified_pipeline.yaml`, then any
  single `*.yaml` in the same directory; pass `--yaml` to
  override. `scripts/config_loader.py` replaces
  `load_resolved_config(ckpt)` with `load_config_for_onnx(onnx,
  ...)` and a new `ResolvedConfig` carrying `policy_inputs` /
  `policy_outputs` specs plus `control_dt` / `physics_dt`
  directly. Default `--onnx` now points at `models/model.onnx`.
- **Runtime USD matches the IsaacGym/IsaacLab training setup.**
  `scripts/assets.py` authors an explicit `PhysicsScene` (TGS
  solver, 4/4 iteration caps, `bounceThreshold = 0.2`), a
  dedicated `Plane "GroundCollider"` (the upstream
  `checkerboard_ground.usda` stays visual-only — its
  `PhysicsRigidBodyAPI` is stripped), `PhysicsMaterial`s for
  ground and SMPL bodies (friction 1.0/1.0), per-body PhysX caps
  (`maxAngularVelocity = 1000` deg/s, `maxLinearVelocity = 1000`
  m/s, `maxDepenetrationVelocity = 1.0` m/s, gyroscopic forces
  on), per-joint effort / velocity limits (500 N·m / 100 rad/s),
  and articulation solver iteration counts (4/4) on the Pelvis.
  Per-DOF PD gains come from the resolved YAML.
- **Body velocities are derived via finite-difference + COM
  correction** in `PhysxWorld.read_state` rather than ovphysx's
  `ARTICULATION_LINK_VELOCITY` (which disagrees with the
  trainer's body-COM convention for non-root bodies). FD across
  the last physics substep, then `v_at_COM = v_at_L + ω × (R ·
  com_offset_local)`; SMPL-specific COM table in
  `scripts/robot_chain.py`. The FD window is invalidated on
  teleport-like operations so post-teleport reads never
  differentiate across a discontinuous state change.
- **Fixed-cadence policy loop with wall-driven pacing.** The
  outer loop accumulates `wall_dt * --time-scale` and drains it
  in `control_dt`-sized chunks via `policy_tick()`, with a
  per-iter clamp so a stall can't dump a backlog on resume.
  Renderer and event polling run once per outer iter, decoupled
  from the policy tick rate. `--time-scale 0` keeps the legacy
  "one tick per outer iter" behaviour.
- **Out-of-process ovrtx renderer** in
  `scripts/remote_renderer.py`: the parent owns ovphysx + ONNX +
  MotionLib; a child subprocess owns ovrtx + pygame and gets
  per-tick body transforms over a bounded `multiprocessing.Queue`
  (drop-oldest-on-full). Required because the two stacks ship
  incompatible vendored carb plugin sets and cannot coexist in a
  single Python process. Clean-shutdown handshake mirrors
  NVIDIA's `SimReadyBrowser`.
- **Blender-style orbit camera** in the renderer: MMB orbit,
  Shift+MMB pan, wheel zoom, `F` to snap to character, `O` to
  toggle follow mode (preserves orbit state).

### Added

- **`--control-dt` / `--physics-dt` / `--drive-type`** CLI
  overrides on `scripts/app.py`. Each defaults to `None` and
  falls through to the YAML / `cfg.drive_type`; when set, applied
  immediately after `load_config_for_onnx` so every downstream
  consumer (`PhysxWorld(dt_physics=...)`, `self.dt_policy`,
  `motion.get_future(dt, step_indices=...)`, the USD drive
  authoring) sees the corrected value. Each fires a WARNING when
  set. A startup warning also fires when `control_dt /
  physics_dt` isn't a near-integer (the PhysxWorld substep
  counter rounds and a bad ratio silently drifts the effective
  per-tick sim time). These exist as the runtime fix for
  sidecars whose exporter wrote the wrong cadence and as an A/B
  knob for `"force"` vs `"acceleration"` PD without re-authoring
  USD.
- **`--draw-mimic-pose`** flag overlays the motion-reference
  body positions as 24 bright spheres in the renderer so you can
  eyeball where the policy thinks each joint should be vs where
  PhysX actually puts it (same quantity the tracking-error
  metric averages over).
- **Per-body tracking-error diagnostic** in `policy_tick`: the
  mean per-body L2 distance between sim and motion-reference
  body positions, accumulated each tick and logged every
  `--tracking-log-every` ticks (default 30). Same quantity
  ProtoMotions logs as `gt_err`; healthy SMPL tracking is
  0.05–0.15 m.
- **`_check_pd_drift`** warns once if the policy's per-step
  `stiffness_targets` / `damping_targets` diverge from the
  USD-authored defaults — PPP keeps gains static because ovphysx
  has no per-step PD-gain binding.
- `CHANGELOG.md` and the `.cursor/rules/changelog.mdc` rule
  requiring future user-visible changes to be logged here in the
  same response that makes the change.

### Fixed

- **PD drive type defaults to `"force"`** in
  `scripts/config_loader.py` (`ResolvedConfig.drive_type`) and
  `JointGains` in `scripts/assets.py`. An earlier patch had set
  this to `"acceleration"` thinking it mirrored IsaacLab's
  `ImplicitActuatorCfg`; per the IsaacLab docs that's wrong —
  `ImplicitActuator` hands `stiffness` / `damping` to PhysX as-is
  and inherits the drive type from the USD, and the upstream
  `smpl_humanoid.usda` authors `"force"`. `"acceleration"` makes
  PhysX multiply the gain by each joint's effective inertia, so
  a hip with ~5 kg·m² of reflected leg inertia turns
  `stiffness=800` into an effective ~4000 N·m/rad — ~5× what the
  policy was trained against.
- **`deployment/export_bm_tracker_onnx.py`** (trainer side) no
  longer runs the simulator config through
  `update_simulator_config_for_test(..., new_simulator="mujoco")`
  before reading `sim.fps` / `sim.decimation` for the YAML's
  `timing.*` block. The MuJoCo conversion overwrote the
  training-time cadence (e.g. 30 Hz on IsaacLab) with MuJoCo's
  defaults (50 Hz / 1 kHz), so the exported YAML lied about the
  cadence the policy was trained at and any deploy host that
  trusted it diverged immediately. Now reads
  `simulator_config.sim.fps` / `.decimation` directly. Pairs
  with the new `--control-dt` / `--physics-dt` overrides so
  pre-fix YAMLs can be corrected at run time without a
  re-export.
- **`MotionPlayer` no longer lifts `ref_respawn_offset_z`** onto
  the reference body positions it returns. Only the character's
  spawn root gets the 5 cm clearance (in `InferenceApp.reset`);
  the mimic targets and the tracking-error reference use the
  un-lifted motion-local pose, matching ProtoMotions'
  `get_spawn_to_ref_pose_offset_with_terrain_height_correction`.
  `reset()` also passes the motion's per-DOF velocities to
  `world.set_dof_positions` so walking motions keep stride
  momentum at frame 0.
- **`ObsBuilder.build`** returns a defensive `np.array(...,
  copy=True)` for every input. Previously `np.ascontiguousarray`
  returned the same buffer for already-contiguous float32
  sources (which `_history_buf` is), so subsequent in-place
  `push_action` mutations leaked back into
  `feed['historical_actions']`. ONNX Runtime had already
  snapshotted the inputs before returning, so the actual policy
  call was unaffected; this only fixes the introspection
  surface.
- **Character no longer falls through the ground.** Switched to
  the canonical ovphysx static-ground pattern (a dedicated
  `Plane "GroundCollider"` with `PhysicsCollisionAPI`) and
  stripped `PhysicsRigidBodyAPI` from the upstream checkerboard;
  also pinned an explicit `PhysicsScene` with `gravityMagnitude
  = 9.81`. The previous `physics:rigidBodyEnabled = 0` +
  `physics:collisionEnabled = 1` chain on the checkerboard
  silenced the inertia warning but did not register the
  descendant mesh as an active collider.
- Renderer install path: `app.py` logs the correct
  `pip install https://pypi.nvidia.com/ovrtx/...` command at
  ERROR level when ovrtx is missing, and the import failure is
  re-raised through `RemoteViewer` so it's never silently
  swallowed.

### Removed

- **Replay / diagnostic infrastructure stripped.**
  `ppp/debug_replay.py`, `scripts/smoke_obs.py`, the experimental
  FK body-velocity backend (`--body-velocity-mode`), the
  deferred-controller order (`--controller-order`), the
  direct-PD path (`apply_direct_pd_torques`,
  `ARTICULATION_DOF_ACTUATION_FORCE` binding), the startup
  diagnostics dump (`--dump-diagnostics`, `ppp/diagnostics.py`),
  every `--replay-*` flag and its supporting code in
  `PhysxWorld`, `RobotChain`, and `InferenceApp` — all gone.
  `run_inference.py` exposes only what motion playback actually
  needs.
- Sleep-based pacing and Windows `timeBeginPeriod(1)` /
  `timeEndPeriod(1)` calls — replaced by the wall-driven
  variable-dt accumulator above.
- The `import ovrtx` preload from `scripts/__init__.py`; the
  subprocess design sidesteps the "ovrtx before ovphysx" import
  constraint entirely.

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
