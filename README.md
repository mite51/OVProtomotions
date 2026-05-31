# OVProtomotions — ProtoMotions Mimic Inference on ovphysx + ovrtx

A standalone Python app that runs the ProtoMotions SMPL motion-tracker policy
(`onnx/smpl_policy_orig/model.onnx`) inside [`ovphysx`](https://nvidia-omniverse.github.io/PhysX/ovphysx/)
for physics and [`ovrtx`](https://github.com/NVIDIA-Omniverse/ovrtx) for
rendering. It replaces the IsaacGym/IsaacLab/Newton dependency of
`protomotions/inference_agent.py` while reusing ProtoMotions' observation
builders, `MotionLib`, and kinematic configs.

## Layout

```
scripts/
  assets.py        # builds a runtime USD scene (ground + smpl + camera + light)
  motion.py        # wraps protomotions.components.motion_lib.MotionLib
  obs_builder.py   # builds the 3 ONNX inputs from sim state + motion ref
  policy.py        # onnxruntime wrapper around model.onnx
  action.py        # mean_action -> 69 PD position targets
  physx_world.py   # ovphysx tensor bindings (read state, write targets, step)
  renderer.py      # ovrtx renderer + pygame window
  app.py           # main loop + CLI entry-point
run_inference.py   # thin shim around scripts.app.cli_main
```

## Install

```powershell
# 1. Editable install of ProtoMotions (reused for math + MotionLib + kinematic_info).
pip install -e C:\Git\ProtoMotions

# 2. Editable install of this project.
cd C:\Git\OVProtomotions
pip install -e .

# 3. ovrtx wheel (from NVIDIA's PyPI index; the GitHub release ships a
#    2 GB binary zip, not a wheel).
pip install https://pypi.nvidia.com/ovrtx/ovrtx-0.3.0.312915-py3-none-win_amd64.whl
```

## Run

```powershell
python run_inference.py `
    --checkpoint "C:\Git\ProtoMotions\data\pretrained_models\motion_tracker\smpl\last.ckpt" `
    --motion-file "C:\Git\ProtoMotions\data\yaml_files\ACCAD\body_jab_left.motion" `
    --onnx "C:\Git\ProtoMotions\onnx\smpl_policy_orig\model.onnx"
```

`--checkpoint` is only used to find `resolved_configs_inference.pt` for the
robot's `kinematic_info` (body/DOF names and order, PD gains).

Add `--draw-mimic-pose` to overlay the motion-reference body positions as
bright orange spheres on top of the simulated character — useful for
eyeballing what the policy is being asked to track at any given moment
(same quantity the tracking-error log line averages over).

## Keys & Mouse

Blender-style orbit camera in the renderer window:

- **MMB drag** — orbit around pivot
- **Shift + MMB drag** — pan (slides the pivot in the screen plane)
- **LMB drag** — orbit fallback for laptops without a middle button
- **Shift + LMB drag** — pan fallback
- **Mouse wheel** — zoom (dolly relative to pivot)
- `F` — focus pivot on the character root
- `O` — toggle follow-character mode (pivot tracks the root each tick)
- `R` — reset to motion frame 0
- `Q` / `Esc` — quit

## Caveats

- ovrtx is pre-release; rendering may fall back to a debug preview if the
  installed wheel and driver are incompatible.
- `ovphysx` and `ovrtx` ship incompatible vendored carb plugin stacks and
  **cannot coexist in a single Python process** — whichever loads first
  wins the carb singletons and the other one fails. This app runs
  `ovphysx` in the main process and `ovrtx` in a child process spawned by
  `scripts/remote_renderer.py`. Body transforms are sent each policy tick
  over `multiprocessing.Queue` (drop-oldest-on-full so the renderer never
  back-pressures the policy loop). NVIDIA's `SimReadyBrowser` solves the
  same conflict the same way, in mirror (their UI process owns `ovrtx`
  and `ovphysx` runs in a child).

## Changelog

Significant changes are tracked in [`CHANGELOG.md`](./CHANGELOG.md).
The `.cursor/rules/changelog.mdc` rule keeps it current as edits happen.
