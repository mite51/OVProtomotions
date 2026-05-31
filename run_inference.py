"""CLI entry-point for PPP inference.

Mirrors the invocation style of ``protomotions/inference_agent.py``:

    python run_inference.py \\
        --checkpoint "C:\\Git\\ProtoMotions\\data\\pretrained_models\\motion_tracker\\smpl\\last.ckpt" \\
        --motion-file "C:\\Git\\ProtoMotions\\data\\yaml_files\\ACCAD\\body_jab_left.motion" \\
        --onnx "C:\\Git\\ProtoMotions\\onnx\\smpl_policy_orig\\model.onnx"
"""

from scripts.app import cli_main


if __name__ == "__main__":
    cli_main()
