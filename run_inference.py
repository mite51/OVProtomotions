"""CLI entry-point for PPP inference (unified-pipeline ONNX).

Mirrors the invocation style of ``protomotions/inference_agent.py`` but
no longer requires the trainer checkpoint — everything PPP needs at
runtime (joint / body schema, PD gains, control timing, future step
indices) is read from the YAML sidecar auto-discovered next to the
``--onnx`` file.

Typical usage:

    python run_inference.py \\
        --onnx "C:\\Git\\OVProtomotions\\models\\model.onnx" \\
        --motion-file "C:\\Git\\ProtoMotions\\data\\yaml_files\\ACCAD\\body_jab_left.motion"

YAML discovery order (next to ``--onnx``):

    1. <onnx_stem>.yaml             (canonical export-script naming)
    2. unified_pipeline.yaml        (export-script default file name)
    3. any single *.yaml            (fallback when neither above exists)

Pass ``--yaml <path>`` to override.

See ``docs/onnx_input_migration.md`` for the new input contract.
"""

from scripts.app import cli_main


if __name__ == "__main__":
    cli_main()
