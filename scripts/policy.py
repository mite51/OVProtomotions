"""ONNX wrapper around the ProtoMotions unified-pipeline tracker model.

The new export (``deployment/export_bm_tracker_onnx.py``) compiles
*everything* — observation computation, actor MLP, and action processing
— into a single ONNX graph. Inputs are raw simulator / mimic context
tensors; outputs are the ready-to-use PD targets:

- ``actions``           — raw policy action (the value PPP feeds back as
                          ``historical_actions`` on the next tick).
- ``joint_pos_targets`` — per-DOF position targets, ready for the
                          simulator's PD controller (no host-side
                          ``pi * tanh(mu)`` post-processing).
- ``stiffness_targets`` — per-DOF PD stiffness from the policy. PPP
                          keeps the gains static (USD-authored, matched
                          to YAML defaults) and only logs a warning if
                          these drift from the defaults.
- ``damping_targets``   — same idea for damping.

PPP defers the entire input/output schema to the YAML sidecar so this
wrapper has no model-specific knowledge — only the four standard
output names are referenced by string.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


log = logging.getLogger(__name__)


OUTPUT_ACTIONS = "actions"
OUTPUT_JOINT_POS_TARGETS = "joint_pos_targets"
OUTPUT_STIFFNESS_TARGETS = "stiffness_targets"
OUTPUT_DAMPING_TARGETS = "damping_targets"


@dataclass
class PolicyOutputs:
    """One forward pass through the unified pipeline.

    All arrays carry a leading batch dim of 1 (single-env inference);
    callers typically index ``[0]`` before handing them to the
    simulator / history buffer.
    """

    actions: np.ndarray              # (1, A) raw policy action
    joint_pos_targets: np.ndarray    # (1, A) PD position targets
    stiffness_targets: Optional[np.ndarray] = None  # (1, A) or None
    damping_targets: Optional[np.ndarray] = None    # (1, A) or None


class OnnxPolicy:
    """Loads ``unified_pipeline.onnx`` and exposes a typed forward pass."""

    def __init__(
        self,
        onnx_path: str | Path,
        providers: Optional[List[str]] = None,
    ) -> None:
        import onnxruntime as ort

        self._path = Path(onnx_path).resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self._path}")

        if providers is None:
            available = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
        log.info("ONNX providers: %s", providers)

        self._session = ort.InferenceSession(str(self._path), providers=providers)

        self.input_names: List[str] = [i.name for i in self._session.get_inputs()]
        self.output_names: List[str] = [o.name for o in self._session.get_outputs()]
        self.input_shapes: Dict[str, List[int | str]] = {
            i.name: list(i.shape) for i in self._session.get_inputs()
        }
        self.output_shapes: Dict[str, List[int | str]] = {
            o.name: list(o.shape) for o in self._session.get_outputs()
        }
        log.info("ONNX inputs: %s", self.input_shapes)
        log.info("ONNX outputs: %s", self.output_shapes)

        for required in (OUTPUT_ACTIONS, OUTPUT_JOINT_POS_TARGETS):
            if required not in self.output_names:
                raise RuntimeError(
                    f"ONNX model is missing the required '{required}' output. "
                    f"Available outputs: {self.output_names}. PPP expects a "
                    "unified-pipeline export "
                    "(deployment/export_bm_tracker_onnx.py)."
                )

    # ------------------------------------------------------------------
    def run(self, feed: Dict[str, np.ndarray]) -> PolicyOutputs:
        """Run one inference pass and return a typed :class:`PolicyOutputs`."""
        missing = [n for n in self.input_names if n not in feed]
        if missing:
            raise RuntimeError(
                f"ONNX feed is missing inputs: {missing}. Provided: "
                f"{list(feed.keys())}."
            )

        names = [OUTPUT_ACTIONS, OUTPUT_JOINT_POS_TARGETS]
        optional = []
        for opt in (OUTPUT_STIFFNESS_TARGETS, OUTPUT_DAMPING_TARGETS):
            if opt in self.output_names:
                names.append(opt)
                optional.append(opt)
        outs = self._session.run(names, feed)

        result = PolicyOutputs(
            actions=outs[0],
            joint_pos_targets=outs[1],
        )
        for opt, arr in zip(optional, outs[2:]):
            if opt == OUTPUT_STIFFNESS_TARGETS:
                result.stiffness_targets = arr
            elif opt == OUTPUT_DAMPING_TARGETS:
                result.damping_targets = arr
        return result
