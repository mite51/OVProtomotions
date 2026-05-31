"""Action post-processing.

The ONNX model exports a ``tanh`` ("mean_action") output already in the
``[-1, 1]`` range. For the SMPL motion-tracker checkpoint the resolved config
ships:

- ``action_transform = 'tanh'`` (already applied inside the ONNX graph)
- ``pd_action_offset = [0]*69``
- ``pd_action_scale  = [pi]*69``

So the PD position targets used by the simulator are simply ``pi * mean_action``.

This module exposes a small helper that does exactly that, plus a per-DOF
range clamp for safety.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class ActionConfig:
    """PD-action parameters extracted from the resolved inference config."""

    pd_action_offset: List[float] = field(
        default_factory=lambda: [0.0] * 69
    )
    pd_action_scale: List[float] = field(
        default_factory=lambda: [math.pi] * 69
    )
    apply_tanh: bool = False  # ONNX graph already applies tanh.
    clamp_value: float = 1.0  # Used only if apply_tanh is False.

    @classmethod
    def from_resolved_config(cls, env_cfg) -> "ActionConfig":
        """Build from a resolved ``EnvConfig.action_config`` dict / dataclass.

        Falls back to ``[0]*69`` offset and ``[pi]*69`` scale if the values
        are not present in the structure (matches the SMPL motion-tracker
        defaults).
        """
        ac = getattr(env_cfg, "action_config", None) or {}

        def _list_of_floats(v, n_default: int, default: float) -> List[float]:
            if v is None:
                return [default] * n_default
            try:
                import torch  # noqa: F401 - imported lazily in case torch is unused.
                if hasattr(v, "tolist"):
                    return [float(x) for x in v.tolist()]
            except Exception:
                pass
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v]
            return [float(v)] * n_default

        offset = _list_of_floats(ac.get("pd_action_offset"), 69, 0.0) if isinstance(
            ac, dict
        ) else _list_of_floats(getattr(ac, "pd_action_offset", None), 69, 0.0)
        scale = _list_of_floats(ac.get("pd_action_scale"), 69, math.pi) if isinstance(
            ac, dict
        ) else _list_of_floats(getattr(ac, "pd_action_scale", None), 69, math.pi)

        action_transform = (
            ac.get("action_transform") if isinstance(ac, dict)
            else getattr(ac, "action_transform", None)
        )
        # If the ONNX graph already applies tanh (the standard PPO export path),
        # we don't apply it again here. Heuristic: if action_transform == "tanh",
        # assume the export wraps it.
        apply_tanh = action_transform != "tanh"

        clamp_value = (
            ac.get("clamp_value", 1.0) if isinstance(ac, dict)
            else float(getattr(ac, "clamp_value", 1.0) or 1.0)
        )

        return cls(
            pd_action_offset=offset,
            pd_action_scale=scale,
            apply_tanh=apply_tanh,
            clamp_value=float(clamp_value),
        )


class ActionProcessor:
    """Stateless mapping from raw policy output to PD position targets."""

    def __init__(self, config: ActionConfig) -> None:
        self._offset = np.asarray(config.pd_action_offset, dtype=np.float32)
        self._scale = np.asarray(config.pd_action_scale, dtype=np.float32)
        self._apply_tanh = config.apply_tanh
        self._clamp = float(config.clamp_value)
        if self._offset.shape != self._scale.shape:
            raise ValueError(
                f"pd_action_offset shape {self._offset.shape} != "
                f"pd_action_scale shape {self._scale.shape}"
            )
        self._num_actions = self._offset.shape[0]

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def process(self, action: np.ndarray) -> np.ndarray:
        """Convert a ``(1, A)`` policy output into ``(1, A)`` PD targets."""
        a = np.asarray(action, dtype=np.float32).reshape(-1, self._num_actions)
        if self._apply_tanh:
            a = np.tanh(a)
        else:
            a = np.clip(a, -self._clamp, self._clamp)

        return self._offset + self._scale * a
