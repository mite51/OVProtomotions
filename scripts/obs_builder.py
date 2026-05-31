"""Assemble the raw context tensors the unified-pipeline ONNX consumes.

With the new export the ONNX graph bakes the observation computation
inside itself, so PPP no longer needs to re-implement
``compute_humanoid_max_coords_observations`` / ``build_max_coords_target_poses``
/ ``compute_historical_actions_from_state`` on the host side. We just
feed raw simulator state, raw mimic future state, and a small
``historical_actions`` ring buffer.

The input contract is driven by the YAML sidecar (``policy_inputs``).
For each entry we look up the dotted ``key`` (``current.rigid_body_pos``,
``mimic.future_pos``, ``historical.actions``, ``ground_heights``) and
fill the corresponding tensor from the inputs of :meth:`build`. This
keeps the builder forward-compatible with future exports that add or
drop inputs — only the YAML changes.

See ``docs/onnx_input_migration.md`` for the full input rundown and the
training-time semantics of each tensor.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from .config_loader import PolicyInputSpec, ResolvedConfig
from .motion import MotionFuture
from .physx_world import PhysxState


log = logging.getLogger(__name__)


# Dotted context keys we know how to populate. Anything else in the
# YAML's ``policy_inputs`` raises during ``build`` so a silent shape
# mismatch can't sneak in.
KEY_RIGID_BODY_POS = "current.rigid_body_pos"
KEY_RIGID_BODY_ROT = "current.rigid_body_rot"
KEY_RIGID_BODY_VEL = "current.rigid_body_vel"
KEY_RIGID_BODY_ANG_VEL = "current.rigid_body_ang_vel"
KEY_GROUND_HEIGHTS = "ground_heights"
KEY_HISTORICAL_ACTIONS = "historical.actions"
KEY_MIMIC_FUTURE_POS = "mimic.future_pos"
KEY_MIMIC_FUTURE_ROT = "mimic.future_rot"
KEY_MIMIC_FUTURE_VEL = "mimic.future_vel"
KEY_MIMIC_FUTURE_ANG_VEL = "mimic.future_ang_vel"

_KNOWN_KEYS = {
    KEY_RIGID_BODY_POS,
    KEY_RIGID_BODY_ROT,
    KEY_RIGID_BODY_VEL,
    KEY_RIGID_BODY_ANG_VEL,
    KEY_GROUND_HEIGHTS,
    KEY_HISTORICAL_ACTIONS,
    KEY_MIMIC_FUTURE_POS,
    KEY_MIMIC_FUTURE_ROT,
    KEY_MIMIC_FUTURE_VEL,
    KEY_MIMIC_FUTURE_ANG_VEL,
}


class ObsBuilder:
    """Build the ONNX feed dict for one policy tick.

    Args:
        config: The :class:`ResolvedConfig` parsed from the YAML
            sidecar — supplies the per-input specs (names, keys, shapes)
            and the action / body / DOF dimensions.
    """

    def __init__(self, config: ResolvedConfig) -> None:
        self._cfg = config
        self._num_bodies = config.num_bodies
        self._num_dofs = config.num_dofs

        # Resolve the historical-actions buffer size from the YAML.
        # The spec carries the full shape ``[B, H, A]``; if no
        # ``historical.actions`` input is configured we keep ``H = 0``
        # and skip the bookkeeping entirely.
        hist_spec = config.get_input_spec_by_key(KEY_HISTORICAL_ACTIONS)
        self._history_steps = (
            int(hist_spec.shape[1]) if (hist_spec and len(hist_spec.shape) >= 3)
            else 0
        )
        self._history_buf = np.zeros(
            (1, max(1, self._history_steps), self._num_dofs),
            dtype=np.float32,
        )

        unknown = [
            spec.key for spec in config.policy_inputs if spec.key not in _KNOWN_KEYS
        ]
        if unknown:
            raise RuntimeError(
                "ObsBuilder doesn't know how to populate these ONNX inputs "
                f"declared in the YAML: {unknown}. Extend obs_builder.py to "
                "handle them (or strip them from the export config)."
            )

        log.info(
            "ObsBuilder: %d inputs, history_steps=%d (action_dim=%d).",
            len(config.policy_inputs),
            self._history_steps,
            self._num_dofs,
        )

    # ------------------------------------------------------------------
    @property
    def history_steps(self) -> int:
        return self._history_steps

    @property
    def input_specs(self) -> List[PolicyInputSpec]:
        return list(self._cfg.policy_inputs)

    # ------------------------------------------------------------------
    def reset_history(self) -> None:
        """Zero the historical-actions buffer (called on env / motion reset).

        Matches what the trainer's ``StateHistoryBuffer`` does on a clean
        episode reset (see ``protomotions/envs/base_env/env.py::
        _reset_state_history``).
        """
        self._history_buf.fill(0.0)

    def push_action(self, actions_1xA: np.ndarray) -> None:
        """Push the latest **raw** policy ``actions`` into the history buffer.

        Per ``docs/onnx_input_migration.md``:

        > This is **raw** policy action (pre-PD-scaling). [...] On the
        > deployment side you typically just store the previous ONNX
        > ``actions`` output and feed it back next step.

        Layout matches the trainer's ``state_history_buffer``:

        - slot 0 = most recent past action
        - slot ``h - 1`` = oldest past action

        New actions land at slot 0 and older entries shift right. With
        ``history_steps == 1`` this collapses to "store the previous
        action"; deeper history rolls correctly.
        """
        if self._history_steps == 0:
            return
        arr = np.asarray(actions_1xA, dtype=np.float32).reshape(1, self._num_dofs)
        if self._history_steps > 1:
            self._history_buf[:, 1:, :] = self._history_buf[:, :-1, :]
        self._history_buf[:, 0, :] = arr

    # ------------------------------------------------------------------
    def build(
        self,
        state: PhysxState,
        future: MotionFuture,
        ground_height: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Return the dict ``onnx_name -> np.ndarray`` for one policy tick.

        Args:
            state: Current PhysX state snapshot (per-body world pos /
                rot / vel / ang_vel, kinematic-info ordering).
            future: Motion-reference future state at the configured
                ``future_step_indices`` (shape ``(1, F, N_b, *)``).
            ground_height: Terrain Z at the root XY. For the flat-ground
                scene PPP composes today this is just ``0.0``.

        The output dict is keyed by the ONNX input names declared in
        the YAML (the same names ``OnnxPolicy.input_names`` reports),
        so callers can hand it straight to onnxruntime.

        See ``docs/onnx_input_migration.md`` -> "What each new input is"
        for the per-tensor frame / units. Quaternions are XYZW (the
        common ProtoMotions convention).
        """
        body_pos = np.ascontiguousarray(state.body_pos, dtype=np.float32)
        body_rot = np.ascontiguousarray(state.body_rot, dtype=np.float32)
        body_lin = np.ascontiguousarray(state.body_lin_vel, dtype=np.float32)
        body_ang = np.ascontiguousarray(state.body_ang_vel, dtype=np.float32)

        future_pos = future.rigid_body_pos.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        future_rot = future.rigid_body_rot.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        future_vel = future.rigid_body_vel.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        future_ang = future.rigid_body_ang_vel.detach().cpu().numpy().astype(
            np.float32, copy=False
        )

        ground = np.asarray([float(ground_height)], dtype=np.float32)

        per_key: Dict[str, np.ndarray] = {
            KEY_RIGID_BODY_POS: body_pos[None, ...],          # (1, L, 3)
            KEY_RIGID_BODY_ROT: body_rot[None, ...],          # (1, L, 4) xyzw
            KEY_RIGID_BODY_VEL: body_lin[None, ...],
            KEY_RIGID_BODY_ANG_VEL: body_ang[None, ...],
            KEY_GROUND_HEIGHTS: ground,                       # (1,)
            KEY_MIMIC_FUTURE_POS: future_pos,                 # (1, F, L, 3)
            KEY_MIMIC_FUTURE_ROT: future_rot,
            KEY_MIMIC_FUTURE_VEL: future_vel,
            KEY_MIMIC_FUTURE_ANG_VEL: future_ang,
            KEY_HISTORICAL_ACTIONS: self._history_buf,        # (1, H, A)
        }

        feed: Dict[str, np.ndarray] = {}
        for spec in self._cfg.policy_inputs:
            arr = per_key[spec.key]
            expected = tuple(int(d) for d in spec.shape)
            if expected and arr.shape != expected:
                raise RuntimeError(
                    f"Shape mismatch for input '{spec.name}' (key {spec.key!r}): "
                    f"got {arr.shape}, expected {expected}."
                )
            # ``ascontiguousarray`` returns the *same* array when the
            # input is already C-contiguous float32 (which our
            # ``_history_buf`` is). We need a copy here so that
            # downstream ``push_action`` mutations of the history
            # buffer don't leak into the feed dict the caller may
            # still be inspecting (e.g. the per-tick diagnostic dump
            # would otherwise show the post-push buffer instead of
            # what the policy actually saw at run time).
            feed[spec.name] = np.array(arr, dtype=np.float32, copy=True)
        return feed
