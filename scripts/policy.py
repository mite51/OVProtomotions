"""ONNX policy wrapper around the ProtoMotions SMPL motion-tracker model.

The exported model has 3 inputs and 4 outputs. Names depend on how the model
was exported. For ``smpl_policy_orig/model.onnx`` (the original export shipped
in the ProtoMotions repo) the inputs are:

- ``max_coords_obs``         (1, 358)
- ``mimic_target_poses``     (1, 577)
- ``historical_previous_actions``  (1, 69)

And the outputs are:

- ``normal``   — sampled action (Gaussian)
- ``tanh``     — deterministic mean action: ``tanh(mu)``
- ``neg_1``    — neglogp (unused here)
- ``linear_11`` — value (unused here)

The ``mean_action`` semantic output (the ONNX node literally named ``tanh``)
is the post-tanh action ``tanh(mu_model_output)`` — that is the value
ProtoMotions inference passes to ``env.step`` and that gets multiplied by
``pi`` to produce the PD position target, *and* the value the
``state_history`` buffer stores. ``ppp/app.py::policy_tick`` feeds it
into both ``ObsBuilder.push_action`` and ``ActionProcessor.process``.

This module also patches the model graph in memory to expose the input of
the ``Tanh`` op as a separate ``mean_action_raw`` output. That value is the
**pre-tanh** ``mu_model`` linear output; it is *not* fed into the obs (an
earlier version of this loop did, which mismatched protomotions and caused
the character to drift off-distribution within ~30 frames — see the
``debug_output.json`` analysis in ``CHANGELOG.md``). It is retained for
diagnostic logging only — it makes the "Action stats: raw_mu range=…"
log line in ``policy_tick`` meaningful and gives operators an early-warning
signal if the raw linear output starts climbing past ~2–3 (which would
indicate the policy is feeding back off-distribution for some other
reason).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


log = logging.getLogger(__name__)


class OnnxPolicy:
    """Loads ``model.onnx`` and exposes inference of ``mean_action`` (= raw ``mu``)
    and ``mean_action_tanh`` (= ``tanh(mu)``) on the same forward pass.
    """

    # Name we use for the freshly-exposed raw mu (Tanh's input) graph output.
    RAW_MU_NAME = "mean_action_raw"

    def __init__(
        self,
        onnx_path: str | Path,
        providers: Optional[List[str]] = None,
        prefer_mean_action: bool = True,
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

        # Sibling metadata, if present, maps onnx_name -> semantic_name.
        self._meta: Dict[str, dict] = {}
        meta_path = self._path.with_suffix(".json")
        if meta_path.exists():
            try:
                self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
                log.info("Loaded model metadata from %s", meta_path)
            except Exception as e:  # pragma: no cover
                log.warning("Failed to parse %s: %s", meta_path, e)

        self._prefer_mean_action = prefer_mean_action

        # ------------------------------------------------------------------
        # Load + patch model so the raw mu (Tanh's input) is also an output.
        # ------------------------------------------------------------------
        model_bytes, exposed_raw = self._load_and_patch_graph()
        self._session = ort.InferenceSession(model_bytes, providers=providers)

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

        self._tanh_name = self._resolve_mean_action_output()
        self._raw_mu_name: Optional[str] = (
            self.RAW_MU_NAME if exposed_raw else None
        )
        if self._raw_mu_name is None:
            log.warning(
                "Could not expose raw pre-tanh mu output; falling back to "
                "tanh output for the action-history buffer. The trained "
                "policy expects raw mu — expect drift/tracking failure."
            )
        log.info(
            "Using ONNX outputs: tanh='%s', raw_mu='%s'",
            self._tanh_name,
            self._raw_mu_name,
        )

    # ------------------------------------------------------------------
    def _load_and_patch_graph(self) -> Tuple[bytes, bool]:
        """Load the ONNX model and ensure the Tanh's input tensor is also a
        graph output (so we can read raw ``mu`` at inference).

        Returns:
            (serialized_model_bytes, exposed_raw): the bytes to hand to
            ``onnxruntime.InferenceSession`` and a flag indicating whether we
            successfully added the raw-mu output. If we couldn't find a Tanh
            node feeding any output, returns the original bytes and ``False``.
        """
        try:
            import onnx
        except Exception as e:  # pragma: no cover
            log.warning(
                "onnx package not importable (%s); using original model "
                "without raw-mu output.",
                e,
            )
            return self._path.read_bytes(), False

        model = onnx.load(str(self._path))
        graph = model.graph

        # Find a Tanh op whose output is one of the graph outputs (preferring
        # an output named 'tanh' or 'mean_action').
        existing_outputs = {o.name for o in graph.output}
        tanh_node = None
        for node in graph.node:
            if node.op_type != "Tanh" or not node.output:
                continue
            if node.output[0] in existing_outputs:
                tanh_node = node
                break

        if tanh_node is None or not tanh_node.input:
            log.warning(
                "No Tanh op feeding a graph output found in %s. The history "
                "buffer will receive tanh(mu) instead of raw mu, which does "
                "not match training.",
                self._path.name,
            )
            return self._path.read_bytes(), False

        raw_mu_internal = tanh_node.input[0]

        # If the raw mu tensor is already an output (or already aliased to our
        # canonical name), do nothing.
        if raw_mu_internal in existing_outputs or self.RAW_MU_NAME in existing_outputs:
            return model.SerializeToString(), True

        # Insert an Identity op that aliases the raw mu tensor into a new
        # graph output with our canonical name. Identity is cheap and avoids
        # naming collisions with any internal consumer of the original tensor.
        identity_node = onnx.helper.make_node(
            "Identity",
            inputs=[raw_mu_internal],
            outputs=[self.RAW_MU_NAME],
            name=f"{self.RAW_MU_NAME}_alias",
        )
        graph.node.append(identity_node)

        # Mirror the shape/dtype of the tanh output (they share dims since
        # Tanh is elementwise).
        tanh_out_info = None
        for o in graph.output:
            if o.name == tanh_node.output[0]:
                tanh_out_info = o
                break
        if tanh_out_info is None:
            log.warning(
                "Tanh output %s not present in graph.output list; skipping "
                "raw-mu exposure.",
                tanh_node.output[0],
            )
            return self._path.read_bytes(), False

        new_output = onnx.helper.make_tensor_value_info(
            self.RAW_MU_NAME,
            tanh_out_info.type.tensor_type.elem_type,
            [
                d.dim_value if (d.dim_value or not d.dim_param) else d.dim_param
                for d in tanh_out_info.type.tensor_type.shape.dim
            ],
        )
        graph.output.append(new_output)

        log.info(
            "Patched ONNX graph: exposed raw mu '%s' as new output '%s' "
            "(via Identity alias).",
            raw_mu_internal,
            self.RAW_MU_NAME,
        )
        return model.SerializeToString(), True

    # ------------------------------------------------------------------
    def _resolve_mean_action_output(self) -> str:
        """Pick the output name that corresponds to the deterministic tanh action.

        Preference order:
        1. metadata.output_mapping[onnx_name] == "mean_action".
        2. an output literally named ``mean_action`` or ``tanh``.
        3. the second output of the model (ProtoMotions convention: outputs
           are ``[normal, tanh, neg_1, linear_11]``).
        """
        if self._prefer_mean_action and "output_mapping" in self._meta:
            mapping = self._meta["output_mapping"]
            for onnx_name, semantic in mapping.items():
                if semantic == "mean_action":
                    return onnx_name

        for candidate in ("mean_action", "tanh"):
            if candidate in self.output_names:
                return candidate

        if len(self.output_names) >= 2:
            return self.output_names[1]
        return self.output_names[0]

    # ------------------------------------------------------------------
    def _build_feed(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feed: Dict[str, np.ndarray] = {}
        unresolved = [n for n in self.input_names if n not in obs]
        if not unresolved:
            for name in self.input_names:
                feed[name] = np.ascontiguousarray(obs[name], dtype=np.float32)
        else:
            obs_keys = list(obs.keys())
            if len(obs_keys) != len(self.input_names):
                raise RuntimeError(
                    f"ONNX inputs {self.input_names!r} do not match provided "
                    f"observation keys {obs_keys!r}."
                )
            for onnx_name, obs_key in zip(self.input_names, obs_keys):
                feed[onnx_name] = np.ascontiguousarray(
                    obs[obs_key], dtype=np.float32
                )
        return feed

    # ------------------------------------------------------------------
    def run(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run inference and return the tanh-bounded mean action ``tanh(mu)``.

        Shape: ``(1, action_dim)``.

        Kept for backward compatibility; prefer ``run_action_outputs`` when
        you need both the raw ``mu`` (for the history buffer) and the
        tanh-bounded value (for the PD post-process).
        """
        feed = self._build_feed(obs)
        outputs = self._session.run([self._tanh_name], feed)
        return outputs[0]

    # ------------------------------------------------------------------
    def run_action_outputs(
        self, obs: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference and return ``(raw_mu, tanh_mu)``.

        - ``raw_mu``: ``(1, action_dim)`` — the network's pre-tanh
          ``mu_model`` linear output. Diagnostic only; ``policy_tick`` uses
          it for the "Action stats: raw_mu range=…" log line and *not* for
          the obs buffer (an earlier version of the loop did, which
          mismatched protomotions — see ``CHANGELOG.md``).
        - ``tanh_mu``: ``(1, action_dim)`` — ``tanh(mu)`` in ``[-1, 1]``.
          This is the post-tanh ``mean_action``: feed it both into
          :meth:`ObsBuilder.push_action` (the history buffer stores the
          post-tanh value, with a 2-step read-back lag) and into
          :class:`ActionProcessor` to produce PD targets ``pi * tanh_mu``.

        If the model couldn't be patched to expose raw mu, falls back to
        returning ``(tanh_mu, tanh_mu)`` so callers don't crash. The "Action
        stats" log will then report ``raw_mu`` ranges identical to
        ``tanh_mu`` (i.e. always within ``[-1, 1]``), which is harmless but
        loses the diagnostic value of the unbounded linear output.
        """
        feed = self._build_feed(obs)
        names = [self._tanh_name]
        if self._raw_mu_name is not None:
            names.append(self._raw_mu_name)
        outputs = self._session.run(names, feed)
        tanh_mu = outputs[0]
        raw_mu = outputs[1] if len(outputs) > 1 else outputs[0]
        return raw_mu, tanh_mu
