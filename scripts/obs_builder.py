"""Build the 3 ONNX policy inputs (max_coords_obs, mimic_target_poses,
historical_previous_actions) from PhysX state + reference motion.

All math is delegated to ProtoMotions to guarantee numerical parity with the
checkpoint the ONNX file was exported from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from .motion import MotionFuture
from .physx_world import PhysxState


@dataclass
class ObsInputs:
    """Numpy arrays ready to feed into the ONNX session.

    Shapes match the policy metadata:

    - ``max_coords_obs``: (1, 358)
    - ``mimic_target_poses``: (1, 577) for ``future_steps=1``
    - ``historical_previous_actions``: (1, 69)
    """

    max_coords_obs: np.ndarray
    mimic_target_poses: np.ndarray
    historical_previous_actions: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "max_coords_obs": self.max_coords_obs,
            "mimic_target_poses": self.mimic_target_poses,
            "historical_previous_actions": self.historical_previous_actions,
        }


class ObsBuilder:
    """Builds policy observations using protomotions' obs functions.

    Single-env: every output starts with a leading dim of 1.

    Args:
        num_bodies: Robot body count (24 for SMPL).
        num_dofs: Robot DOF count (69 for SMPL).
        action_dim: Action dim (== num_dofs for SMPL).
        device: torch device used for intermediate tensors (CPU is fine).
        local_obs: Match training (``True``).
        root_height_obs: Match training (``True``).
        observe_contacts: Match training (``False``).
        with_velocities: Match training (``True``).
        with_relative: Match training (``True``).
        future_steps: Match training (``1``).
        w_last: ``True`` (XYZW quaternions).
    """

    def __init__(
        self,
        num_bodies: int,
        num_dofs: int,
        action_dim: int,
        device: torch.device | str = "cpu",
        local_obs: bool = True,
        root_height_obs: bool = True,
        observe_contacts: bool = False,
        with_velocities: bool = True,
        with_relative: bool = True,
        future_steps: int = 1,
        w_last: bool = True,
    ) -> None:
        from protomotions.envs.obs.humanoid import (
            compute_humanoid_max_coords_observations,
        )
        from protomotions.envs.obs.target_poses import build_max_coords_target_poses
        from protomotions.envs.obs.humanoid_historical import (
            compute_historical_actions_from_state,
        )

        self._max_coords_fn = compute_humanoid_max_coords_observations
        self._target_poses_fn = build_max_coords_target_poses
        self._historical_actions_fn = compute_historical_actions_from_state

        self.num_bodies = num_bodies
        self.num_dofs = num_dofs
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.local_obs = local_obs
        self.root_height_obs = root_height_obs
        self.observe_contacts = observe_contacts
        self.with_velocities = with_velocities
        self.with_relative = with_relative
        self.future_steps = future_steps
        self.w_last = w_last

        # Persistent ``(envs=1, num_state_history_steps + 1, action_dim)``
        # buffer of previous-action slots. Layout matches ProtoMotions'
        # :class:`StateHistoryBuffer` exactly:
        #
        # - slot 0 = current step's action (just rotated in)
        # - slot 1 = action from one step ago
        # - ``historical_actions`` (used by the obs) = ``slots[1:]``
        #
        # For the SMPL motion-tracker checkpoint ``num_state_history_steps=1``
        # so ``buffer_size=2``. ``compute_historical_actions_from_state`` is
        # then called with ``history_steps=1`` and reads slot 1 only.
        #
        # Why the obs sees ``action[N-2]`` and not ``action[N-1]``: protomotions
        # rotates this buffer with ``_current_raw_action`` *inside* ``env.step``
        # (see ``protomotions/envs/base_env/env.py::post_physics_step``) and
        # only *then* builds the obs that gets returned to the caller. That
        # obs is consumed by the *next* policy call, so frame N's policy sees
        # ``slot 1 = action[N-2]``. Empirically verified against the captured
        # ``debug_output.json`` (shift -2 matches all 128 frames to fp32; every
        # other shift drifts by 0.10–0.28).
        self._history_buf = torch.zeros(
            1, 2, action_dim, dtype=torch.float32, device=self.device
        )

    # ------------------------------------------------------------------
    # State-history maintenance
    # ------------------------------------------------------------------
    def reset_history(self) -> None:
        """Zero out both slots of the previous-action buffer (called on
        motion / env reset)."""
        self._history_buf.zero_()

    def seed_history(
        self,
        slot1: torch.Tensor | np.ndarray,
        slot0: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Pre-populate the action-history slots from a known capture.

        Used by the replay harness to align the first 2 ticks of obs
        with the trainer's mid-episode capture. The trainer's
        :class:`StateHistoryBuffer` carries warmup actions across
        frame 0; those warmup actions aren't recorded as standalone
        frames in ``debug_output.json``, but they *are* visible in
        ``frame[N].historical_previous_actions`` as
        ``actions[N - 2]``. Seeding the buffer with frames 0 and 1's
        ``historical_previous_actions`` reproduces the warmup state
        exactly:

        - ``slot1`` should be ``frame[0].historical_previous_actions``
          (== ``actions[-2]``). The first ``build()`` reads slot 1, so
          tick 0's obs match the capture.
        - ``slot0`` should be ``frame[1].historical_previous_actions``
          (== ``actions[-1]``). The first :meth:`push_action` rotates
          slot 0 → slot 1, so tick 1 also matches the capture. Pass
          ``None`` (default) to leave slot 0 untouched, which only
          fixes tick 0.

        For live inference no seeding is possible (no captured
        warmup); see :func:`InferenceApp.reset` docstring for why
        zeros are still trainer-equivalent on a fresh env reset.
        """
        s1 = torch.as_tensor(
            slot1, dtype=torch.float32, device=self.device
        ).reshape(1, self.action_dim)
        self._history_buf[:, 1] = s1
        if slot0 is not None:
            s0 = torch.as_tensor(
                slot0, dtype=torch.float32, device=self.device
            ).reshape(1, self.action_dim)
            self._history_buf[:, 0] = s0

    def push_action(self, post_tanh_action_69: torch.Tensor | np.ndarray) -> None:
        """Rotate the action history and store the latest **post-tanh**
        ``mean_action`` in slot 0.

        Pipeline contract (must match ProtoMotions inference):

        - Caller passes the ONNX ``tanh`` output (= ``mean_action`` =
          ``tanh(mu_model_output)``), the same value that gets multiplied by
          ``pi`` to produce the PD position target.
        - We rotate ``slot 1 <- slot 0`` and write the new action into slot 0.
        - The *next* :meth:`build` call reads slot 1 only (via
          ``self._history_buf[:, 1:]``) for the
          ``historical_previous_actions`` obs, giving the policy the
          2-step-delayed action it was trained on.

        The previous version pushed the pre-tanh ``raw_mu`` and read it back
        immediately (1-step lag), which mismatches both fields the policy
        cares about. Diagnosed against
        ``c:/Dev/Protomotion3/Assets/onnx/smpl_policy/debug_output.json``:

        - ``historical_previous_actions[N] == actions[N-2]`` exactly for all
          128 captured frames (shift -2 matches to fp32; -1 is off by up to
          0.76).
        - ``actions[N]`` equals the ONNX ``tanh`` output (post-tanh) and
          ``pd_targets[N] = pi * actions[N]`` to fp32 — no extra tanh in the
          PD path, so the buffer must store the post-tanh value.
        """
        arr = torch.as_tensor(
            post_tanh_action_69, dtype=torch.float32, device=self.device
        ).reshape(1, self.action_dim)
        self._history_buf[:, 1] = self._history_buf[:, 0]
        self._history_buf[:, 0] = arr

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _state_to_torch(self, state: PhysxState) -> dict:
        """Convert a ``PhysxState`` to (1, ...) torch tensors."""
        def t(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        body_pos = t(state.body_pos).unsqueeze(0)              # (1, L, 3)
        body_rot = t(state.body_rot).unsqueeze(0)              # (1, L, 4)
        body_vel = t(state.body_lin_vel).unsqueeze(0)          # (1, L, 3)
        body_ang_vel = t(state.body_ang_vel).unsqueeze(0)      # (1, L, 3)
        return dict(
            body_pos=body_pos,
            body_rot=body_rot,
            body_vel=body_vel,
            body_ang_vel=body_ang_vel,
        )

    # ------------------------------------------------------------------
    # Build all 3 inputs
    # ------------------------------------------------------------------
    def build(
        self,
        state: PhysxState,
        future: MotionFuture,
        ground_height: float = 0.0,
        body_contacts: torch.Tensor | None = None,
        motion_time: float | None = None,
        motion_length: float | None = None,
        future_dt: float | None = None,
    ) -> ObsInputs:
        """Build the three ONNX inputs from the latest physics state + future ref.

        Args:
            state: Current PhysX state.
            future: Motion reference at ``t + dt * [1..future_steps]``.
            ground_height: Terrain Z at the root XY (0 for flat ground).
            body_contacts: Per-body contact flags (unused when
                ``observe_contacts=False``).
            motion_time: Current motion time ``t`` (seconds). Required for the
                trailing time-to-target scalar in ``mimic_target_poses``.
            motion_length: Total motion length (seconds). Required when
                ``motion_time`` is provided.
            future_dt: Step size used to build ``future``. Defaults to
                ``1/30`` if not provided.
        """
        s = self._state_to_torch(state)

        if body_contacts is None:
            # observe_contacts is False during training, so the value never
            # affects the obs vector. We still need a shape-correct dummy.
            body_contacts = torch.zeros(1, 0, dtype=torch.bool, device=self.device)

        ground_h = torch.full(
            (1,), float(ground_height), dtype=torch.float32, device=self.device
        )

        max_coords_obs = self._max_coords_fn(
            body_pos=s["body_pos"],
            body_rot=s["body_rot"],
            body_vel=s["body_vel"],
            body_ang_vel=s["body_ang_vel"],
            ground_height=ground_h,
            body_contacts=body_contacts,
            local_obs=self.local_obs,
            root_height_obs=self.root_height_obs,
            observe_contacts=self.observe_contacts,
            w_last=self.w_last,
        )

        future_pos = future.rigid_body_pos.to(self.device)            # (1, F, L, 3)
        future_rot = future.rigid_body_rot.to(self.device)            # (1, F, L, 4)
        future_vel = future.rigid_body_vel.to(self.device)
        future_ang_vel = future.rigid_body_ang_vel.to(self.device)

        mimic_target_poses = self._target_poses_fn(
            current_state_body_pos=s["body_pos"],
            current_state_body_rot=s["body_rot"],
            current_state_body_vel=s["body_vel"],
            current_state_body_ang_vel=s["body_ang_vel"],
            mimic_ref_pos=future_pos,
            mimic_ref_rot=future_rot,
            mimic_ref_vel=future_vel,
            mimic_ref_ang_vel=future_ang_vel,
            with_velocities=self.with_velocities,
            w_last=self.w_last,
            future_steps=self.future_steps,
            with_relative=self.with_relative,
        )

        # The trained checkpoint expects one extra "time-to-target" scalar per
        # future step appended to mimic_target_poses (576 -> 577 for future=1).
        # See protomotions <= v3.0 MimicObs.add_time_to_target_poses.
        if motion_time is not None and motion_length is not None:
            dt = float(future_dt) if future_dt is not None else 1.0 / 30.0
            offsets = torch.arange(
                1,
                self.future_steps + 1,
                device=self.device,
                dtype=torch.float32,
            ) * dt
            future_times = torch.clamp(
                torch.tensor(float(motion_time), device=self.device) + offsets,
                max=float(motion_length),
            )
            time_to_target = (
                future_times - float(motion_time)
            ).reshape(1, self.future_steps)
            mimic_target_poses = torch.cat(
                [
                    mimic_target_poses.reshape(1, self.future_steps, -1),
                    time_to_target.unsqueeze(-1),
                ],
                dim=-1,
            ).reshape(1, -1)

        # ProtoMotions: ``historical_actions`` property = ``actions[:, 1:]``,
        # i.e. excludes slot 0 (the current step's action). With ``history_steps=1``
        # ``compute_historical_actions_from_state`` then takes the first entry
        # of *that*, which is slot 1 of the full buffer = the action from one
        # step ago. Combined with the rotate-then-obs ordering in
        # ``post_physics_step``, the policy at frame N receives ``action[N-2]``.
        historical_previous_actions = self._historical_actions_fn(
            historical_actions=self._history_buf[:, 1:],
            history_steps=1,
        )

        def np32(t: torch.Tensor) -> np.ndarray:
            return t.detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32)

        return ObsInputs(
            max_coords_obs=np32(max_coords_obs),
            mimic_target_poses=np32(mimic_target_poses),
            historical_previous_actions=np32(historical_previous_actions),
        )
