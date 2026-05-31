"""Wraps ``protomotions.components.motion_lib.MotionLib`` for inference.

For inference we only ever need a single motion (id = 0), queried at
arbitrary times. This module hides the tensor-batch plumbing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import torch


log = logging.getLogger(__name__)


@dataclass
class MotionSample:
    """One snapshot from the reference motion at a single time ``t``.

    Tensors carry a leading batch dim of 1 (matching ProtoMotions' obs
    builders, which were authored for a batch of envs).
    """

    rigid_body_pos: torch.Tensor    # (1, num_bodies, 3)
    rigid_body_rot: torch.Tensor    # (1, num_bodies, 4) XYZW
    rigid_body_vel: torch.Tensor    # (1, num_bodies, 3)
    rigid_body_ang_vel: torch.Tensor  # (1, num_bodies, 3)
    dof_pos: torch.Tensor           # (1, num_dofs)
    dof_vel: torch.Tensor           # (1, num_dofs)


@dataclass
class MotionFuture:
    """Future reference poses with leading dim ``(1, future_steps, ...)``."""

    rigid_body_pos: torch.Tensor    # (1, F, num_bodies, 3)
    rigid_body_rot: torch.Tensor    # (1, F, num_bodies, 4)
    rigid_body_vel: torch.Tensor    # (1, F, num_bodies, 3)
    rigid_body_ang_vel: torch.Tensor  # (1, F, num_bodies, 3)
    dof_pos: torch.Tensor           # (1, F, num_dofs)
    dof_vel: torch.Tensor           # (1, F, num_dofs)


class MotionPlayer:
    """Loads a motion file via ``MotionLib`` and exposes time-indexed queries.

    Args:
        motion_file: Path to ``.motion`` / ``.npz`` / ``.yaml`` motion blob.
        device: torch device used for the MotionLib query tensors.
        get_motion_state_use_blend: Standard MotionLib knob, leave True.
        ref_pos_z_offset: Z (world up) offset added to every reference
            ``rigid_body_pos`` returned by :meth:`get_state` and
            :meth:`get_future`. **Default 0.0**, which mirrors how
            ProtoMotions feeds the obs builder. Specifically,
            ``env.get_spawn_to_ref_pose_offset_with_terrain_height_correction``
            (called from ``mimic_control.py`` to lift the ref pose into the
            env's spawn frame) returns ``(respawn_root_offset.xy,
            terrain_z)`` — the env-spawn XY translation plus terrain
            height, but **NOT** ``ref_respawn_offset`` (the 0.05 m
            character lift). The 0.05 m is only added to the character's
            initial root position at reset, not to ``mimic_ref_pos`` or
            the tracking-error reference. Verified against
            ``debug_output.json``: with this set to 0 the captured
            ``mimic_target_poses`` matches our obs-builder output to fp32
            from frame 1 onward (see commit notes / CHANGELOG). Setting
            this to 0.05 (the pre-fix behaviour) injects a constant
            +5 cm Z bias into ``mimic_ref_pos`` versus what the policy
            saw in training, off-distribution and most visible on
            translation-heavy motions (walking).
    """

    def __init__(
        self,
        motion_file: str | Path,
        device: torch.device | str = "cpu",
        get_motion_state_use_blend: bool = True,
        ref_pos_z_offset: float = 0.0,
    ) -> None:
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig

        self.device = torch.device(device)
        self._cfg = MotionLibConfig(
            motion_file=str(motion_file),
            get_motion_state_use_blend=get_motion_state_use_blend,
        )
        log.info("Loading motion file: %s", motion_file)
        self._lib: MotionLib = MotionLib(self._cfg, device=str(self.device))
        if self._lib.num_motions() == 0:
            raise RuntimeError(f"MotionLib loaded 0 motions from {motion_file!r}.")

        self._motion_id = 0
        self._motion_length = float(self._lib.motion_lengths[self._motion_id].item())
        self._motion_dt = float(self._lib.motion_dt[self._motion_id].item())
        self._ref_pos_z_offset = float(ref_pos_z_offset)
        log.info(
            "Motion 0: %.3fs, dt=%.4fs, frames=%d. Ref body-pos Z-offset = %.4f m.",
            self._motion_length,
            self._motion_dt,
            int(self._lib.motion_num_frames[self._motion_id].item()),
            self._ref_pos_z_offset,
        )

    @property
    def length(self) -> float:
        return self._motion_length

    @property
    def dt(self) -> float:
        return self._motion_dt

    def _ids_times(self, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        t = max(0.0, min(self._motion_length, float(t)))
        motion_ids = torch.tensor(
            [self._motion_id], dtype=torch.long, device=self.device
        )
        motion_times = torch.tensor([t], dtype=torch.float32, device=self.device)
        return motion_ids, motion_times

    @property
    def ref_pos_z_offset(self) -> float:
        """Z-offset added to ``rigid_body_pos`` on every query (see ``__init__``)."""
        return self._ref_pos_z_offset

    def _apply_z_offset(self, body_pos: torch.Tensor) -> torch.Tensor:
        """Add the configured Z offset to every body's Z coordinate.

        Returns a new tensor (clone) so callers can mutate freely without
        touching the underlying MotionLib cache.
        """
        if self._ref_pos_z_offset == 0.0:
            return body_pos
        out = body_pos.clone()
        out[..., 2] += self._ref_pos_z_offset
        return out

    def get_state(self, t: float) -> MotionSample:
        """Reference state at time ``t`` (single-env, batched as ``(1, ...)``).

        ``rigid_body_pos`` is shifted up by ``self.ref_pos_z_offset`` (set at
        construction) so callers receive the same offset-corrected reference
        the training env's ``ctx.mimic.future_pos`` / ``ctx.mimic.ref_state``
        carries.
        """
        motion_ids, motion_times = self._ids_times(t)
        rs = self._lib.get_motion_state(motion_ids, motion_times)
        return MotionSample(
            rigid_body_pos=self._apply_z_offset(rs.rigid_body_pos),
            rigid_body_rot=rs.rigid_body_rot,
            rigid_body_vel=rs.rigid_body_vel,
            rigid_body_ang_vel=rs.rigid_body_ang_vel,
            dof_pos=rs.dof_pos,
            dof_vel=rs.dof_vel,
        )

    def get_future(
        self,
        t: float,
        dt: float,
        n: int = 1,
        step_indices: Sequence[int] | None = None,
    ) -> MotionFuture:
        """Future poses at times ``t + dt * step_indices``, clamped to motion length.

        Reference body positions are Z-offset (see :meth:`get_state`).

        Args:
            t: Current motion time in seconds.
            dt: Step size in seconds (typically the trained control_dt).
            n: Convenience shorthand for ``step_indices=[1..n]``. Ignored
                when ``step_indices`` is provided explicitly.
            step_indices: Explicit list of integer offsets (in units of
                ``dt``) at which to sample the motion. Mirrors the
                trainer's ``MimicControlConfig.future_step_indices``
                ([1] = "one control step ahead"). Required when the
                unified-pipeline YAML declares non-contiguous offsets.
        """
        if step_indices is not None:
            indices = list(step_indices)
        else:
            if n <= 0:
                raise ValueError(f"n must be > 0, got {n}.")
            indices = list(range(1, n + 1))

        n_steps = len(indices)
        if n_steps == 0:
            raise ValueError("step_indices must contain at least one offset.")

        ids = torch.zeros(n_steps, dtype=torch.long, device=self.device)
        offsets = torch.tensor(
            indices, device=self.device, dtype=torch.float32
        ) * dt
        times = torch.clamp(t + offsets, min=0.0, max=self._motion_length)

        rs = self._lib.get_motion_state(ids, times)

        # rs.* is shape (n, num_bodies, ...) — reshape to (1, n, num_bodies, ...).
        return MotionFuture(
            rigid_body_pos=self._apply_z_offset(rs.rigid_body_pos).unsqueeze(0),
            rigid_body_rot=rs.rigid_body_rot.unsqueeze(0),
            rigid_body_vel=rs.rigid_body_vel.unsqueeze(0),
            rigid_body_ang_vel=rs.rigid_body_ang_vel.unsqueeze(0),
            dof_pos=rs.dof_pos.unsqueeze(0),
            dof_vel=rs.dof_vel.unsqueeze(0),
        )
