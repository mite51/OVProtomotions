"""Real-time visualization through ovrtx + pygame.

ovrtx renders to an RTX render product whose ``LdrColor`` var we map back to
the CPU each tick and blit into a small pygame window. Body transforms are
written into the rendered stage every frame so the visualized humanoid tracks
the PhysX-simulated one.

The ovrtx and ovphysx instances do not share a USD stage at runtime, but they
were both loaded from the same source ``.usda`` file so prim paths match.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np


log = logging.getLogger(__name__)


class RtxViewer:
    """ovrtx renderer + pygame window with body-xform updates each frame.

    Args:
        usd_path: Path to the composed runtime USD (same file ovphysx loads).
        body_prim_paths: Per-body USD prim paths to push transforms onto, in
            ``kinematic_info.body_names`` order. Length must match the PhysX
            link count.
        camera_prim: USD prim path of the camera to render through.
        window_size: Pygame window size (overrides RTX render product size if
            different — we blit with scaling).
        mimic_target_prim_paths: Optional per-body USD prim paths for the
            mimic-target overlay (rendered by ``--draw-mimic-pose``).
            Must be in the same body order as ``body_prim_paths``. When set,
            :meth:`push_mimic_target_positions` writes per-tick world
            positions to these prims; when ``None`` the overlay is disabled
            and the related write path is a no-op.
    """

    def __init__(
        self,
        usd_path: Path | str,
        body_prim_paths: Sequence[str],
        camera_prim: str = "/Render/Camera",
        camera_xform_prim: str = "/World/Camera",
        window_size: tuple[int, int] = (1280, 720),
        mimic_target_prim_paths: Sequence[str] | None = None,
    ) -> None:
        import ovrtx  # lazy import
        import pygame

        self._ovrtx = ovrtx
        self._pygame = pygame

        self._usd_path = str(Path(usd_path).resolve())
        self._body_paths: List[str] = list(body_prim_paths)
        self._mimic_paths: List[str] = (
            list(mimic_target_prim_paths) if mimic_target_prim_paths else []
        )
        self._camera_prim = camera_prim
        self._camera_xform_prim = camera_xform_prim
        self._window_size = window_size

        log.info("Creating ovrtx.Renderer (first run compiles shaders)...")
        self._renderer = ovrtx.Renderer()
        log.info("Opening USD: %s", self._usd_path)
        self._renderer.open_usd(self._usd_path)

        pygame.init()
        pygame.display.set_caption("PPP — ProtoMotions on ovphysx+ovrtx")
        self._screen = pygame.display.set_mode(window_size)
        self._clock = pygame.time.Clock()

        # Pre-allocate body-transform buffer (N, 4, 4) float64 — populated each
        # frame from the latest PhysX state.
        n = len(self._body_paths)
        self._xform_buf = np.zeros((n, 4, 4), dtype=np.float64)
        for i in range(n):
            self._xform_buf[i] = np.eye(4)

        # Mimic-target overlay buffer. We only ever write identity-rotation
        # translation matrices to the markers (orientation on a sphere is
        # invisible), so the 3x3 block stays as identity for the lifetime
        # of this viewer and ``push_mimic_target_positions`` only updates
        # the translation row.
        m = len(self._mimic_paths)
        self._mimic_xform_buf = np.zeros((m, 4, 4), dtype=np.float64)
        if m > 0:
            self._mimic_xform_buf[:, 0, 0] = 1.0
            self._mimic_xform_buf[:, 1, 1] = 1.0
            self._mimic_xform_buf[:, 2, 2] = 1.0
            self._mimic_xform_buf[:, 3, 3] = 1.0

        # ------------------------------------------------------------------
        # Blender-style orbit camera state.
        #
        # The camera is parameterized by a pivot point in world space plus
        # spherical (azimuth, elevation, distance) coordinates around it,
        # and always looks at the pivot. Z-up world; azimuth is measured
        # CCW from +X when viewed from +Z. Defaults roughly match the
        # camera that's authored in the USD (eye ≈ (3.5, -3.5, 1.5) gazing
        # at the origin around hip height).
        #
        # Mouse bindings (Blender defaults):
        #   - MMB drag           → orbit
        #   - Shift + MMB drag   → pan
        #   - Mouse wheel        → zoom (dolly the eye relative to pivot)
        #   - F                  → snap pivot to character root
        #   - O                  → toggle follow-character mode
        # ------------------------------------------------------------------
        self._cam_pivot = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._cam_distance: float = 5.0
        self._cam_azimuth_deg: float = -45.0  # +X, -Y side
        self._cam_elevation_deg: float = 15.0
        self._cam_min_dist: float = 0.05
        self._cam_max_dist: float = 100.0
        self._cam_elev_limit_deg: float = 89.0  # avoid gimbal flip
        self._orbit_sens_deg_per_pixel: float = 0.4
        # Pan scales with distance so the world tracks the cursor 1:1 in
        # screen space regardless of zoom level. The constant was picked
        # empirically against the default FoV (~35° vertical); tweak if it
        # feels sluggish or hyper.
        self._pan_sens_per_pixel: float = 0.0025
        self._zoom_factor_per_notch: float = 1.10

        # Mouse drag state (left button = orbit alt, middle = canonical
        # Blender, shift modifier turns either into pan).
        self._mmb_held: bool = False
        self._lmb_held: bool = False
        self._last_mouse: tuple[int, int] | None = None

        self._follow_camera: bool = False
        self._latest_root_pos: np.ndarray | None = None

        # Push the initial camera matrix so the renderer doesn't show the
        # stale USD-authored pose for one frame.
        self._write_camera()

    # ------------------------------------------------------------------
    # Stage updates
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_xyzw_to_mat3(qxyzw: np.ndarray) -> np.ndarray:
        """Convert XYZW quaternions ``(N, 4)`` to rotation matrices ``(N, 3, 3)``."""
        q = np.asarray(qxyzw, dtype=np.float64).reshape(-1, 4)
        # Normalize to avoid drift from PhysX accumulating numerical error.
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        q = q / norms
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        n = q.shape[0]
        m = np.empty((n, 3, 3), dtype=np.float64)
        m[:, 0, 0] = 1 - 2 * (y * y + z * z)
        m[:, 0, 1] = 2 * (x * y - z * w)
        m[:, 0, 2] = 2 * (x * z + y * w)
        m[:, 1, 0] = 2 * (x * y + z * w)
        m[:, 1, 1] = 1 - 2 * (x * x + z * z)
        m[:, 1, 2] = 2 * (y * z - x * w)
        m[:, 2, 0] = 2 * (x * z - y * w)
        m[:, 2, 1] = 2 * (y * z + x * w)
        m[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return m

    def push_body_transforms(
        self,
        body_pos: np.ndarray,
        body_quat_xyzw: np.ndarray,
    ) -> None:
        """Write per-body world transforms to the ovrtx stage.

        ovrtx uses USD's row-vector convention: translation in matrix[3][0:3].
        """
        ovrtx = self._ovrtx
        n = len(self._body_paths)
        if body_pos.shape[0] != n or body_quat_xyzw.shape[0] != n:
            raise ValueError(
                f"Expected {n} bodies, got pos {body_pos.shape}, "
                f"quat {body_quat_xyzw.shape}"
            )

        rot = self._quat_xyzw_to_mat3(body_quat_xyzw)
        # The matrix layout below puts the rotation in the upper-left 3x3 and
        # the translation in the last row, matching USD row-vector semantics:
        #   [ R^T | 0 ]
        #   [ t   | 1 ]
        self._xform_buf[:] = 0.0
        self._xform_buf[:, 0:3, 0:3] = rot.transpose(0, 2, 1)
        self._xform_buf[:, 3, 0:3] = body_pos.astype(np.float64, copy=False)
        self._xform_buf[:, 3, 3] = 1.0

        self._renderer.write_attribute(
            prim_paths=list(self._body_paths),
            attribute_name="omni:xform",
            tensor=self._xform_buf,
            semantic=ovrtx.Semantic.XFORM_MAT4x4,
        )

    def push_mimic_target_positions(self, positions: np.ndarray) -> None:
        """Write motion-reference body positions to the overlay markers.

        ``positions`` is ``(N, 3)`` in the same body order as
        ``mimic_target_prim_paths``. No-op when the overlay is disabled
        (no marker prim paths were provided at construction time). We
        deliberately ignore orientation: each marker is a sphere, so the
        rotation block of the 4x4 stays at identity for the life of the
        viewer.
        """
        if not self._mimic_paths:
            return
        ovrtx = self._ovrtx
        n = len(self._mimic_paths)
        if positions.shape[0] != n:
            raise ValueError(
                f"Expected {n} mimic-target positions, got {positions.shape}"
            )
        self._mimic_xform_buf[:, 3, 0:3] = positions.astype(
            np.float64, copy=False
        )
        self._renderer.write_attribute(
            prim_paths=list(self._mimic_paths),
            attribute_name="omni:xform",
            tensor=self._mimic_xform_buf,
            semantic=ovrtx.Semantic.XFORM_MAT4x4,
        )

    # ------------------------------------------------------------------
    # Camera math + IO
    # ------------------------------------------------------------------
    def _compute_camera_matrix(self) -> np.ndarray:
        """Build the world-from-camera xform matrix in USD row-vector form.

        USD row-vector convention: ``p_world = p_local @ M``. The rows of
        ``M[0:3, 0:3]`` are the camera's local axes expressed in world.
        For a USD camera looking down -Z with +Y up:

        - row 0 = camera right (camera-local +X) in world
        - row 1 = camera up    (camera-local +Y) in world
        - row 2 = camera back  (camera-local +Z) in world  = -forward
        - row 3 = camera eye position
        """
        az = float(np.radians(self._cam_azimuth_deg))
        el = float(np.radians(self._cam_elevation_deg))
        ce, se = np.cos(el), np.sin(el)
        ca, sa = np.cos(az), np.sin(az)
        # Unit direction from pivot toward eye (so the camera looks back
        # along -dir_to_eye at the pivot).
        dir_to_eye = np.array([ce * ca, ce * sa, se], dtype=np.float64)
        eye = self._cam_pivot + self._cam_distance * dir_to_eye

        forward = -dir_to_eye  # camera looks from eye toward pivot
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-8:
            # Looking essentially straight up/down — pick an arbitrary
            # tangent. We clamp elevation so this shouldn't fire, but
            # guard anyway.
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
            rn = float(np.linalg.norm(right))
            if rn < 1e-8:
                right = np.array([1.0, 0.0, 0.0])
                rn = 1.0
        right = right / rn
        cam_up = np.cross(right, forward)
        cam_up = cam_up / float(np.linalg.norm(cam_up))

        m = np.eye(4, dtype=np.float64)
        m[0, 0:3] = right
        m[1, 0:3] = cam_up
        m[2, 0:3] = -forward  # camera-local +Z (back of camera)
        m[3, 0:3] = eye
        return m

    def _write_camera(self) -> None:
        m = self._compute_camera_matrix()
        self._renderer.write_attribute(
            prim_paths=[self._camera_xform_prim],
            attribute_name="omni:xform",
            tensor=m.reshape(1, 4, 4),
            semantic=self._ovrtx.Semantic.XFORM_MAT4x4,
        )

    def update_follow_camera(self, root_pos: np.ndarray) -> None:
        """Track the character root and re-emit the camera matrix.

        Called every frame by the renderer subprocess. In follow mode the
        orbit pivot is yanked to the root each tick (orbital params kept,
        so the user keeps their MMB-orbit view, just re-anchored). Out of
        follow mode the pivot stays wherever pan / F-focus left it. We
        unconditionally re-emit because mouse drags between frames also
        dirty the camera; cheap (one 4×4 write per tick).
        """
        self._latest_root_pos = np.asarray(root_pos, dtype=np.float64).copy()
        if self._follow_camera:
            self._cam_pivot = self._latest_root_pos.copy()
        self._write_camera()

    # ------------------------------------------------------------------
    # Render + display
    # ------------------------------------------------------------------
    def render(self, delta_time: float) -> None:
        """Step the renderer one frame and blit the LdrColor into the window."""
        ovrtx = self._ovrtx
        products = self._renderer.step(
            render_products={self._camera_prim},
            delta_time=float(delta_time),
        )
        for _name, product in products.items():
            for frame in product.frames:
                var = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
                pixels = np.from_dlpack(var)
                self._blit(pixels)

    def _blit(self, pixels: np.ndarray) -> None:
        pygame = self._pygame
        # ovrtx LdrColor is RGBA8 in (H, W, 4). pygame surfarray wants (W, H, C).
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            log.warning("Unexpected LdrColor shape: %s", pixels.shape)
            return
        if pixels.shape[2] == 4:
            rgb = pixels[..., :3]
        else:
            rgb = pixels
        # Transpose H,W,C -> W,H,C for surfarray.
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        # Scale to window size.
        if surf.get_size() != self._window_size:
            surf = pygame.transform.scale(surf, self._window_size)
        self._screen.blit(surf, (0, 0))
        pygame.display.flip()
        self._clock.tick(60)

    # ------------------------------------------------------------------
    # Window events
    # ------------------------------------------------------------------
    def poll_events(self) -> dict:
        """Pump pygame events and return a dict of single-shot actions.

        Returns keys: ``quit``, ``reset``, ``toggle_follow``. Mouse-driven
        camera changes are absorbed locally — they only mutate the orbit
        params, the next ``update_follow_camera`` write picks them up.

        Camera bindings (Blender-style):
            MMB drag            orbit
            Shift + MMB drag    pan
            LMB drag            orbit (laptop fallback — no MMB needed)
            Shift + LMB drag    pan
            Mouse wheel         zoom
            F                   focus pivot on character root
            O                   toggle follow-character
        """
        pygame = self._pygame
        actions = {"quit": False, "reset": False, "toggle_follow": False}
        cam_dirty = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions["quit"] = True

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    actions["quit"] = True
                elif event.key == pygame.K_r:
                    actions["reset"] = True
                elif event.key == pygame.K_o:
                    self._follow_camera = not self._follow_camera
                    actions["toggle_follow"] = True
                elif event.key == pygame.K_f:
                    # Snap pivot to the latest root position so the user
                    # can re-frame the character after panning away.
                    if self._latest_root_pos is not None:
                        self._cam_pivot = self._latest_root_pos.copy()
                        cam_dirty = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:  # middle
                    self._mmb_held = True
                    self._last_mouse = event.pos
                elif event.button == 1:  # left (laptop fallback)
                    self._lmb_held = True
                    self._last_mouse = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    self._mmb_held = False
                elif event.button == 1:
                    self._lmb_held = False
                if not (self._mmb_held or self._lmb_held):
                    self._last_mouse = None

            elif event.type == pygame.MOUSEMOTION:
                if self._last_mouse is None:
                    self._last_mouse = event.pos
                if self._mmb_held or self._lmb_held:
                    mx, my = event.pos
                    lx, ly = self._last_mouse
                    dx = mx - lx
                    dy = my - ly
                    mods = pygame.key.get_mods()
                    shift = bool(mods & pygame.KMOD_SHIFT)
                    if shift:
                        self._apply_pan(dx, dy)
                    else:
                        self._apply_orbit(dx, dy)
                    cam_dirty = True
                self._last_mouse = event.pos

            elif event.type == pygame.MOUSEWHEEL:
                # event.y > 0 = scroll up = zoom in (closer to pivot).
                if event.y != 0:
                    self._apply_zoom(event.y)
                    cam_dirty = True

        # Mid-frame mouse input needs to render immediately — paint a fresh
        # camera xform now so the user sees the orbit before the next sim
        # tick arrives (sim may be running at <30 Hz on slower hardware).
        if cam_dirty:
            self._write_camera()
        return actions

    # ------------------------------------------------------------------
    # Camera input handlers (Blender-style)
    # ------------------------------------------------------------------
    def _apply_orbit(self, dx: int, dy: int) -> None:
        """MMB drag → orbit around pivot.

        Blender convention: dragging right rotates the *view* such that
        the scene appears to spin left, which means the camera moves
        clockwise around the pivot (viewed from above). With our azimuth
        measured CCW from +X, that's an azimuth *decrease* for dx > 0.
        Dragging down (dy > 0 on screen) tilts the camera down, which
        means elevation increases (camera rises above the pivot).
        """
        s = self._orbit_sens_deg_per_pixel
        self._cam_azimuth_deg = (self._cam_azimuth_deg - dx * s) % 360.0
        new_el = self._cam_elevation_deg + dy * s
        lim = self._cam_elev_limit_deg
        self._cam_elevation_deg = float(np.clip(new_el, -lim, lim))

    def _apply_pan(self, dx: int, dy: int) -> None:
        """Shift+MMB drag → pan: slide the pivot in the camera's screen plane.

        Drag right → world appears to move right → pivot moves *left*
        along the camera's right axis. We recompute the camera basis
        from the current orbit params rather than caching it, so pan
        composes cleanly with concurrent orbit/zoom edits.
        """
        az = float(np.radians(self._cam_azimuth_deg))
        el = float(np.radians(self._cam_elevation_deg))
        ce, se = np.cos(el), np.sin(el)
        ca, sa = np.cos(az), np.sin(az)
        dir_to_eye = np.array([ce * ca, ce * sa, se], dtype=np.float64)
        forward = -dir_to_eye
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-8:
            return
        right = right / rn
        cam_up = np.cross(right, forward)
        cn = float(np.linalg.norm(cam_up))
        if cn < 1e-8:
            return
        cam_up = cam_up / cn
        scale = self._pan_sens_per_pixel * self._cam_distance
        self._cam_pivot = self._cam_pivot - right * (dx * scale) + cam_up * (dy * scale)

    def _apply_zoom(self, notches: int) -> None:
        """Mouse wheel → dolly. ``notches > 0`` = scroll up = zoom in."""
        # Multiplicative so each notch gives a consistent perceptual step
        # at any current distance.
        factor = self._zoom_factor_per_notch ** (-notches)
        new_dist = self._cam_distance * factor
        self._cam_distance = float(
            np.clip(new_dist, self._cam_min_dist, self._cam_max_dist)
        )

    def close(self) -> None:
        try:
            self._pygame.quit()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
