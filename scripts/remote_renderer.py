"""Out-of-process ovrtx renderer.

``ovphysx`` and ``ovrtx`` ship incompatible carb plugin stacks (different
vendored versions of ``carb.dictionary``, ``carb.settings``, ``omni.fabric``,
``omni.usdphysics``, ...). Whichever loads first wins the carb singletons and
the other one fails to initialize:

* ``ovphysx`` first  → ``ovrtx`` crashes in ``IStageReaderWriter`` (access
  violation) when constructing ``Renderer()``.
* ``ovrtx`` first    → ``ovphysx`` fails with "PhysX plugins could not be
  loaded" because ``omni::physx::IPhysx`` / ``omni::physics::schema::IUsdPhysics``
  dependencies cannot be resolved.

To get both, we run them in separate processes:

* **Parent**: ``ovphysx`` + ``onnxruntime`` + ``MotionLib`` + ``torch``.
* **Child**:  ``ovrtx`` + ``pygame``, owns the visualization window.

Per policy tick the parent ships ``(body_pos, body_quat_xyzw, root_pos,
delta_time)`` over a ``multiprocessing.Queue`` and drains keyboard events
back from the child. On Windows ``multiprocessing`` uses ``spawn`` by
default, so the child gets a fresh interpreter with no ovphysx state.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


log = logging.getLogger(__name__)


# Protocol tags exchanged across the queue. Kept short to minimize pickle size.
_TAG_XFORM = "x"
_TAG_QUIT = "q"
_TAG_READY = "r"
_TAG_BYE = "b"
_TAG_STARTUP_ERROR = "se"
_TAG_RENDER_ERROR = "re"
_TAG_EVENT_QUIT = "eq"
_TAG_EVENT_RESET = "er"
_TAG_EVENT_TOGGLE_FOLLOW = "ef"


# ----------------------------------------------------------------------
# Child process entry point
# ----------------------------------------------------------------------
def _child_main(
    in_q: "mp.Queue",
    out_q: "mp.Queue",
    usd_path: str,
    body_prim_paths: List[str],
    camera_prim: str,
    window_size: Tuple[int, int],
    log_level: str,
    mimic_target_prim_paths: Optional[List[str]] = None,
) -> None:
    """Renderer subprocess: never imports ``ovphysx``.

    The child speaks the tiny protocol defined above. It pulls the latest
    transforms off the queue (draining stale frames so we never lag behind
    the parent), pushes them to ovrtx, renders one frame, blits to pygame,
    then forwards any keyboard events back to the parent.
    """
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s[child]: %(message)s",
    )

    try:
        from scripts.renderer import RtxViewer
    except Exception as exc:
        out_q.put((_TAG_STARTUP_ERROR, f"import RtxViewer failed: {exc!r}"))
        return

    try:
        viewer = RtxViewer(
            usd_path=usd_path,
            body_prim_paths=body_prim_paths,
            camera_prim=camera_prim,
            window_size=tuple(window_size),
            mimic_target_prim_paths=mimic_target_prim_paths,
        )
    except Exception as exc:
        out_q.put((_TAG_STARTUP_ERROR, f"RtxViewer construction failed: {exc!r}"))
        return

    out_q.put((_TAG_READY,))

    latest: Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], float]
    ] = None
    quit_requested = False

    while not quit_requested:
        # Drain everything currently in the queue and keep only the freshest
        # xform message; older ones are dropped so we never render stale state.
        msg = None
        if latest is None:
            # No frames yet — block briefly waiting for the first one.
            try:
                msg = in_q.get(timeout=0.25)
            except queue.Empty:
                msg = None
        else:
            try:
                msg = in_q.get_nowait()
            except queue.Empty:
                msg = None

        while msg is not None:
            tag = msg[0]
            if tag == _TAG_QUIT:
                quit_requested = True
                break
            elif tag == _TAG_XFORM:
                _, body_pos, body_quat, root_pos, mimic_pos, dt = msg
                latest = (body_pos, body_quat, root_pos, mimic_pos, float(dt))
            try:
                msg = in_q.get_nowait()
            except queue.Empty:
                msg = None

        if quit_requested:
            break
        if latest is None:
            continue

        body_pos, body_quat, root_pos, mimic_pos, dt = latest
        try:
            viewer.push_body_transforms(body_pos, body_quat)
            viewer.update_follow_camera(root_pos)
            if mimic_pos is not None:
                viewer.push_mimic_target_positions(mimic_pos)
            viewer.render(dt)
        except Exception as exc:
            out_q.put((_TAG_RENDER_ERROR, repr(exc)))
            quit_requested = True
            break

        events = viewer.poll_events()
        if events.get("quit"):
            out_q.put((_TAG_EVENT_QUIT,))
            quit_requested = True
        if events.get("reset"):
            out_q.put((_TAG_EVENT_RESET,))
        if events.get("toggle_follow"):
            out_q.put((_TAG_EVENT_TOGGLE_FOLLOW,))

    try:
        viewer.close()
    except Exception:
        pass

    # Acknowledge clean shutdown so the parent can join() without resorting
    # to terminate(). The crash-reporter in ovrtx's carb framework fires a
    # noisy ``[crash] A crash has occurred`` when the renderer is killed
    # mid-op, even with exit code 0; the bye handshake lets us avoid that.
    try:
        out_q.put((_TAG_BYE,))
    except Exception:
        pass


# ----------------------------------------------------------------------
# Parent-side handle
# ----------------------------------------------------------------------
class RemoteViewer:
    """Parent-side wrapper that mimics :class:`RtxViewer`'s interface.

    Drop-in for the in-process viewer in ``app.py``: same
    ``push_body_transforms`` / ``update_follow_camera`` / ``render`` /
    ``poll_events`` / ``close`` surface. The render call sends the latest
    transforms to the child and returns immediately; the child does the
    actual ovrtx step at its own pace.
    """

    def __init__(
        self,
        usd_path: Path | str,
        body_prim_paths: Sequence[str],
        camera_prim: str = "/Render/Camera",
        window_size: Tuple[int, int] = (1280, 720),
        startup_timeout: float = 180.0,
        log_level: str = "INFO",
        mimic_target_prim_paths: Optional[Sequence[str]] = None,
    ) -> None:
        ctx = mp.get_context("spawn")
        self._in_q: "mp.Queue" = ctx.Queue(maxsize=4)
        self._out_q: "mp.Queue" = ctx.Queue()
        self._proc = ctx.Process(
            target=_child_main,
            name="ppp-ovrtx-renderer",
            args=(
                self._in_q,
                self._out_q,
                str(Path(usd_path).resolve()),
                list(body_prim_paths),
                camera_prim,
                tuple(window_size),
                log_level,
                list(mimic_target_prim_paths) if mimic_target_prim_paths else None,
            ),
            daemon=True,
        )
        self._proc.start()
        log.info(
            "Spawned ovrtx renderer subprocess (pid=%s); waiting up to %.0fs for "
            "shader compile + ready signal.",
            self._proc.pid,
            startup_timeout,
        )

        deadline = time.monotonic() + startup_timeout
        ready = False
        while time.monotonic() < deadline:
            if not self._proc.is_alive():
                raise RuntimeError(
                    "Renderer subprocess exited during startup "
                    f"(exit code {self._proc.exitcode})."
                )
            try:
                msg = self._out_q.get(timeout=1.0)
            except queue.Empty:
                continue
            tag = msg[0]
            if tag == _TAG_READY:
                ready = True
                break
            if tag == _TAG_STARTUP_ERROR:
                self._terminate()
                raise RuntimeError(f"Renderer subprocess startup error: {msg[1]}")
            # Any other tag this early is unexpected; log and keep waiting.
            log.warning("Unexpected message from renderer subprocess: %r", msg)

        if not ready:
            self._terminate()
            raise TimeoutError(
                "Renderer subprocess did not signal ready within "
                f"{startup_timeout:.0f}s."
            )

        self._closed = False
        self._pending_pos: Optional[np.ndarray] = None
        self._pending_quat: Optional[np.ndarray] = None
        self._pending_root: Optional[np.ndarray] = None
        self._pending_mimic: Optional[np.ndarray] = None
        log.info("Remote ovrtx renderer ready.")

    # ------------------------------------------------------------------
    # Drop-in interface mirroring RtxViewer.
    # ------------------------------------------------------------------
    def push_body_transforms(
        self, body_pos: np.ndarray, body_quat_xyzw: np.ndarray
    ) -> None:
        # Copy so the parent can reuse its read_state buffers without racing
        # the queue's pickler.
        self._pending_pos = np.asarray(body_pos, dtype=np.float64).copy()
        self._pending_quat = np.asarray(body_quat_xyzw, dtype=np.float64).copy()

    def update_follow_camera(self, root_pos: np.ndarray) -> None:
        self._pending_root = np.asarray(root_pos, dtype=np.float64).copy()

    def push_mimic_target_positions(self, positions: np.ndarray) -> None:
        """Stage the next per-body motion-reference positions for the overlay.

        Mirrors :meth:`RtxViewer.push_mimic_target_positions` from the parent
        side: positions are buffered locally and shipped with the next
        ``render`` call. ``None`` (the default after construction) tells the
        child to skip the marker write that frame.
        """
        self._pending_mimic = np.asarray(positions, dtype=np.float64).copy()

    def render(self, delta_time: float) -> None:
        if self._pending_pos is None or self._pending_quat is None:
            return
        root = (
            self._pending_root
            if self._pending_root is not None
            else self._pending_pos[0]
        )
        msg = (
            _TAG_XFORM,
            self._pending_pos,
            self._pending_quat,
            root,
            self._pending_mimic,
            float(delta_time),
        )
        # Non-blocking put with overflow handling: if the child is behind
        # (queue full), drop the oldest frame so we always ship the latest.
        try:
            self._in_q.put_nowait(msg)
        except queue.Full:
            try:
                _ = self._in_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._in_q.put_nowait(msg)
            except queue.Full:
                pass

    def poll_events(self) -> dict:
        actions = {"quit": False, "reset": False, "toggle_follow": False}
        try:
            while True:
                msg = self._out_q.get_nowait()
                tag = msg[0]
                if tag == _TAG_EVENT_QUIT:
                    actions["quit"] = True
                elif tag == _TAG_EVENT_RESET:
                    actions["reset"] = True
                elif tag == _TAG_EVENT_TOGGLE_FOLLOW:
                    actions["toggle_follow"] = True
                elif tag == _TAG_RENDER_ERROR:
                    log.error("Remote renderer error: %s", msg[1])
                    actions["quit"] = True
        except queue.Empty:
            pass
        if not self._proc.is_alive():
            log.warning(
                "Renderer subprocess died (exit code %s); treating as quit.",
                self._proc.exitcode,
            )
            actions["quit"] = True
        return actions

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        # Drain any in-flight xforms so the child can react to ``quit``
        # quickly instead of working through a backlog first.
        try:
            while True:
                self._in_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._in_q.put_nowait((_TAG_QUIT,))
        except Exception:
            pass
        # Wait briefly for the child to acknowledge with ``bye`` (which it
        # sends from the renderer loop after closing its viewer). This lets
        # the child finish the in-flight ovrtx step before we move on.
        deadline = time.monotonic() + 5.0
        saw_bye = False
        while time.monotonic() < deadline:
            try:
                msg = self._out_q.get(timeout=0.5)
            except queue.Empty:
                if not self._proc.is_alive():
                    break
                continue
            if msg[0] == _TAG_BYE:
                saw_bye = True
                break
        # ovrtx's renderer destructor itself can hang or fire its breakpad
        # crash reporter at process tear-down (a known issue documented in
        # NVIDIA's SimReadyBrowser README under the same dual-SDK setup).
        # After bye we give it a short grace window and then fall back to
        # terminate/kill — this is the *expected* clean-exit path here, not
        # a bug, so we log at INFO.
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            log.info(
                "Renderer subprocess terminating after bye (saw_bye=%s) — ovrtx "
                "shutdown is expected to require terminate(). Any "
                "'[carb.crashreporter-breakpad] [crash]' line in stderr is "
                "noise from ovrtx's tear-down, not a real failure.",
                saw_bye,
            )
            self._terminate()
        else:
            log.info("Renderer subprocess exited cleanly.")

    def _terminate(self) -> None:
        try:
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
