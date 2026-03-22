"""EmbeddedRenderer: renders MuJoCo frames into a DearPyGUI texture.

Uses mujoco.Renderer (offscreen, no GLFW) in a single daemon thread.
The render thread owns the Renderer instance exclusively — it is not
thread-safe, so no locking is needed around the C-level render state.

Usage:
    renderer = EmbeddedRenderer("my_texture_tag")
    renderer.register_texture()   # call before dpg.setup_dearpygui()
    # ... build DPG layout, add dpg.add_image("my_texture_tag") ...
    renderer.load_trajectory(scenario, params, cache)   # after each sim
    renderer.seek(frame)          # responds to scrub slider
    renderer.play() / pause()    # playback controls
    renderer.close()              # on shutdown
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import dearpygui.dearpygui as dpg
import mujoco
import numpy as np

from mjgrok.scenarios.base import Scenario
from mjgrok.simulation.trajectory import TrajectoryCache


class EmbeddedRenderer:
    RENDER_W: int = 640
    RENDER_H: int = 480

    def __init__(self, texture_tag: str) -> None:
        self._texture_tag = texture_tag
        self._lock = threading.Lock()

        # Trajectory state — protected by _lock
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._qpos_traj: np.ndarray | None = None
        self._qvel_traj: np.ndarray | None = None
        self._n_frames: int = 0
        self._current_frame: int = 0
        self._dt: float = 0.002
        self._playing: bool = False

        # Pending load — overwritten by latest load_trajectory() call
        self._pending: tuple[Scenario, dict[str, Any], TrajectoryCache] | None = None

        # Pixel buffer — render thread writes, main thread flushes to DPG.
        # Metal requires GPU texture uploads on the main thread.
        self._pixel_lock = threading.Lock()
        self._pending_pixels: list | None = None  # RGBA float32 list, ready for set_value

        self._on_frame: Callable[[int], None] | None = None

        # Events
        self._render_event = threading.Event()
        self._stop_event = threading.Event()

        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="EmbeddedRenderer"
        )
        self._render_thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def register_texture(self) -> None:
        """Register a dynamic DPG texture. Must be called before dpg.setup_dearpygui().

        Uses add_dynamic_texture which is designed for frequent per-frame updates.
        Always RGBA float32 in [0, 1].
        """
        default = [0.0] * (self.RENDER_H * self.RENDER_W * 4)
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=self.RENDER_W,
                height=self.RENDER_H,
                default_value=default,
                tag=self._texture_tag,
            )

    def set_on_frame(self, callback: Callable[[int], None]) -> None:
        self._on_frame = callback

    def load_trajectory(
        self,
        scenario: Scenario,
        params: dict[str, Any],
        cache: TrajectoryCache,
    ) -> None:
        """Queue a trajectory load. Thread-safe; only the latest call takes effect.

        Preserves current frame position (clamped) and playing state so that
        parameter tuning during playback doesn't interrupt the viewer.
        """
        with self._lock:
            self._pending = (scenario, params, cache)
        self._render_event.set()

    def seek(self, frame: int) -> None:
        """Seek to a frame and render it. Thread-safe."""
        with self._lock:
            self._current_frame = max(0, min(frame, self._n_frames - 1))
        self._render_event.set()

    def play(self) -> None:
        with self._lock:
            self._playing = True
        self._render_event.set()

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def step_forward(self) -> int:
        with self._lock:
            self._current_frame = min(self._current_frame + 1, self._n_frames - 1)
            frame = self._current_frame
        self._render_event.set()
        return frame

    def step_backward(self) -> int:
        with self._lock:
            self._current_frame = max(self._current_frame - 1, 0)
            frame = self._current_frame
        self._render_event.set()
        return frame

    def flush_to_dpg(self) -> None:
        """Upload the latest rendered frame to the DPG texture.

        Must be called from the main thread (e.g. in a viewport render callback).
        Metal requires GPU texture uploads to happen on the render/main thread.
        """
        with self._pixel_lock:
            pixels = self._pending_pixels
            self._pending_pixels = None
        if pixels is not None:
            dpg.set_value(self._texture_tag, pixels)

    def close(self) -> None:
        self._stop_event.set()
        self._render_event.set()
        self._render_thread.join(timeout=2.0)
        with self._lock:
            self._close_renderer()

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def n_frames(self) -> int:
        return self._n_frames

    # ── Render thread ─────────────────────────────────────────────────────────

    def _render_loop(self) -> None:
        from traceback import print_exc

        fps = 30.0
        frame_interval = 1.0 / fps

        while not self._stop_event.is_set():
            self._render_event.wait(timeout=frame_interval)
            self._render_event.clear()

            if self._stop_event.is_set():
                break

            try:
                # Apply any pending load first
                self._apply_pending()

                with self._lock:
                    has_renderer = self._renderer is not None
                    frame = self._current_frame
                    playing = self._playing
                    at_end = frame >= self._n_frames - 1 if self._n_frames > 0 else True

                if not has_renderer:
                    continue

                # Render the current frame
                self._render_frame(frame)

                if playing:
                    if at_end:
                        with self._lock:
                            self._current_frame = 0
                        if self._on_frame:
                            self._on_frame(0)
                        self._render_event.set()
                    else:
                        time.sleep(frame_interval)
                        with self._lock:
                            frame_step = max(1, round(frame_interval / self._dt))
                            self._current_frame = min(
                                self._current_frame + frame_step, self._n_frames - 1
                            )
                            next_frame = self._current_frame
                        if self._on_frame:
                            self._on_frame(next_frame)
                        # Signal ourselves to render the next frame
                        self._render_event.set()

            except Exception:  # noqa: BLE001
                print("[EmbeddedRenderer] render thread error:")
                print_exc()

    def _apply_pending(self) -> None:
        """Apply a queued load_trajectory() call. Runs on the render thread."""
        with self._lock:
            pending = self._pending
            self._pending = None

        if pending is None:
            return

        scenario, params, cache = pending

        # Rebuild renderer for the new model
        new_model = scenario.build_model(params)
        nq = sum(1 for k in cache.series_arr if k.startswith("qpos_"))
        nv = sum(1 for k in cache.series_arr if k.startswith("qvel_"))
        qpos = np.column_stack([cache.series_arr[f"qpos_{i}"] for i in range(nq)])
        qvel = np.column_stack([cache.series_arr[f"qvel_{i}"] for i in range(nv)])

        new_renderer = mujoco.Renderer(new_model, self.RENDER_H, self.RENDER_W)
        new_data = mujoco.MjData(new_model)

        with self._lock:
            prev_frame = self._current_frame
            self._close_renderer()
            self._model = new_model
            self._data = new_data
            self._renderer = new_renderer
            self._qpos_traj = qpos
            self._qvel_traj = qvel
            self._n_frames = cache.frame_count()
            # Preserve frame position across reloads (e.g. parameter tuning during playback)
            self._current_frame = min(prev_frame, self._n_frames - 1) if self._n_frames > 0 else 0
            self._dt = cache.times[1] - cache.times[0] if len(cache.times) >= 2 else 0.002

    def _render_frame(self, frame: int) -> None:
        """Render one frame and push pixels to the DPG texture."""
        with self._lock:
            if self._renderer is None or self._qpos_traj is None:
                return
            model = self._model
            data = self._data
            renderer = self._renderer
            qpos_row = self._qpos_traj[frame]
            qvel_row = self._qvel_traj[frame]

        data.qpos[: len(qpos_row)] = qpos_row
        data.qvel[: len(qvel_row)] = qvel_row
        mujoco.mj_forward(model, data)

        renderer.update_scene(data)
        raw = renderer.render()  # (H, W, 3) uint8

        # Convert to RGBA float32 list. Metal requires GPU uploads on the main
        # thread, so we store here and flush via flush_to_dpg() from the main thread.
        rgba = np.ones((self.RENDER_H, self.RENDER_W, 4), dtype=np.float32)
        rgba[:, :, :3] = raw.astype(np.float32) / 255.0
        with self._pixel_lock:
            self._pending_pixels = rgba.ravel().tolist()

    def _close_renderer(self) -> None:
        """Close and release the renderer. Must be called with _lock held."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._model = None
        self._data = None
        self._qpos_traj = None
        self._qvel_traj = None
        self._n_frames = 0
        self._current_frame = 0
