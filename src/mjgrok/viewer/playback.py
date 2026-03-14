"""ViewerController: offscreen MuJoCo rendering via mujoco.Renderer → DPG texture."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

import dearpygui.dearpygui as dpg
import mujoco
import numpy as np

from mjgrok.scenarios.base import Scenario
from mjgrok.simulation.trajectory import TrajectoryCache

RENDER_W = 640
RENDER_H = 480
TEX_TAG = "viewer_texture"


def create_viewer_texture() -> None:
    """Register the DPG raw texture. Must be called from the main thread during UI setup."""
    blank = [0.1] * (RENDER_W * RENDER_H * 4)
    with dpg.texture_registry(tag="viewer_tex_registry"):
        dpg.add_raw_texture(
            width=RENDER_W,
            height=RENDER_H,
            default_value=blank,
            tag=TEX_TAG,
            format=dpg.mvFormat_Float_rgba,
        )


class ViewerController:
    def __init__(self) -> None:
        self._renderer: mujoco.Renderer | None = None
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._cache: TrajectoryCache | None = None
        self._current_frame: int = 0
        self._play_thread: threading.Thread | None = None
        self._stop_play = threading.Event()
        self._lock = threading.Lock()

    def load(self, scenario: Scenario, params: dict[str, Any], cache: TrajectoryCache) -> None:
        """Build model, create renderer, render first frame. Safe to call from any thread."""
        self.pause()

        model = scenario.build_model(params)
        data = mujoco.MjData(model)

        with contextlib.suppress(Exception):
            if self._renderer is not None:
                self._renderer.close()

        renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

        with self._lock:
            self._model = model
            self._data = data
            self._cache = cache
            self._renderer = renderer
            self._current_frame = 0

        self._render_frame(0)

    def close(self) -> None:
        self.pause()
        with self._lock:
            if self._renderer is not None:
                with contextlib.suppress(Exception):
                    self._renderer.close()
                self._renderer = None
            self._model = None
            self._data = None
            self._cache = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._cache is not None

    def seek(self, frame_idx: int) -> None:
        if self._cache is None:
            return
        frame_idx = max(0, min(frame_idx, self._cache.frame_count() - 1))
        with self._lock:
            self._current_frame = frame_idx
        self._render_frame(frame_idx)

    def play(self, realtime_factor: float = 1.0) -> None:
        self.pause()
        self._stop_play.clear()

        def _loop() -> None:
            if self._cache is None or self._model is None:
                return
            # Cap render rate at ~60 fps regardless of timestep
            frame_dt = max(self._model.opt.timestep * realtime_factor, 1.0 / 60.0)
            while not self._stop_play.is_set():
                with self._lock:
                    frame = self._current_frame
                if frame >= self._cache.frame_count() - 1:
                    break
                self._render_frame(frame)
                with self._lock:
                    self._current_frame = frame + 1
                time.sleep(frame_dt)

        self._play_thread = threading.Thread(target=_loop, daemon=True)
        self._play_thread.start()

    def pause(self) -> None:
        self._stop_play.set()
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=1.0)
        self._play_thread = None

    def step_forward(self) -> None:
        with self._lock:
            if self._cache is None:
                return
            self._current_frame = min(self._current_frame + 1, self._cache.frame_count() - 1)
            frame = self._current_frame
        self._render_frame(frame)

    def step_backward(self) -> None:
        with self._lock:
            self._current_frame = max(self._current_frame - 1, 0)
            frame = self._current_frame
        self._render_frame(frame)

    @property
    def current_frame(self) -> int:
        with self._lock:
            return self._current_frame

    def _render_frame(self, frame_idx: int) -> None:
        """Apply frame state, render via mujoco.Renderer, upload to DPG texture."""
        with self._lock:
            if self._model is None or self._data is None or self._cache is None:
                return
            if self._renderer is None:
                return

            series = self._cache.series
            for i in range(len(self._data.qpos)):
                key = f"qpos_{i}"
                if key in series and frame_idx < len(series[key]):
                    self._data.qpos[i] = series[key][frame_idx]
            for i in range(len(self._data.qvel)):
                key = f"qvel_{i}"
                if key in series and frame_idx < len(series[key]):
                    self._data.qvel[i] = series[key][frame_idx]

            mujoco.mj_forward(self._model, self._data)
            self._renderer.update_scene(self._data)
            pixels = self._renderer.render()  # (H, W, 3) uint8

        # Build RGBA float32 array and upload (dpg.set_value is thread-safe)
        rgba = np.ones((RENDER_H, RENDER_W, 4), dtype=np.float32)
        rgba[:, :, :3] = pixels.astype(np.float32) / 255.0
        dpg.set_value(TEX_TAG, rgba.flatten().tolist())
