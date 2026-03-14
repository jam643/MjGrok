"""ViewerController: MuJoCo passive viewer with seek/play/pause/step."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

import mujoco
import mujoco.viewer

from mjgrok.scenarios.base import Scenario
from mjgrok.simulation.trajectory import TrajectoryCache


class ViewerController:
    def __init__(self) -> None:
        self._handle: Any | None = None
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._cache: TrajectoryCache | None = None
        self._current_frame: int = 0
        self._play_thread: threading.Thread | None = None
        self._stop_play = threading.Event()
        self._lock = threading.Lock()

    def load(self, scenario: Scenario, params: dict[str, Any], cache: TrajectoryCache) -> None:
        """Close existing viewer, rebuild model+data, launch passive viewer.

        Must be called from a daemon thread (not the main thread) on macOS.
        mjpython dispatches OpenGL to the main thread internally.
        """
        self.close()

        model = scenario.build_model(params)
        data = mujoco.MjData(model)

        with self._lock:
            self._model = model
            self._data = data
            self._cache = cache
            self._current_frame = 0

        # Seek to first frame before launching so viewer shows initial state
        self._apply_frame(0)

        self._handle = mujoco.viewer.launch_passive(model, data)

    def close(self) -> None:
        self.pause()
        handle = self._handle
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()
            self._handle = None

    def is_open(self) -> bool:
        return self._handle is not None and self._handle.is_running()

    def seek(self, frame_idx: int) -> None:
        """Set viewer state to the given trajectory frame."""
        if self._cache is None or not self.is_open():
            return
        frame_idx = max(0, min(frame_idx, self._cache.frame_count() - 1))
        with self._lock:
            self._current_frame = frame_idx
        self._apply_frame(frame_idx)

    def _apply_frame(self, frame_idx: int) -> None:
        """Write qpos/qvel from cache into data and sync the viewer."""
        if self._model is None or self._data is None or self._cache is None:
            return

        series = self._cache.series
        nq = len(self._data.qpos)
        nv = len(self._data.qvel)

        with self._lock:
            for i in range(nq):
                key = f"qpos_{i}"
                if key in series and frame_idx < len(series[key]):
                    self._data.qpos[i] = series[key][frame_idx]
            for i in range(nv):
                key = f"qvel_{i}"
                if key in series and frame_idx < len(series[key]):
                    self._data.qvel[i] = series[key][frame_idx]

            mujoco.mj_forward(self._model, self._data)

        if self._handle is not None:
            try:
                with self._handle.lock():
                    self._handle.sync()
            except Exception:
                pass

    def play(self, realtime_factor: float = 1.0) -> None:
        """Start playback from current frame in a daemon thread."""
        self.pause()
        self._stop_play.clear()

        def _play_loop() -> None:
            if self._cache is None or self._model is None:
                return
            dt = self._model.opt.timestep * realtime_factor
            while not self._stop_play.is_set():
                with self._lock:
                    frame = self._current_frame
                if frame >= self._cache.frame_count() - 1:
                    break
                self._apply_frame(frame)
                with self._lock:
                    self._current_frame = frame + 1
                time.sleep(dt)

        self._play_thread = threading.Thread(target=_play_loop, daemon=True)
        self._play_thread.start()

    def pause(self) -> None:
        """Stop playback thread."""
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
        self._apply_frame(frame)

    def step_backward(self) -> None:
        with self._lock:
            self._current_frame = max(self._current_frame - 1, 0)
            frame = self._current_frame
        self._apply_frame(frame)

    @property
    def current_frame(self) -> int:
        with self._lock:
            return self._current_frame
