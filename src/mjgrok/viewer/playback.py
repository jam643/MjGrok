"""ViewerLauncher: MuJoCo interactive viewer in a subprocess.

macOS: subprocess runs under mjpython (its own main thread → no NSWindow conflict).
Linux: subprocess runs under the current Python executable (no main-thread restriction).

IPC:
  .npz  — model XML + qpos/qvel trajectory arrays (written once at launch)
  .json — control file: {"frame": N} (written by GUI on every seek)
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from mjgrok.scenarios.base import Scenario
from mjgrok.simulation.trajectory import TrajectoryCache


class ViewerLauncher:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._ctrl_path: str | None = None
        self._npz_path: str | None = None
        self._n_frames: int = 0
        self._current_frame: int = 0
        self._dt: float = 0.002
        self._play_thread: threading.Thread | None = None
        self._play_stop = threading.Event()
        self._on_frame: Callable[[int], None] | None = None
        self.fps: float = 30.0
        self._version: int = 0

    def set_on_frame(self, callback: Callable[[int], None]) -> None:
        """Register a callback invoked (from play thread) on each frame advance."""
        self._on_frame = callback

    def play(self) -> None:
        """Start auto-advancing frames at self.fps. No-op if already playing."""
        if self._play_thread and self._play_thread.is_alive():
            return
        self._play_stop.clear()

        def _loop() -> None:
            interval = 1.0 / self.fps
            frame_step = max(1, round(interval / self._dt))
            while not self._play_stop.is_set():
                if self._current_frame >= self._n_frames - 1:
                    self._play_stop.set()
                    break
                self.seek(self._current_frame + frame_step)
                if self._on_frame:
                    self._on_frame(self._current_frame)
                time.sleep(interval)

        self._play_thread = threading.Thread(target=_loop, daemon=True)
        self._play_thread.start()

    def pause(self) -> None:
        self._play_stop.set()

    def load(self, scenario: Scenario, params: dict[str, Any], cache: TrajectoryCache) -> None:
        """Write trajectory to temp files and (re)load the viewer subprocess."""
        # Build qpos/qvel matrices from the finalized cache
        nq = sum(1 for k in cache.series_arr if k.startswith("qpos_"))
        nv = sum(1 for k in cache.series_arr if k.startswith("qvel_"))
        qpos = np.column_stack([cache.series_arr[f"qpos_{i}"] for i in range(nq)])
        qvel = np.column_stack([cache.series_arr[f"qvel_{i}"] for i in range(nv)])

        self._n_frames = cache.frame_count()
        self._current_frame = 0

        if self.is_running() and self._npz_path and self._ctrl_path:
            # Hot-reload: overwrite the npz atomically, then bump version in ctrl
            fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=os.path.dirname(self._npz_path))
            os.close(fd)
            np.savez(
                tmp_path,
                scenario_name=np.array(scenario.name),
                params_json=np.array(json.dumps(params)),
                qpos=qpos,
                qvel=qvel,
            )
            os.replace(tmp_path, self._npz_path)
            self._version += 1
            _write_ctrl(self._ctrl_path, frame=0, version=self._version)
            return

        self.close()

        # Write trajectory file — pass scenario name + params so the worker can
        # reconstruct the model directly via build_model(), no XML serialization needed
        fd, npz_path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        np.savez(
            npz_path,
            scenario_name=np.array(scenario.name),
            params_json=np.array(json.dumps(params)),
            qpos=qpos,
            qvel=qvel,
        )
        self._npz_path = npz_path

        # Write initial control file
        ctrl_path = npz_path + "_ctrl.json"
        self._version = 0
        _write_ctrl(ctrl_path, frame=0, version=0)
        self._ctrl_path = ctrl_path

        # Spawn subprocess — mjpython on macOS, current interpreter on Linux
        if platform.system() == "Darwin":
            cmd = ["uv", "run", "mjpython", "-m", "mjgrok.viewer.worker"]
        else:
            cmd = [sys.executable, "-m", "mjgrok.viewer.worker"]
        cmd += ["--npz", npz_path, "--ctrl", ctrl_path]

        self._proc = subprocess.Popen(cmd)

    def close(self) -> None:
        self.pause()
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        for path in [self._npz_path, self._ctrl_path]:
            if path:
                with contextlib.suppress(OSError):
                    os.unlink(path)
        self._npz_path = None
        self._ctrl_path = None
        self._n_frames = 0
        self._current_frame = 0

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def seek(self, frame: int) -> None:
        self._current_frame = max(0, min(frame, self._n_frames - 1))
        if self._ctrl_path:
            _write_ctrl(self._ctrl_path, self._current_frame, version=self._version)

    def step_forward(self) -> int:
        self.seek(self._current_frame + 1)
        return self._current_frame

    def step_backward(self) -> int:
        self.seek(self._current_frame - 1)
        return self._current_frame

    @property
    def current_frame(self) -> int:
        return self._current_frame


def _write_ctrl(path: str, frame: int, version: int = 0) -> None:
    with open(path, "w") as f:
        json.dump({"frame": frame, "version": version}, f)
