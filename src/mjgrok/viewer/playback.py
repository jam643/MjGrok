"""ViewerLauncher: MuJoCo interactive viewer in a subprocess.

macOS: subprocess runs under mjpython (its own main thread → no NSWindow conflict).
Linux: subprocess runs under the current Python executable (no main-thread restriction).

IPC:
  .npz  — model XML + qpos/qvel trajectory arrays (written once at launch)
  .json — control file: {"frame": N} (written by GUI on every seek)

InProcessViewer: same interface but runs launch_passive() in a background thread.
  Works on Linux (no main-thread restriction). Not suitable for macOS.
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
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from mjgrok.scenarios.base import Scenario
from mjgrok.simulation.trajectory import TrajectoryCache


class ViewerBase(ABC):
    """Shared interface for all viewer implementations."""

    @abstractmethod
    def set_on_frame(self, callback: Callable[[int], None]) -> None: ...

    @abstractmethod
    def load(self, scenario: Scenario, params: dict[str, Any], cache: TrajectoryCache) -> None: ...

    @abstractmethod
    def reload_trajectory(
        self, scenario: Scenario, params: dict[str, Any], cache: TrajectoryCache
    ) -> bool: ...

    @abstractmethod
    def configure(self, n_frames: int, dt: float) -> None: ...

    @abstractmethod
    def seek(self, frame: int) -> None: ...

    @abstractmethod
    def play(self) -> None: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def step_forward(self) -> int: ...

    @abstractmethod
    def step_backward(self) -> int: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def is_running(self) -> bool: ...

    @property
    @abstractmethod
    def current_frame(self) -> int: ...


class ViewerLauncher(ViewerBase):
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
        """Write trajectory to temp files and spawn the viewer subprocess."""
        self.close()

        # Build qpos/qvel matrices from the finalized cache
        nq = sum(1 for k in cache.series_arr if k.startswith("qpos_"))
        nv = sum(1 for k in cache.series_arr if k.startswith("qvel_"))
        qpos = np.column_stack([cache.series_arr[f"qpos_{i}"] for i in range(nq)])
        qvel = np.column_stack([cache.series_arr[f"qvel_{i}"] for i in range(nv)])

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
        _write_ctrl(ctrl_path, frame=0)
        self._ctrl_path = ctrl_path

        self._n_frames = cache.frame_count()
        self._current_frame = 0

        # Spawn subprocess — mjpython on macOS, current interpreter on Linux
        if platform.system() == "Darwin":
            cmd = ["uv", "run", "mjpython", "-m", "mjgrok.viewer.worker"]
        else:
            cmd = [sys.executable, "-m", "mjgrok.viewer.worker"]
        cmd += ["--npz", npz_path, "--ctrl", ctrl_path]

        self._proc = subprocess.Popen(cmd)

    def reload_trajectory(
        self, scenario: Scenario, params: dict[str, Any], cache: TrajectoryCache
    ) -> bool:
        """Subprocess viewer cannot hot-reload; requires full reopen via load()."""
        return False

    def configure(self, n_frames: int, dt: float) -> None:
        self._n_frames = n_frames
        self._dt = dt
        self._current_frame = 0

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
            _write_ctrl(self._ctrl_path, self._current_frame)

    def step_forward(self) -> int:
        self.seek(self._current_frame + 1)
        return self._current_frame

    def step_backward(self) -> int:
        self.seek(self._current_frame - 1)
        return self._current_frame

    @property
    def current_frame(self) -> int:
        return self._current_frame


def _write_ctrl(path: str, frame: int) -> None:
    with open(path, "w") as f:
        json.dump({"frame": frame}, f)


def _copy_model_arrays(dest: mujoco.MjModel, src: mujoco.MjModel) -> None:
    """Copy all writable numpy array fields from src into dest in-place.

    Only copies arrays whose shapes match — this is safe for parameter-only
    changes (friction, mass, geometry size, etc.) where topology is unchanged.
    Scalar fields and read-only attributes are skipped silently.
    """
    for name in dir(src):
        if name.startswith("_"):
            continue
        src_val = getattr(src, name, None)
        if not isinstance(src_val, np.ndarray):
            continue
        try:
            dest_val = getattr(dest, name)
            if isinstance(dest_val, np.ndarray) and dest_val.shape == src_val.shape:
                dest_val[:] = src_val
        except (AttributeError, ValueError, TypeError):
            pass


class InProcessViewer(ViewerBase):
    """MuJoCo passive viewer running in a background thread (same process).

    Works on Linux where launch_passive() has no main-thread restriction.
    Not suitable for macOS (NSWindow requires the main thread).

    Both trajectory data and model parameters are hot-swappable while the
    viewer runs: call reload_trajectory() after a new simulation completes
    and the viewer will update on the next render tick without reopening.
    """

    def __init__(self) -> None:
        self._viewer_thread: threading.Thread | None = None
        self._handle: Any = None
        self._stop_event = threading.Event()
        # Trajectory + model state — protected by _traj_lock for atomic swaps
        self._traj_lock = threading.Lock()
        self._model: mujoco.MjModel | None = None
        self._pending_model: mujoco.MjModel | None = None  # applied inside handle.lock()
        self._qpos_traj: np.ndarray | None = None
        self._qvel_traj: np.ndarray | None = None
        self._n_frames: int = 0
        self._current_frame: int = 0
        self._dt: float = 0.002
        self._play_thread: threading.Thread | None = None
        self._play_stop = threading.Event()
        self._on_frame: Callable[[int], None] | None = None
        self.fps: float = 30.0

    def set_on_frame(self, callback: Callable[[int], None]) -> None:
        self._on_frame = callback

    @staticmethod
    def _arrays_from_cache(cache: TrajectoryCache) -> tuple[np.ndarray, np.ndarray]:
        nq = sum(1 for k in cache.series_arr if k.startswith("qpos_"))
        nv = sum(1 for k in cache.series_arr if k.startswith("qvel_"))
        qpos = np.column_stack([cache.series_arr[f"qpos_{i}"] for i in range(nq)])
        qvel = np.column_stack([cache.series_arr[f"qvel_{i}"] for i in range(nv)])
        return qpos, qvel

    def load(self, scenario: Scenario, params: dict[str, Any], cache: TrajectoryCache) -> None:
        """Build model and launch passive viewer in a background thread."""
        self.close()

        qpos, qvel = self._arrays_from_cache(cache)
        self._model = scenario.build_model(params)
        with self._traj_lock:
            self._pending_model = None
            self._qpos_traj = qpos
            self._qvel_traj = qvel
            self._n_frames = cache.frame_count()
            self._current_frame = 0
            self._dt = cache.times[1] - cache.times[0] if len(cache.times) >= 2 else 0.002

        self._stop_event.clear()

        data = mujoco.MjData(self._model)

        def _viewer_loop() -> None:
            with mujoco.viewer.launch_passive(self._model, data) as handle:
                self._handle = handle
                while handle.is_running() and not self._stop_event.is_set():
                    with handle.lock():
                        with self._traj_lock:
                            # Apply pending model update (in-place copy so the viewer's
                            # C-side pointer remains valid)
                            if self._pending_model is not None:
                                _copy_model_arrays(self._model, self._pending_model)
                                self._pending_model = None
                            f = min(self._current_frame, self._n_frames - 1)
                            qpos_row = self._qpos_traj[f]
                            qvel_row = self._qvel_traj[f]
                        data.qpos[: len(qpos_row)] = qpos_row
                        data.qvel[: len(qvel_row)] = qvel_row
                        mujoco.mj_forward(self._model, data)
                    handle.sync()
                    time.sleep(1.0 / 60.0)
                self._handle = None

        self._viewer_thread = threading.Thread(target=_viewer_loop, daemon=True)
        self._viewer_thread.start()

    def reload_trajectory(
        self, scenario: Scenario, params: dict[str, Any], cache: TrajectoryCache
    ) -> bool:
        """Hot-swap model parameters and trajectory data in the running viewer.

        The viewer window stays open. On the next render tick the model arrays
        are copied in-place (inside handle.lock) and the new trajectory replayed.
        Returns True if the viewer was running.
        """
        if not self.is_running():
            return False
        new_model = scenario.build_model(params)
        qpos, qvel = self._arrays_from_cache(cache)
        with self._traj_lock:
            self._pending_model = new_model
            self._qpos_traj = qpos
            self._qvel_traj = qvel
            self._n_frames = cache.frame_count()
            self._current_frame = min(self._current_frame, self._n_frames - 1)
            self._dt = cache.times[1] - cache.times[0] if len(cache.times) >= 2 else 0.002
        return True

    def configure(self, n_frames: int, dt: float) -> None:
        with self._traj_lock:
            self._n_frames = n_frames
            self._dt = dt
            self._current_frame = 0

    def close(self) -> None:
        self.pause()
        self._stop_event.set()
        handle = self._handle
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()
        self._handle = None
        self._model = None
        with self._traj_lock:
            self._pending_model = None
            self._qpos_traj = None
            self._qvel_traj = None
            self._n_frames = 0
            self._current_frame = 0

    def is_running(self) -> bool:
        return self._viewer_thread is not None and self._viewer_thread.is_alive()

    def seek(self, frame: int) -> None:
        with self._traj_lock:
            self._current_frame = max(0, min(frame, self._n_frames - 1))

    def play(self) -> None:
        if self._play_thread and self._play_thread.is_alive():
            return
        self._play_stop.clear()

        def _loop() -> None:
            interval = 1.0 / self.fps
            while not self._play_stop.is_set():
                with self._traj_lock:
                    frame_step = max(1, round(interval / self._dt))
                    at_end = self._current_frame >= self._n_frames - 1
                    if at_end:
                        self._play_stop.set()
                        break
                    self._current_frame = min(
                        self._current_frame + frame_step, self._n_frames - 1
                    )
                    current = self._current_frame
                if self._on_frame:
                    self._on_frame(current)
                time.sleep(interval)

        self._play_thread = threading.Thread(target=_loop, daemon=True)
        self._play_thread.start()

    def pause(self) -> None:
        self._play_stop.set()

    def step_forward(self) -> int:
        with self._traj_lock:
            self._current_frame = min(self._current_frame + 1, self._n_frames - 1)
            return self._current_frame

    def step_backward(self) -> int:
        with self._traj_lock:
            self._current_frame = max(self._current_frame - 1, 0)
            return self._current_frame

    @property
    def current_frame(self) -> int:
        return self._current_frame
