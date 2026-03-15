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

    def close(self) -> None:
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
