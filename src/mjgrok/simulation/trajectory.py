"""TrajectoryCache: stores time-series data from a simulation run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrajectoryCache:
    params: dict[str, Any]
    label: str = ""
    times: list[float] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)

    rollout_ms: float = 0.0  # wall-clock time for the mj_step loop (milliseconds)

    # Finalized arrays (set after finalize())
    _times_arr: np.ndarray | None = field(default=None, repr=False)
    _series_arr: dict[str, np.ndarray] | None = field(default=None, repr=False)

    def append(self, t: float, values: dict[str, float]) -> None:
        self.times.append(t)
        for k, v in values.items():
            if k not in self.series:
                self.series[k] = []
            self.series[k].append(v)

    def finalize(self) -> None:
        """Convert lists to numpy arrays for efficient access."""
        self._times_arr = np.array(self.times, dtype=np.float64)
        self._series_arr = {k: np.array(v, dtype=np.float64) for k, v in self.series.items()}

    @property
    def times_arr(self) -> np.ndarray:
        if self._times_arr is None:
            raise RuntimeError("Call finalize() before accessing times_arr")
        return self._times_arr

    @property
    def series_arr(self) -> dict[str, np.ndarray]:
        if self._series_arr is None:
            raise RuntimeError("Call finalize() before accessing series_arr")
        return self._series_arr

    def frame_count(self) -> int:
        return len(self.times)
