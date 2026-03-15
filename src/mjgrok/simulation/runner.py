"""SimulationRunner: runs MuJoCo simulations on a background daemon thread."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import mujoco

from mjgrok.scenarios.base import Scenario
from mjgrok.simulation.trajectory import TrajectoryCache


class SimulationRunner:
    def __init__(
        self,
        on_done: Callable[[TrajectoryCache], None],
        on_error: Callable[[Exception], None],
        on_progress: Callable[[float], None],  # fraction 0.0–1.0 across whole batch
    ) -> None:
        self.on_done = on_done
        self.on_error = on_error
        self.on_progress = on_progress

        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

    def run(
        self,
        scenario: Scenario,
        params: dict[str, Any],
        duration: float = 5.0,
        dt: float = 0.002,
        label: str = "",
    ) -> None:
        """Run a single simulation."""
        self._start(scenario, [(label, params)], duration, dt)

    def run_batch(
        self,
        scenario: Scenario,
        labeled_params: list[tuple[str, dict[str, Any]]],
        duration: float = 5.0,
        dt: float = 0.002,
    ) -> None:
        """Run multiple simulations sequentially, calling on_done for each."""
        self._start(scenario, labeled_params, duration, dt)

    def cancel(self) -> None:
        """Signal cancellation and wait up to 2s for thread to exit."""
        self._cancel_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _start(
        self,
        scenario: Scenario,
        labeled_params: list[tuple[str, dict[str, Any]]],
        duration: float,
        dt: float,
    ) -> None:
        self.cancel()
        self._cancel_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(scenario, labeled_params, duration, dt),
            daemon=True,
        )
        self._thread.start()

    def _run_loop(
        self,
        scenario: Scenario,
        labeled_params: list[tuple[str, dict[str, Any]]],
        duration: float,
        dt: float,
    ) -> None:
        total_runs = len(labeled_params)
        total_steps = int(duration / dt)

        for run_idx, (label, params) in enumerate(labeled_params):
            if self._cancel_event.is_set():
                return
            try:
                model = scenario.build_model(params)
                data = mujoco.MjData(model)
                cache = TrajectoryCache(params=dict(params), label=label)

                for step in range(total_steps):
                    if self._cancel_event.is_set():
                        return

                    scenario.apply_ctrl(model, data, params)
                    mujoco.mj_step(model, data)

                    values = scenario.extract_series(model, data, data.time)
                    qpos_values = {f"qpos_{i}": float(data.qpos[i]) for i in range(len(data.qpos))}
                    qvel_values = {f"qvel_{i}": float(data.qvel[i]) for i in range(len(data.qvel))}
                    cache.append(data.time, {**values, **qpos_values, **qvel_values})

                    if step % 50 == 0:
                        overall = (run_idx + (step + 1) / total_steps) / total_runs
                        self.on_progress(overall)

                cache.finalize()
                self.on_done(cache)

            except Exception as e:
                self.on_error(e)
                return
