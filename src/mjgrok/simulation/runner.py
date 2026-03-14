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
        on_progress: Callable[[float], None],  # fraction 0.0–1.0
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
    ) -> None:
        """Cancel any in-progress run, then start a new daemon thread."""
        self.cancel()
        self._cancel_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(scenario, params, duration, dt),
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        """Signal cancellation and wait up to 2s for thread to exit."""
        self._cancel_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run_loop(
        self,
        scenario: Scenario,
        params: dict[str, Any],
        duration: float,
        dt: float,
    ) -> None:
        try:
            model = scenario.build_model(params)
            data = mujoco.MjData(model)

            box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
            force_x = float(params.get("force_x", 0.0))

            cache = TrajectoryCache(params=dict(params))
            total_steps = int(duration / dt)

            for step in range(total_steps):
                if self._cancel_event.is_set():
                    return

                # Inject external force before each step
                if box_id >= 0:
                    data.xfrc_applied[box_id][0] = force_x

                mujoco.mj_step(model, data)

                values = scenario.extract_series(model, data, data.time)
                # Store qpos/qvel for viewer playback
                qpos_values = {f"qpos_{i}": float(data.qpos[i]) for i in range(len(data.qpos))}
                qvel_values = {f"qvel_{i}": float(data.qvel[i]) for i in range(len(data.qvel))}
                all_values = {**values, **qpos_values, **qvel_values}

                cache.append(data.time, all_values)

                if step % 50 == 0:
                    self.on_progress((step + 1) / total_steps)

            cache.finalize()
            self.on_done(cache)

        except Exception as e:
            self.on_error(e)
