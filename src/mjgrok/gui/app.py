"""MjGrokApp: DearPyGUI main loop and integration hub."""

from __future__ import annotations

import threading

import dearpygui.dearpygui as dpg

from mjgrok.gui.param_panel import ParamPanel
from mjgrok.gui.playback_panel import PlaybackPanel
from mjgrok.gui.plot_panel import PlotPanel
from mjgrok.scenarios import SCENARIOS
from mjgrok.scenarios.base import Scenario
from mjgrok.simulation.runner import SimulationRunner
from mjgrok.simulation.trajectory import TrajectoryCache
from mjgrok.viewer.playback import RENDER_H, RENDER_W, ViewerController, create_viewer_texture


class MjGrokApp:
    def __init__(self) -> None:
        self._scenario: Scenario = SCENARIOS[0]
        self._cache: TrajectoryCache | None = None
        self._viewer_ctrl = ViewerController()
        self._runner = SimulationRunner(
            on_done=self._on_sim_done,
            on_error=self._on_sim_error,
            on_progress=self._on_sim_progress,
        )

        self._param_panel: ParamPanel | None = None
        self._plot_panel: PlotPanel | None = None
        self._playback_panel: PlaybackPanel | None = None

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="MjGrok — MuJoCo Physics Sandbox", width=1440, height=900)

        self._build_ui()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

        self._runner.cancel()
        self._viewer_ctrl.close()

    def _build_ui(self) -> None:
        # Register viewer texture before any windows (must happen before show_viewport)
        create_viewer_texture()

        # Floating viewer window (hidden until user loads it)
        with dpg.window(
            tag="viewer_window",
            label="MuJoCo Viewer",
            show=False,
            width=RENDER_W + 20,
            height=RENDER_H + 40,
            pos=[330, 30],
            no_scrollbar=True,
        ):
            dpg.add_image("viewer_texture", width=RENDER_W, height=RENDER_H)

        with dpg.window(tag="main_window", label="MjGrok", no_title_bar=True):
            dpg.set_primary_window("main_window", True)

            with dpg.group(horizontal=True):
                # Left column: scenario picker + params + run controls
                with dpg.child_window(tag="left_panel", width=640, height=-1):
                    dpg.add_text("Scenario")
                    dpg.add_combo(
                        tag="scenario_picker",
                        items=[s.name for s in SCENARIOS],
                        default_value=self._scenario.name,
                        callback=self._on_scenario_changed,
                        width=-1,
                    )
                    dpg.add_text("", tag="scenario_desc", wrap=0)
                    dpg.add_separator()

                    dpg.add_text("Parameters")
                    with dpg.child_window(tag="param_container", height=560, border=False):
                        pass

                    dpg.add_separator()
                    dpg.add_button(
                        label="Run Simulation",
                        tag="run_btn",
                        callback=self._on_run_clicked,
                        width=-1,
                    )
                    dpg.add_progress_bar(tag="progress_bar", default_value=0.0, width=-1)
                    dpg.add_text("Ready", tag="status_text")

                # Right column: plots + playback
                with dpg.child_window(tag="right_panel", border=False):
                    with dpg.child_window(tag="plot_container", height=780, border=False):
                        pass
                    dpg.add_separator()
                    with dpg.child_window(tag="playback_container", height=-1, border=False):
                        pass

        self._param_panel = ParamPanel("param_container")
        self._param_panel.build(self._scenario)

        self._plot_panel = PlotPanel("plot_container")
        self._plot_panel.build(self._scenario)

        self._playback_panel = PlaybackPanel(
            "playback_container",
            on_seek=self._on_seek,
            on_play=self._on_play,
            on_pause=self._on_pause,
            on_step_forward=self._on_step_forward,
            on_step_backward=self._on_step_backward,
            on_open_viewer=self._on_open_viewer,
        )
        self._playback_panel.build()

        dpg.set_value("scenario_desc", self._scenario.description)

    # ── Scenario selection ──────────────────────────────────────────────────

    def _on_scenario_changed(self, sender, app_data, user_data) -> None:
        for s in SCENARIOS:
            if s.name == app_data:
                self._scenario = s
                break
        dpg.set_value("scenario_desc", self._scenario.description)
        self._param_panel.build(self._scenario)
        self._plot_panel.build(self._scenario)
        self._cache = None

    # ── Simulation ──────────────────────────────────────────────────────────

    def _on_run_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        params = self._param_panel.collect_params()
        dpg.set_value("status_text", "Running...")
        dpg.set_value("progress_bar", 0.0)
        self._plot_panel.clear()
        self._runner.run(self._scenario, params, duration=5.0, dt=0.002)

    def _on_sim_done(self, cache: TrajectoryCache) -> None:
        """Called from simulation thread — only dpg.set_value is safe here."""
        self._cache = cache
        self._plot_panel.update(cache)
        dpg.set_value("status_text", f"Done — {cache.frame_count()} frames")
        dpg.set_value("progress_bar", 1.0)
        if self._playback_panel:
            self._playback_panel.set_frame_count(cache.frame_count())

    def _on_sim_error(self, exc: Exception) -> None:
        dpg.set_value("status_text", f"Error: {exc}")
        dpg.set_value("progress_bar", 0.0)

    def _on_sim_progress(self, frac: float) -> None:
        dpg.set_value("progress_bar", frac)

    # ── Playback controls ───────────────────────────────────────────────────

    def _on_seek(self, frame: int) -> None:
        self._viewer_ctrl.seek(frame)

    def _on_play(self) -> None:
        self._viewer_ctrl.play(realtime_factor=1.0)

    def _on_pause(self) -> None:
        self._viewer_ctrl.pause()

    def _on_step_forward(self) -> None:
        self._viewer_ctrl.step_forward()
        if self._playback_panel:
            self._playback_panel.set_current_frame(self._viewer_ctrl.current_frame)

    def _on_step_backward(self) -> None:
        self._viewer_ctrl.step_backward()
        if self._playback_panel:
            self._playback_panel.set_current_frame(self._viewer_ctrl.current_frame)

    def _on_open_viewer(self) -> None:
        if self._cache is None:
            dpg.set_value("status_text", "Run a simulation first")
            return

        # Show the floating viewer window (main thread — safe)
        dpg.show_item("viewer_window")
        dpg.set_value("status_text", "Loading viewer...")

        cache = self._cache
        scenario = self._scenario
        params = self._param_panel.collect_params()

        def _load() -> None:
            try:
                self._viewer_ctrl.load(scenario, params, cache)
                dpg.set_value("status_text", "Viewer ready")
            except Exception as e:
                dpg.set_value("status_text", f"Viewer error: {e}")

        threading.Thread(target=_load, daemon=True).start()
