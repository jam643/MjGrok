"""MjGrokApp: DearPyGUI main loop and integration hub."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg

from mjgrok.gui.param_panel import ParamPanel
from mjgrok.gui.playback_panel import PlaybackPanel
from mjgrok.gui.plot_panel import PlotPanel
from mjgrok.gui.saveload_panel import SaveLoadPanel
from mjgrok.scenarios import SCENARIOS
from mjgrok.scenarios.base import Scenario
from mjgrok.simulation.runner import SimulationRunner
from mjgrok.simulation.trajectory import TrajectoryCache
from mjgrok.viewer.playback import ViewerLauncher

_FONTS_DIR = Path(__file__).parent.parent / "assets" / "fonts"


class MjGrokApp:
    def __init__(self) -> None:
        self._scenario: Scenario = SCENARIOS[0]
        self._caches: dict[str, TrajectoryCache] = {}
        self._viewer = ViewerLauncher()
        self._runner = SimulationRunner(
            on_done=self._on_sim_done,
            on_error=self._on_sim_error,
            on_progress=self._on_sim_progress,
        )

        self._param_panel: ParamPanel | None = None
        self._plot_panel: PlotPanel | None = None
        self._playback_panel: PlaybackPanel | None = None
        self._saveload_panel: SaveLoadPanel | None = None

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="MjGrok — MuJoCo Physics Sandbox", width=1440, height=900)

        self._setup_fonts()
        self._build_ui()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

        self._runner.cancel()
        self._viewer.close()

    def _setup_fonts(self) -> None:
        self._font_header: int | None = None
        regular = _FONTS_DIR / "Inter-Regular.ttf"
        semibold = _FONTS_DIR / "Inter-SemiBold.ttf"
        if not regular.exists():
            return
        with dpg.font_registry():
            dpg.bind_font(dpg.add_font(str(regular), 14))
            header_src = semibold if semibold.exists() else regular
            self._font_header = dpg.add_font(str(header_src), 18)

    def _build_ui(self) -> None:
        with dpg.window(tag="main_window", label="MjGrok", no_title_bar=True):
            dpg.set_primary_window("main_window", True)

            with dpg.group(horizontal=True):
                # Left column: scenario picker + params + run controls
                with dpg.child_window(tag="left_panel", width=640, height=-1):
                    dpg.add_text("Scenario", tag="lbl_scenario")
                    dpg.add_combo(
                        tag="scenario_picker",
                        items=[s.name for s in SCENARIOS],
                        default_value=self._scenario.name,
                        callback=self._on_scenario_changed,
                        width=-1,
                    )
                    dpg.add_text("", tag="scenario_desc", wrap=0)

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    with dpg.group(horizontal=True):
                        dpg.add_text("Parameters", tag="lbl_parameters")
                        dpg.add_button(
                            label="Reset to Defaults",
                            tag="reset_btn",
                            callback=self._on_reset_clicked,
                            small=True,
                        )
                    with dpg.child_window(tag="param_container", height=540, border=False):
                        pass

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    with dpg.child_window(tag="saveload_container", height=90, border=False):
                        pass

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    dpg.add_button(
                        label="Run Simulation",
                        tag="run_btn",
                        callback=self._on_run_clicked,
                        width=-1,
                    )
                    dpg.add_checkbox(
                        tag="auto_run",
                        label="Auto-run on change",
                        default_value=False,
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

        self._param_panel = ParamPanel("param_container", on_change=self._on_param_changed)
        self._param_panel.build(self._scenario)

        self._saveload_panel = SaveLoadPanel("saveload_container", on_load=self._on_preset_loaded)
        self._saveload_panel.set_params_getter(lambda: self._param_panel.collect_params())
        self._saveload_panel.build(self._scenario.name)

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
            on_trajectory_changed=self._on_trajectory_changed,
        )
        self._playback_panel.build()

        dpg.set_value("scenario_desc", self._scenario.description)

        if self._font_header is not None:
            for tag in ("lbl_scenario", "lbl_parameters"):
                dpg.bind_item_font(tag, self._font_header)

    # ── Scenario selection ──────────────────────────────────────────────────

    def _on_scenario_changed(self, sender, app_data, user_data) -> None:
        for s in SCENARIOS:
            if s.name == app_data:
                self._scenario = s
                break
        dpg.set_value("scenario_desc", self._scenario.description)
        self._param_panel.build(self._scenario)
        self._plot_panel.build(self._scenario)
        self._saveload_panel.refresh(self._scenario.name)
        self._caches = {}

    # ── Simulation ──────────────────────────────────────────────────────────

    def _on_preset_loaded(self, params: dict) -> None:
        self._param_panel.apply_params(params)

    def _on_reset_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        if self._param_panel:
            self._param_panel.reset_to_defaults()

    def _on_param_changed(self, sender=None, app_data=None, user_data=None) -> None:
        if dpg.get_value("auto_run"):
            self._on_run_clicked()

    def _on_run_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        base_params = self._param_panel.collect_params()
        sweep_configs = self._param_panel.get_sweep_configs()

        labeled_params = self._build_labeled_params(base_params, sweep_configs)
        labels = [lp[0] for lp in labeled_params]

        self._caches = {}
        dpg.set_value("status_text", "Running...")
        dpg.set_value("progress_bar", 0.0)

        # Pre-create series and populate trajectory dropdown from main thread
        self._plot_panel.prepare_trajectories(labels)
        self._playback_panel.set_trajectories(labels)

        if len(labeled_params) == 1:
            self._runner.run(
                self._scenario,
                labeled_params[0][1],
                duration=5.0,
                dt=0.002,
                label=labeled_params[0][0],
            )
        else:
            self._runner.run_batch(self._scenario, labeled_params, duration=5.0, dt=0.002)

    def _build_labeled_params(
        self,
        base_params: dict[str, Any],
        sweep_configs,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return (label, params) pairs for all sweep combinations, or a single default run."""
        if not sweep_configs:
            return [("", base_params)]

        # Cartesian product across all active sweeps
        sweep_options = [
            [(sc.name, v) for v in sc.values] for sc in sweep_configs
        ]
        runs = []
        for combo in itertools.product(*sweep_options):
            params = dict(base_params)
            label_parts = []
            for name, val in combo:
                params[name] = val
                label_parts.append(f"{name}={val:.3g}" if isinstance(val, float) else f"{name}={val}")
            runs.append((", ".join(label_parts), params))
        return runs

    def _on_sim_done(self, cache: TrajectoryCache) -> None:
        """Called from simulation thread — only dpg.set_value is safe here."""
        self._caches[cache.label] = cache
        self._plot_panel.update(cache)

        # Update scrub to the first completed trajectory if none selected yet
        selected = self._playback_panel.get_selected_trajectory()
        if not selected or selected == cache.label:
            self._playback_panel.set_frame_count(cache.frame_count())

        n_done = len(self._caches)
        n_total = len(self._plot_panel._traj_labels)
        if n_done == n_total:
            dpg.set_value("status_text", f"Done — {n_done} trajectory/ies, {cache.frame_count()} frames each")
            dpg.set_value("progress_bar", 1.0)
        else:
            dpg.set_value("status_text", f"Running — {n_done}/{n_total} trajectories done")

    def _on_sim_error(self, exc: Exception) -> None:
        dpg.set_value("status_text", f"Error: {exc}")
        dpg.set_value("progress_bar", 0.0)

    def _on_sim_progress(self, frac: float) -> None:
        dpg.set_value("progress_bar", frac)

    # ── Playback controls ───────────────────────────────────────────────────

    def _on_trajectory_changed(self, label: str) -> None:
        cache = self._caches.get(label)
        if cache and self._playback_panel:
            self._playback_panel.set_frame_count(cache.frame_count())

    def _on_seek(self, frame: int) -> None:
        self._viewer.seek(frame)

    def _on_play(self) -> None:
        self._viewer.seek(self._viewer.current_frame)

    def _on_pause(self) -> None:
        pass  # Play/pause is managed within the interactive viewer window

    def _on_step_forward(self) -> None:
        frame = self._viewer.step_forward()
        if self._playback_panel:
            self._playback_panel.set_current_frame(frame)

    def _on_step_backward(self) -> None:
        frame = self._viewer.step_backward()
        if self._playback_panel:
            self._playback_panel.set_current_frame(frame)

    def _on_open_viewer(self) -> None:
        label = self._playback_panel.get_selected_trajectory()
        cache = self._caches.get(label)
        if cache is None:
            # Fall back to any available cache
            cache = next(iter(self._caches.values()), None)
        if cache is None:
            dpg.set_value("status_text", "Run a simulation first")
            return
        try:
            self._viewer.load(self._scenario, cache.params, cache)
            dpg.set_value("status_text", "Viewer opened")
        except Exception as e:
            dpg.set_value("status_text", f"Viewer error: {e}")
