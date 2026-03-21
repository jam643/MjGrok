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
from mjgrok.viewer.playback import InProcessViewer, ViewerBase, ViewerLauncher

_FONTS_DIR = Path(__file__).parent.parent / "assets" / "fonts"


class MjGrokApp:
    def __init__(self) -> None:
        self._scenario: Scenario = SCENARIOS[0]
        self._caches: dict[str, TrajectoryCache] = {}
        self._analytical_labels: set[str] = set()
        self._n_sim_expected: int = 0
        self._viewer: ViewerBase = ViewerLauncher()
        self._viewer.set_on_frame(self._on_viewer_frame)
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
        dpg.maximize_viewport()
        dpg.show_viewport()
        self._on_run_clicked()
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
            dpg.bind_font(dpg.add_font(str(regular), 18))
            header_src = semibold if semibold.exists() else regular
            self._font_header = dpg.add_font(str(header_src), 24)

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
                    with dpg.child_window(tag="param_container", height=700, border=False):
                        pass

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    with dpg.child_window(tag="saveload_container", height=110, border=False):
                        pass

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    dpg.add_separator()
                    dpg.add_spacer(height=4)
                    dpg.add_text("Analytical Solution", tag="lbl_analytical")
                    dpg.add_checkbox(
                        tag="show_analytical",
                        label="Show analytical solution (Coulomb friction)",
                        default_value=True,
                        callback=self._on_param_changed,
                    )
                    dpg.add_spacer(height=4)
                    dpg.add_separator()
                    dpg.add_spacer(height=4)

                    dpg.add_button(
                        label="Run Simulation",
                        tag="run_btn",
                        callback=self._on_run_clicked,
                        width=-1,
                    )
                    dpg.add_checkbox(
                        tag="auto_run",
                        label="Auto-run on change",
                        default_value=True,
                    )
                    dpg.add_progress_bar(tag="progress_bar", default_value=0.0, width=-1)
                    dpg.add_text("Ready", tag="status_text")

                # Right column: plots (scrollable) + playback (fixed footer)
                with dpg.child_window(tag="right_panel", border=False):
                    with dpg.child_window(tag="plot_container", height=-170, border=False):
                        pass
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
            for tag in ("lbl_scenario", "lbl_parameters", "lbl_analytical"):
                dpg.bind_item_font(tag, self._font_header)

        # Hide analytical section if scenario doesn't support it
        self._refresh_analytical_visibility()

    def _refresh_analytical_visibility(self) -> None:
        """Show/hide the analytical solution checkbox based on scenario support."""
        has_analytical = (
            self._scenario.analytical_solution(self._scenario.default_params()) is not None
        )
        if has_analytical:
            dpg.show_item("lbl_analytical")
            dpg.show_item("show_analytical")
        else:
            dpg.hide_item("lbl_analytical")
            dpg.hide_item("show_analytical")

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
        self._refresh_analytical_visibility()

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
        sim_labels = [lp[0] for lp in labeled_params]

        self._caches = {}
        self._analytical_labels = set()
        self._n_sim_expected = len(labeled_params)

        # Compute analytical trajectories (fast, done synchronously on main thread)
        show_analytical = dpg.get_value("show_analytical")
        analytical_caches: list[TrajectoryCache] = []
        if show_analytical:
            for sim_label, params in labeled_params:
                dur = self._scenario.sim_duration
                a_cache = self._scenario.analytical_solution(params, duration=dur, dt=0.002)
                if a_cache is not None:
                    a_label = f"Analytical ({sim_label})" if sim_label else "Analytical"
                    a_cache.label = a_label
                    analytical_caches.append(a_cache)
                    self._analytical_labels.add(a_label)

        all_labels = sim_labels + [c.label for c in analytical_caches]

        dpg.set_value("status_text", "Running...")
        dpg.set_value("progress_bar", 0.0)

        # Pre-create series and populate trajectory dropdown from main thread
        self._plot_panel.prepare_trajectories(all_labels)
        self._playback_panel.set_trajectories(all_labels)

        # Populate analytical trajectories immediately (series already created above)
        for a_cache in analytical_caches:
            self._caches[a_cache.label] = a_cache
            self._plot_panel.update(a_cache)

        if len(labeled_params) == 1:
            self._runner.run(
                self._scenario,
                labeled_params[0][1],
                duration=self._scenario.sim_duration,
                dt=0.002,
                label=labeled_params[0][0],
            )
        else:
            self._runner.run_batch(
                self._scenario, labeled_params, duration=self._scenario.sim_duration, dt=0.002
            )

    def _build_labeled_params(
        self,
        base_params: dict[str, Any],
        sweep_configs,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return (label, params) pairs for all sweep combinations, or a single default run."""
        if not sweep_configs:
            return [("", base_params)]

        # Cartesian product across all active sweeps
        sweep_options = [[(sc.name, v) for v in sc.values] for sc in sweep_configs]
        runs = []
        for combo in itertools.product(*sweep_options):
            params = dict(base_params)
            label_parts = []
            for name, val in combo:
                params[name] = val
                fmt = f"{name}={val:.3g}" if isinstance(val, float) else f"{name}={val}"
                label_parts.append(fmt)
            runs.append((", ".join(label_parts), params))
        return runs

    def _on_sim_done(self, cache: TrajectoryCache) -> None:
        """Called from simulation thread — only dpg.set_value is safe here."""
        self._caches[cache.label] = cache
        self._plot_panel.update(cache)

        # Update scrub to the first completed trajectory if none selected yet
        selected = self._playback_panel.get_selected_trajectory()
        if not selected or selected == cache.label:
            n = cache.frame_count()
            # Hot-reload InProcessViewer (updates n_frames/dt internally under lock)
            reloaded = self._viewer.reload_trajectory(self._scenario, cache.params, cache)
            if reloaded:
                self._playback_panel.update_frame_count(n)
            else:
                dt = cache.times[1] - cache.times[0] if len(cache.times) >= 2 else 0.002
                self._viewer.configure(n, dt)
                self._playback_panel.set_frame_count(n)

        # Count only simulation completions (analytical caches are pre-populated)
        n_sim_done = sum(1 for lbl in self._caches if lbl not in self._analytical_labels)
        if n_sim_done == self._n_sim_expected:
            total_ms = sum(
                c.rollout_ms
                for lbl, c in self._caches.items()
                if lbl not in self._analytical_labels
            )
            dpg.set_value(
                "status_text",
                f"Done - {self._n_sim_expected} trajectory, {cache.frame_count()} frames each"
                f" | rollout {total_ms:.1f} ms",
            )
            dpg.set_value("progress_bar", 1.0)
        else:
            dpg.set_value(
                "status_text",
                f"Running — {n_sim_done}/{self._n_sim_expected} sim trajectories done",
            )

    def _on_sim_error(self, exc: Exception) -> None:
        dpg.set_value("status_text", f"Error: {exc}")
        dpg.set_value("progress_bar", 0.0)

    def _on_sim_progress(self, frac: float) -> None:
        dpg.set_value("progress_bar", frac)

    # ── Playback controls ───────────────────────────────────────────────────

    def _on_trajectory_changed(self, label: str) -> None:
        cache = self._caches.get(label)
        if cache and self._playback_panel:
            n = cache.frame_count()
            dt = cache.times[1] - cache.times[0] if len(cache.times) >= 2 else 0.002
            reloaded = self._viewer.reload_trajectory(self._scenario, cache.params, cache)
            if not reloaded:
                self._viewer.configure(n, dt)
            self._playback_panel.set_frame_count(n)

    def _on_seek(self, frame: int) -> None:
        self._viewer.seek(frame)

    def _on_play(self) -> None:
        self._viewer.play()

    def _on_pause(self) -> None:
        self._viewer.pause()

    def _on_viewer_frame(self, frame: int) -> None:
        """Called from play thread — only dpg.set_value is safe here."""
        if self._playback_panel:
            self._playback_panel.set_current_frame(frame)

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

        use_inprocess = self._playback_panel.get_use_inprocess_viewer()
        new_viewer: ViewerBase = InProcessViewer() if use_inprocess else ViewerLauncher()

        # Close old viewer and swap
        self._viewer.close()
        self._viewer = new_viewer
        self._viewer.set_on_frame(self._on_viewer_frame)
        # Re-sync frame state via public API
        dt = cache.times[1] - cache.times[0] if len(cache.times) >= 2 else 0.002
        self._viewer.configure(cache.frame_count(), dt)

        try:
            self._viewer.load(self._scenario, cache.params, cache)
            mode = "in-process" if use_inprocess else "subprocess"
            dpg.set_value("status_text", f"Viewer opened ({mode})")
        except Exception as e:
            dpg.set_value("status_text", f"Viewer error: {e}")
