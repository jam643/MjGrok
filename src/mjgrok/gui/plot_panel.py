"""Plot panel: pre-created DearPyGUI plots, updated via dpg.set_value (thread-safe)."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from mjgrok.scenarios.base import PlotSpec, Scenario
from mjgrok.simulation.trajectory import TrajectoryCache


class PlotPanel:
    def __init__(self, parent_tag: str | int) -> None:
        self._parent = parent_tag
        self._specs: list[PlotSpec] = []
        self._series_tags: list[str] = []
        self._traj_labels: list[str] = []

    def build(self, scenario: Scenario) -> None:
        """Create plot containers with axes only — series are added by prepare_trajectories()."""
        self._specs = scenario.plot_specs()
        self._series_tags = []
        self._traj_labels = []
        dpg.delete_item(self._parent, children_only=True)

        for spec in self._specs:
            with dpg.plot(
                label=spec.title,
                tag=f"plot_{spec.plot_id}",
                parent=self._parent,
                height=180,
                width=-1,
            ):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label=spec.x_label, tag=f"xaxis_{spec.plot_id}")
                dpg.add_plot_axis(dpg.mvYAxis, label=spec.y_label, tag=f"yaxis_{spec.plot_id}")

    def prepare_trajectories(self, labels: list[str]) -> None:
        """Pre-create line series for each trajectory label. Must be called from main thread."""
        for tag in self._series_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self._series_tags = []
        self._traj_labels = list(labels)

        multi_traj = len(labels) > 1

        for spec in self._specs:
            y_axis_tag = f"yaxis_{spec.plot_id}"
            multi_key = len(spec.series_keys) > 1

            for traj_idx, label in enumerate(labels):
                for key in spec.series_keys:
                    series_tag = f"series_{spec.plot_id}_{traj_idx}_{key}"
                    if multi_traj and multi_key:
                        legend_label = f"{label} ({key})"
                    elif multi_traj:
                        legend_label = label
                    else:
                        legend_label = key
                    dpg.add_line_series(
                        x=[],
                        y=[],
                        label=legend_label,
                        tag=series_tag,
                        parent=y_axis_tag,
                    )
                    self._series_tags.append(series_tag)

    def update(self, cache: TrajectoryCache) -> None:
        """Update series for one trajectory. Safe to call from background threads."""
        if cache.label not in self._traj_labels:
            return
        traj_idx = self._traj_labels.index(cache.label)
        times = list(cache.times_arr)

        for spec in self._specs:
            for key in spec.series_keys:
                series_tag = f"series_{spec.plot_id}_{traj_idx}_{key}"
                if key in cache.series_arr and dpg.does_item_exist(series_tag):
                    dpg.set_value(series_tag, [times, list(cache.series_arr[key])])
            dpg.fit_axis_data(f"xaxis_{spec.plot_id}")
            dpg.fit_axis_data(f"yaxis_{spec.plot_id}")

    def clear(self) -> None:
        """Reset all series to empty."""
        for tag in self._series_tags:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, [[], []])
