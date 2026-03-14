"""Plot panel: pre-created DearPyGUI plots, updated via dpg.set_value (thread-safe)."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from mjgrok.scenarios.base import PlotSpec, Scenario
from mjgrok.simulation.trajectory import TrajectoryCache


class PlotPanel:
    def __init__(self, parent_tag: str | int) -> None:
        self._parent = parent_tag
        self._specs: list[PlotSpec] = []

    def build(self, scenario: Scenario) -> None:
        """Pre-create one dpg plot per PlotSpec with empty line series."""
        self._specs = scenario.plot_specs()
        dpg.delete_item(self._parent, children_only=True)

        for spec in self._specs:
            plot_tag = f"plot_{spec.plot_id}"
            x_axis_tag = f"xaxis_{spec.plot_id}"
            y_axis_tag = f"yaxis_{spec.plot_id}"

            with dpg.plot(
                label=spec.title,
                tag=plot_tag,
                parent=self._parent,
                height=180,
                width=-1,
            ):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label=spec.x_label, tag=x_axis_tag)
                with dpg.plot_axis(dpg.mvYAxis, label=spec.y_label, tag=y_axis_tag):
                    for key in spec.series_keys:
                        series_tag = f"series_{spec.plot_id}_{key}"
                        dpg.add_line_series(
                            x=[],
                            y=[],
                            label=key,
                            tag=series_tag,
                        )

    def update(self, cache: TrajectoryCache) -> None:
        """Update all series from the trajectory cache.

        Safe to call from background threads (only uses dpg.set_value).
        """
        times = list(cache.times_arr)
        for spec in self._specs:
            for key in spec.series_keys:
                series_tag = f"series_{spec.plot_id}_{key}"
                if key in cache.series_arr:
                    values = list(cache.series_arr[key])
                    dpg.set_value(series_tag, [times, values])
            # Fit both axes after updating series
            dpg.fit_axis_data(f"xaxis_{spec.plot_id}")
            dpg.fit_axis_data(f"yaxis_{spec.plot_id}")

    def clear(self) -> None:
        """Reset all series to empty."""
        for spec in self._specs:
            for key in spec.series_keys:
                series_tag = f"series_{spec.plot_id}_{key}"
                dpg.set_value(series_tag, [[], []])
