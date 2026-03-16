"""Plot panel: pre-created DearPyGUI plots, updated via dpg.set_value (thread-safe)."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from mjgrok.scenarios.base import PlotSpec, Scenario
from mjgrok.simulation.trajectory import TrajectoryCache

# Fixed color palette — colors stay stable across re-runs because they're assigned
# by traj_idx, not by DearPyGUI's auto-increment counter.
_PALETTE: list[tuple[int, int, int, int]] = [
    (31, 119, 180, 255),   # blue
    (255, 127, 14, 255),   # orange
    (44, 160, 44, 255),    # green
    (214, 39, 40, 255),    # red
    (148, 103, 189, 255),  # purple
    (140, 86, 75, 255),    # brown
    (227, 119, 194, 255),  # pink
    (127, 127, 127, 255),  # gray
    (188, 189, 34, 255),   # yellow-green
    (23, 190, 207, 255),   # cyan
]


def _get_series_theme(traj_idx: int) -> int:
    """Return (creating if needed) a reusable theme tag for a given trajectory index."""
    theme_tag = f"__plot_series_theme_{traj_idx}"
    if dpg.does_item_exist(theme_tag):
        return dpg.get_item_alias(theme_tag)
    color = _PALETTE[traj_idx % len(_PALETTE)]
    with dpg.theme(tag=theme_tag), dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
    return dpg.get_item_alias(theme_tag)


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

        # Split into ungrouped (shown at top) and grouped sections
        ungrouped = [s for s in self._specs if not s.group]
        groups: dict[str, list[PlotSpec]] = {}
        for spec in self._specs:
            if spec.group:
                groups.setdefault(spec.group, []).append(spec)

        dpg.add_checkbox(
            tag="plot_autoresize",
            label="Autoresize axes",
            default_value=True,
            parent=self._parent,
        )

        for spec in ungrouped:
            self._build_plot(spec, parent=self._parent)

        for group_name, specs in groups.items():
            header = dpg.add_collapsing_header(
                label=group_name,
                default_open=True,
                parent=self._parent,
            )
            inner = dpg.add_group(parent=header, indent=0)
            for spec in specs:
                self._build_plot(spec, parent=inner)

    def _build_plot(self, spec: PlotSpec, parent: str | int) -> None:
        dpg.add_plot(
            label=spec.title,
            tag=f"plot_{spec.plot_id}",
            parent=parent,
            height=180,
            width=-1,
        )
        dpg.add_plot_legend(parent=f"plot_{spec.plot_id}")
        dpg.add_plot_axis(
            dpg.mvXAxis, label=spec.x_label, tag=f"xaxis_{spec.plot_id}",
            parent=f"plot_{spec.plot_id}",
        )
        dpg.add_plot_axis(
            dpg.mvYAxis, label=spec.y_label, tag=f"yaxis_{spec.plot_id}",
            parent=f"plot_{spec.plot_id}",
        )

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
                display_label = label if label else "MuJoCo rollout"
                for key in spec.series_keys:
                    series_tag = f"series_{spec.plot_id}_{traj_idx}_{key}"
                    if multi_traj and multi_key:
                        legend_label = f"{display_label} ({key})"
                    elif multi_traj:
                        legend_label = display_label
                    elif multi_key:
                        legend_label = f"MuJoCo rollout ({key})"
                    else:
                        legend_label = "MuJoCo rollout"
                    dpg.add_line_series(
                        x=[],
                        y=[],
                        label=legend_label,
                        tag=series_tag,
                        parent=y_axis_tag,
                    )
                    dpg.bind_item_theme(series_tag, _get_series_theme(traj_idx))
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
            if dpg.get_value("plot_autoresize"):
                dpg.fit_axis_data(f"xaxis_{spec.plot_id}")
                dpg.fit_axis_data(f"yaxis_{spec.plot_id}")

    def clear(self) -> None:
        """Reset all series to empty."""
        for tag in self._series_tags:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, [[], []])
