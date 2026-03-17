"""Playback panel: scrub slider, play/pause/step, open viewer button."""

from __future__ import annotations

from collections.abc import Callable

import dearpygui.dearpygui as dpg


class PlaybackPanel:
    def __init__(
        self,
        parent_tag: str | int,
        on_seek: Callable[[int], None],
        on_play: Callable[[], None],
        on_pause: Callable[[], None],
        on_step_forward: Callable[[], None],
        on_step_backward: Callable[[], None],
        on_open_viewer: Callable[[], None],
        on_trajectory_changed: Callable[[str], None] | None = None,
    ) -> None:
        self._parent = parent_tag
        self._on_seek = on_seek
        self._on_play = on_play
        self._on_pause = on_pause
        self._on_step_forward = on_step_forward
        self._on_step_backward = on_step_backward
        self._on_open_viewer = on_open_viewer
        self._on_trajectory_changed = on_trajectory_changed

    def build(self) -> None:
        dpg.add_text("Playback", parent=self._parent)
        dpg.add_separator(parent=self._parent)

        dpg.add_combo(
            tag="traj_combo",
            label="Trajectory",
            items=[],
            default_value="",
            parent=self._parent,
            width=-1,
            show=False,
            callback=lambda s, a, u: self._on_trajectory_changed(a) if self._on_trajectory_changed else None,
        )

        dpg.add_slider_int(
            tag="playback_scrub",
            label="Frame",
            default_value=0,
            min_value=0,
            max_value=0,
            parent=self._parent,
            width=-1,
            callback=lambda s, a, u: self._on_seek(a),
        )

        with dpg.group(horizontal=True, parent=self._parent):
            dpg.add_button(
                label="|<",
                callback=lambda: self._on_step_backward(),
                width=40,
            )
            dpg.add_button(
                label="Play",
                callback=lambda: self._on_play(),
                width=60,
            )
            dpg.add_button(
                label="Pause",
                callback=lambda: self._on_pause(),
                width=60,
            )
            dpg.add_button(
                label=">|",
                callback=lambda: self._on_step_forward(),
                width=40,
            )
            dpg.add_button(
                label="Open Viewer",
                callback=lambda: self._on_open_viewer(),
                width=100,
            )

        dpg.add_checkbox(
            tag="inprocess_viewer",
            label="In-process viewer (Linux; uncheck for macOS subprocess mode)",
            default_value=True,
            parent=self._parent,
        )

    def set_trajectories(self, labels: list[str]) -> None:
        """Populate the trajectory dropdown. Must be called from main thread."""
        if len(labels) > 1:
            dpg.configure_item("traj_combo", items=labels, default_value=labels[0])
            dpg.set_value("traj_combo", labels[0])
            dpg.show_item("traj_combo")
        else:
            dpg.hide_item("traj_combo")

    def get_selected_trajectory(self) -> str:
        """Return the currently selected trajectory label."""
        return dpg.get_value("traj_combo")

    def set_frame_count(self, n: int) -> None:
        """Update scrub slider max when a trajectory is selected."""
        if n > 0:
            dpg.configure_item("playback_scrub", max_value=n - 1)
            dpg.set_value("playback_scrub", 0)

    def update_frame_count(self, n: int) -> None:
        """Update scrub slider max without resetting position (used during hot-reload)."""
        if n > 0:
            current = dpg.get_value("playback_scrub")
            dpg.configure_item("playback_scrub", max_value=n - 1)
            dpg.set_value("playback_scrub", min(current, n - 1))

    def set_current_frame(self, frame: int) -> None:
        dpg.set_value("playback_scrub", frame)

    def get_use_inprocess_viewer(self) -> bool:
        return dpg.get_value("inprocess_viewer")
