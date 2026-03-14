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
    ) -> None:
        self._parent = parent_tag
        self._on_seek = on_seek
        self._on_play = on_play
        self._on_pause = on_pause
        self._on_step_forward = on_step_forward
        self._on_step_backward = on_step_backward
        self._on_open_viewer = on_open_viewer

    def build(self) -> None:
        dpg.add_text("Playback", parent=self._parent)
        dpg.add_separator(parent=self._parent)

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

    def set_frame_count(self, n: int) -> None:
        """Update scrub slider max when a new trajectory is loaded."""
        if n > 0:
            dpg.configure_item("playback_scrub", max_value=n - 1)
            dpg.set_value("playback_scrub", 0)

    def set_current_frame(self, frame: int) -> None:
        dpg.set_value("playback_scrub", frame)
