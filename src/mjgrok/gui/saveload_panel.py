"""Save/Load panel: persist and restore named parameter sets per scenario."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg

# Stored in the repo root so presets can be committed and shared
_PRESETS_DIR = Path(__file__).parents[3] / "presets"
_PLACEHOLDER = "(select to load)"


def _scenario_dir(scenario_name: str) -> Path:
    return _PRESETS_DIR / scenario_name


def _list_presets(scenario_name: str) -> list[str]:
    d = _scenario_dir(scenario_name)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


class SaveLoadPanel:
    def __init__(
        self,
        parent_tag: str | int,
        on_load: Callable[[dict[str, Any]], None],
    ) -> None:
        self._parent = parent_tag
        self._on_load = on_load
        self._scenario_name: str = ""

    def build(self, scenario_name: str) -> None:
        self._scenario_name = scenario_name
        dpg.delete_item(self._parent, children_only=True)

        dpg.add_text("Load / Save Parameter Sets", parent=self._parent)
        dpg.add_separator(parent=self._parent)
        dpg.add_spacer(height=4, parent=self._parent)

        # Save row: name input + button
        with dpg.group(horizontal=True, parent=self._parent):
            dpg.add_input_text(
                tag="preset_name_input",
                hint="preset name",
                width=-80,
            )
            dpg.add_button(
                label="Save",
                tag="preset_save_btn",
                width=72,
                callback=self._on_save_clicked,
            )

        dpg.add_spacer(height=4, parent=self._parent)

        # Load row: dropdown (auto-loads on selection)
        presets = _list_presets(scenario_name)
        dpg.add_combo(
            tag="preset_load_combo",
            items=[_PLACEHOLDER] + presets,
            default_value=_PLACEHOLDER,
            label="##preset_load",
            width=-1,
            parent=self._parent,
            callback=self._on_preset_selected,
        )

        dpg.add_text("", tag="preset_status", parent=self._parent)

    def refresh(self, scenario_name: str) -> None:
        """Called when the scenario changes — repopulate the dropdown."""
        self._scenario_name = scenario_name
        presets = _list_presets(scenario_name)
        dpg.configure_item("preset_load_combo", items=[_PLACEHOLDER] + presets)
        dpg.set_value("preset_load_combo", _PLACEHOLDER)
        dpg.set_value("preset_status", "")

    def _on_save_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        name = dpg.get_value("preset_name_input").strip()
        if not name:
            dpg.set_value("preset_status", "Enter a name first.")
            return

        # The app sets this via set_params_getter before save
        if self._params_getter is None:
            return
        params = self._params_getter()

        dest = _scenario_dir(self._scenario_name)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{name}.json").write_text(json.dumps(params, indent=2))

        # Refresh dropdown and select the saved preset
        presets = _list_presets(self._scenario_name)
        dpg.configure_item("preset_load_combo", items=[_PLACEHOLDER] + presets)
        dpg.set_value("preset_load_combo", name)
        dpg.set_value("preset_status", f'Saved "{name}".')

    def _on_preset_selected(self, sender, app_data: str, user_data=None) -> None:
        if app_data == _PLACEHOLDER:
            return
        path = _scenario_dir(self._scenario_name) / f"{app_data}.json"
        if not path.exists():
            dpg.set_value("preset_status", f'"{app_data}" not found.')
            return
        params = json.loads(path.read_text())
        self._on_load(params)
        dpg.set_value("preset_status", f'Loaded "{app_data}".')

    def set_params_getter(self, getter: Callable[[], dict[str, Any]]) -> None:
        """Provide a callback that returns the current params dict for saving."""
        self._params_getter: Callable[[], dict[str, Any]] | None = getter

    # initialise the private attribute so mypy is happy
    _params_getter: Callable[[], dict[str, Any]] | None = None
