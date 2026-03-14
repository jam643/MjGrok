"""Parameter panel: auto-generates DearPyGUI widgets from ParamSpec list."""

from __future__ import annotations

from typing import Any

import dearpygui.dearpygui as dpg

from mjgrok.scenarios.base import ParamSpec, Scenario


class ParamPanel:
    def __init__(self, parent_tag: str | int) -> None:
        self._parent = parent_tag
        self._specs: list[ParamSpec] = []

    def build(self, scenario: Scenario) -> None:
        """Create widgets for all param specs of the given scenario."""
        self._specs = scenario.param_specs()
        dpg.delete_item(self._parent, children_only=True)

        for spec in self._specs:
            tag = f"param_{spec.name}"
            label = spec.label
            tooltip = spec.tooltip

            if spec.dtype == "float":
                dpg.add_slider_float(
                    tag=tag,
                    label=label,
                    default_value=float(spec.default),
                    min_value=float(spec.min_val or 0.0),
                    max_value=float(spec.max_val or 1.0),
                    parent=self._parent,
                    width=-1,
                )
            elif spec.dtype == "int":
                dpg.add_slider_int(
                    tag=tag,
                    label=label,
                    default_value=int(spec.default),
                    min_value=int(spec.min_val or 0),
                    max_value=int(spec.max_val or 10),
                    parent=self._parent,
                    width=-1,
                )
            elif spec.dtype == "enum":
                dpg.add_combo(
                    tag=tag,
                    label=label,
                    items=spec.choices or [],
                    default_value=str(spec.default),
                    parent=self._parent,
                    width=-1,
                )

            if tooltip:
                with dpg.tooltip(parent=tag):
                    dpg.add_text(tooltip)

    def collect_params(self) -> dict[str, Any]:
        """Read current widget values into a params dict."""
        params: dict[str, Any] = {}
        for spec in self._specs:
            tag = f"param_{spec.name}"
            val = dpg.get_value(tag)
            if spec.dtype == "int":
                params[spec.name] = int(val)
            elif spec.dtype == "float":
                params[spec.name] = float(val)
            else:
                params[spec.name] = str(val)
        return params
