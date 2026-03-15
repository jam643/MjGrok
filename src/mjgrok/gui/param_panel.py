"""Parameter panel: auto-generates DearPyGUI widgets from ParamSpec list."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np

from mjgrok.scenarios.base import ParamSpec, Scenario

_INPUT_TYPE_OPTIONS = ["slider", "text input", "sweep"]


@dataclass
class SweepConfig:
    name: str
    values: list[float | int]


class ParamPanel:
    def __init__(self, parent_tag: str | int, on_change: Callable | None = None) -> None:
        self._parent = parent_tag
        self._specs: list[ParamSpec] = []
        self._on_change = on_change

    def build(self, scenario: Scenario) -> None:
        self._specs = scenario.param_specs()
        dpg.delete_item(self._parent, children_only=True)

        # Split into ungrouped (shown at top) and grouped sections
        ungrouped = [s for s in self._specs if not s.group]
        groups: dict[str, list[ParamSpec]] = {}
        for spec in self._specs:
            if spec.group:
                groups.setdefault(spec.group, []).append(spec)

        for spec in ungrouped:
            self._build_param(spec, parent=self._parent)

        for group_name, specs in groups.items():
            header = dpg.add_collapsing_header(
                label=group_name,
                default_open=True,
                parent=self._parent,
            )
            inner = dpg.add_group(parent=header, indent=8)
            for spec in specs:
                self._build_param(spec, parent=inner)

    def _build_param(self, spec: ParamSpec, parent: str | int) -> None:
        # ── Parameter label (full width, tooltip on hover) ────────────────────
        lbl_id = dpg.add_text(spec.label, parent=parent)
        if spec.tooltip:
            with dpg.tooltip(parent=lbl_id):
                dpg.add_text(spec.tooltip)

        # ── Selection type combo (sweepable params only) ──────────────────────
        if spec.sweepable:
            dpg.add_combo(
                tag=f"input_type_{spec.name}",
                items=_INPUT_TYPE_OPTIONS,
                default_value="slider",
                label=f"##input_type_{spec.name}",
                callback=self._on_input_type_changed,
                user_data=spec.name,
                parent=parent,
                width=-1,
            )

        # ── Slider widget ─────────────────────────────────────────────────────
        if spec.dtype == "float":
            dpg.add_slider_float(
                tag=f"param_{spec.name}",
                label=f"##{spec.name}",
                default_value=float(spec.default),
                min_value=float(spec.min_val or 0.0),
                max_value=float(spec.max_val or 1.0),
                callback=self._on_change,
                parent=parent,
                width=-1,
            )
        elif spec.dtype == "int":
            dpg.add_slider_int(
                tag=f"param_{spec.name}",
                label=f"##{spec.name}",
                default_value=int(spec.default),
                min_value=int(spec.min_val or 0),
                max_value=int(spec.max_val or 10),
                callback=self._on_change,
                parent=parent,
                width=-1,
            )
        elif spec.dtype == "enum":
            dpg.add_combo(
                tag=f"param_{spec.name}",
                label=f"##{spec.name}",
                items=spec.choices or [],
                default_value=str(spec.default),
                callback=self._on_change,
                parent=parent,
                width=-1,
            )

        # ── Text input widget (hidden until "text input" is selected) ─────────
        if spec.sweepable and spec.dtype != "enum":
            if spec.dtype == "int":
                dpg.add_input_int(
                    tag=f"param_input_{spec.name}",
                    default_value=int(spec.default),
                    min_value=int(spec.min_val or 0),
                    max_value=int(spec.max_val or 10),
                    min_clamped=True,
                    max_clamped=True,
                    callback=self._on_change,
                    parent=parent,
                    width=-1,
                    show=False,
                    step=0,
                )
            else:
                dpg.add_input_float(
                    tag=f"param_input_{spec.name}",
                    default_value=float(spec.default),
                    min_value=float(spec.min_val or 0.0),
                    max_value=float(spec.max_val or 1.0),
                    min_clamped=True,
                    max_clamped=True,
                    callback=self._on_change,
                    parent=parent,
                    width=-1,
                    show=False,
                    step=0,
                    format="%.4g",
                )

        # ── Sweep range inputs (hidden until "sweep" is selected) ─────────────
        if spec.sweepable:
            with (
                dpg.group(tag=f"sweep_range_{spec.name}", parent=parent, show=False),
                dpg.group(horizontal=True),
            ):
                dpg.add_text("Min")
                if spec.dtype == "int":
                    dpg.add_input_int(
                        tag=f"sweep_min_{spec.name}",
                        default_value=int(spec.min_val or 0),
                        width=80,
                        step=0,
                    )
                    dpg.add_text("Max")
                    dpg.add_input_int(
                        tag=f"sweep_max_{spec.name}",
                        default_value=int(spec.max_val or 10),
                        width=80,
                        step=0,
                    )
                else:
                    dpg.add_input_float(
                        tag=f"sweep_min_{spec.name}",
                        default_value=float(spec.min_val or 0.0),
                        width=80,
                        step=0,
                        format="%.4g",
                    )
                    dpg.add_text("Max")
                    dpg.add_input_float(
                        tag=f"sweep_max_{spec.name}",
                        default_value=float(spec.max_val or 1.0),
                        width=80,
                        step=0,
                        format="%.4g",
                    )
                dpg.add_text("N")
                dpg.add_input_int(
                    tag=f"sweep_n_{spec.name}",
                    default_value=5,
                    min_value=2,
                    max_value=50,
                    width=55,
                    step=0,
                )

        dpg.add_spacer(height=3, parent=parent)

    def _on_input_type_changed(self, sender, app_data: str, user_data: str) -> None:
        name = user_data
        show_slider = app_data == "slider"
        show_text = app_data == "text input"
        show_sweep = app_data == "sweep"

        dpg.configure_item(f"param_{name}", show=show_slider)
        if dpg.does_item_exist(f"param_input_{name}"):
            dpg.configure_item(f"param_input_{name}", show=show_text)
        dpg.configure_item(f"sweep_range_{name}", show=show_sweep)

    def reset_to_defaults(self) -> None:
        """Reset all parameter widgets to their default values."""
        for spec in self._specs:
            if spec.dtype == "float":
                dpg.set_value(f"param_{spec.name}", float(spec.default))
                if dpg.does_item_exist(f"param_input_{spec.name}"):
                    dpg.set_value(f"param_input_{spec.name}", float(spec.default))
            elif spec.dtype == "int":
                dpg.set_value(f"param_{spec.name}", int(spec.default))
                if dpg.does_item_exist(f"param_input_{spec.name}"):
                    dpg.set_value(f"param_input_{spec.name}", int(spec.default))
            else:
                dpg.set_value(f"param_{spec.name}", str(spec.default))
        if self._on_change:
            self._on_change(None, None, None)

    def apply_params(self, params: dict[str, Any]) -> None:
        """Set widget values from a params dict (e.g. loaded from a preset)."""
        for spec in self._specs:
            if spec.name not in params:
                continue
            val = params[spec.name]
            if spec.dtype == "float":
                dpg.set_value(f"param_{spec.name}", float(val))
                if dpg.does_item_exist(f"param_input_{spec.name}"):
                    dpg.set_value(f"param_input_{spec.name}", float(val))
            elif spec.dtype == "int":
                dpg.set_value(f"param_{spec.name}", int(val))
                if dpg.does_item_exist(f"param_input_{spec.name}"):
                    dpg.set_value(f"param_input_{spec.name}", int(val))
            else:
                dpg.set_value(f"param_{spec.name}", str(val))
        if self._on_change:
            self._on_change(None, None, None)

    def collect_params(self) -> dict[str, Any]:
        """Return single-value params, reading from whichever widget is active."""
        params: dict[str, Any] = {}
        for spec in self._specs:
            mode = dpg.get_value(f"input_type_{spec.name}") if spec.sweepable else "slider"
            if mode == "text input" and dpg.does_item_exist(f"param_input_{spec.name}"):
                raw = dpg.get_value(f"param_input_{spec.name}")
            else:
                raw = dpg.get_value(f"param_{spec.name}")
            if spec.dtype == "int":
                params[spec.name] = int(raw)
            elif spec.dtype == "float":
                params[spec.name] = float(raw)
            else:
                params[spec.name] = str(raw)
        return params

    def has_active_sweeps(self) -> bool:
        return any(
            spec.sweepable and dpg.get_value(f"input_type_{spec.name}") == "sweep"
            for spec in self._specs
        )

    def get_sweep_configs(self) -> list[SweepConfig]:
        """Return sweep ranges for all params set to sweep mode."""
        configs = []
        for spec in self._specs:
            if not spec.sweepable or dpg.get_value(f"input_type_{spec.name}") != "sweep":
                continue
            lo = float(dpg.get_value(f"sweep_min_{spec.name}"))
            hi = float(dpg.get_value(f"sweep_max_{spec.name}"))
            n = int(dpg.get_value(f"sweep_n_{spec.name}"))
            if spec.dtype == "int":
                values: list[float | int] = [int(v) for v in np.linspace(lo, hi, n)]
            else:
                values = list(np.linspace(lo, hi, n))
            configs.append(SweepConfig(name=spec.name, values=values))
        return configs
