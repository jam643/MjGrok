"""Parameter panel: auto-generates DearPyGUI widgets from ParamSpec list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np

from mjgrok.scenarios.base import ParamSpec, Scenario


@dataclass
class SweepConfig:
    name: str
    values: list[float | int]


class ParamPanel:
    def __init__(self, parent_tag: str | int) -> None:
        self._parent = parent_tag
        self._specs: list[ParamSpec] = []

    def build(self, scenario: Scenario) -> None:
        self._specs = scenario.param_specs()
        dpg.delete_item(self._parent, children_only=True)
        for spec in self._specs:
            self._build_param(spec)

    def _build_param(self, spec: ParamSpec) -> None:
        # ── Parameter label (full width, tooltip on hover) ────────────────────
        lbl_id = dpg.add_text(spec.label, parent=self._parent)
        if spec.tooltip:
            with dpg.tooltip(parent=lbl_id):
                dpg.add_text(spec.tooltip)

        # ── Sweep checkbox (sweepable params only) ────────────────────────────
        if spec.sweepable:
            dpg.add_checkbox(
                tag=f"sweep_{spec.name}",
                label="Sweep",
                default_value=False,
                callback=self._on_sweep_toggle,
                user_data=spec.name,
                parent=self._parent,
            )

        # ── Input widget (slider or combo) ────────────────────────────────────
        tag = f"param_{spec.name}"
        if spec.dtype == "float":
            dpg.add_slider_float(
                tag=tag,
                label=f"##{spec.name}",
                default_value=float(spec.default),
                min_value=float(spec.min_val or 0.0),
                max_value=float(spec.max_val or 1.0),
                parent=self._parent,
                width=-1,
            )
        elif spec.dtype == "int":
            dpg.add_slider_int(
                tag=tag,
                label=f"##{spec.name}",
                default_value=int(spec.default),
                min_value=int(spec.min_val or 0),
                max_value=int(spec.max_val or 10),
                parent=self._parent,
                width=-1,
            )
        elif spec.dtype == "enum":
            dpg.add_combo(
                tag=tag,
                label=f"##{spec.name}",
                items=spec.choices or [],
                default_value=str(spec.default),
                parent=self._parent,
                width=-1,
            )

        # ── Sweep range inputs (hidden until checkbox is ticked) ──────────────
        if spec.sweepable:
            with (
                dpg.group(tag=f"sweep_range_{spec.name}", parent=self._parent, show=False),
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

        dpg.add_spacer(height=3, parent=self._parent)

    def _on_sweep_toggle(self, sender, app_data: bool, user_data: str) -> None:
        name = user_data
        if app_data:
            dpg.hide_item(f"param_{name}")
            dpg.show_item(f"sweep_range_{name}")
        else:
            dpg.show_item(f"param_{name}")
            dpg.hide_item(f"sweep_range_{name}")

    def collect_params(self) -> dict[str, Any]:
        """Return single-value params. Swept params return their slider's current value."""
        params: dict[str, Any] = {}
        for spec in self._specs:
            val = dpg.get_value(f"param_{spec.name}")
            if spec.dtype == "int":
                params[spec.name] = int(val)
            elif spec.dtype == "float":
                params[spec.name] = float(val)
            else:
                params[spec.name] = str(val)
        return params

    def has_active_sweeps(self) -> bool:
        return any(spec.sweepable and dpg.get_value(f"sweep_{spec.name}") for spec in self._specs)

    def get_sweep_configs(self) -> list[SweepConfig]:
        """Return sweep ranges for all params with sweep enabled."""
        configs = []
        for spec in self._specs:
            if not spec.sweepable or not dpg.get_value(f"sweep_{spec.name}"):
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
