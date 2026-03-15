"""Core abstractions for MjGrok scenarios."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import mujoco


@dataclass
class ParamSpec:
    name: str
    label: str
    dtype: Literal["float", "int", "enum"]
    default: Any
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    choices: list[str] | None = None  # for enum dtype
    sweepable: bool = True
    tooltip: str = ""
    group: str = ""  # collapsible section label; empty = ungrouped (shown at top)


@dataclass
class PlotSpec:
    plot_id: str
    title: str
    x_label: str
    y_label: str
    series_keys: list[str]  # keys into TrajectoryCache.series


class Scenario(ABC):
    name: str
    description: str

    @abstractmethod
    def param_specs(self) -> list[ParamSpec]: ...

    @abstractmethod
    def plot_specs(self) -> list[PlotSpec]: ...

    @abstractmethod
    def build_spec(self, params: dict[str, Any]) -> mujoco.MjSpec:
        """Build and return a configured MjSpec for the given parameters.

        Concrete scenarios implement this. build_model() and build_model_xml()
        are derived from it automatically.
        """
        ...

    def build_model(self, params: dict[str, Any]) -> mujoco.MjModel:
        return self.build_spec(params).compile()

    def build_model_xml(self, params: dict[str, Any]) -> str:
        """Return the compiled XML string (used for debugging / IPC)."""
        return self.build_spec(params).to_xml()

    @abstractmethod
    def extract_series(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
    ) -> dict[str, float]:
        """Called at each sim step. Returns {series_key: scalar_value}."""
        ...

    def apply_ctrl(  # noqa: B027
        self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]
    ) -> None:
        """Set data.ctrl each step before mj_step. Default: no-op."""

    def default_params(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.param_specs()}
