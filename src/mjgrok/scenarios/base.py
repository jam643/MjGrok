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
    def build_model_xml(self, params: dict[str, Any]) -> str: ...

    def build_model(self, params: dict[str, Any]) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string(self.build_model_xml(params))

    @abstractmethod
    def extract_series(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
    ) -> dict[str, float]:
        """Called at each sim step. Returns {series_key: scalar_value}."""
        ...

    def default_params(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.param_specs()}
