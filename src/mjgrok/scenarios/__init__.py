"""Scenario registry."""

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario
from mjgrok.scenarios.sliding_box import SlidingBoxScenario

SCENARIOS: list[Scenario] = [
    SlidingBoxScenario(),
]

__all__ = ["ParamSpec", "PlotSpec", "Scenario", "SlidingBoxScenario", "SCENARIOS"]
