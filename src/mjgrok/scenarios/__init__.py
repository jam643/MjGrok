"""Scenario registry."""

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario
from mjgrok.scenarios.sliding_box import SlidingBoxScenario

SCENARIOS: list[Scenario] = [
    SlidingBoxScenario(),
]

# Keyed by Scenario.name — used by viewer worker to reconstruct model in subprocess
SCENARIO_REGISTRY: dict[str, Scenario] = {s.name: s for s in SCENARIOS}

__all__ = [
    "ParamSpec",
    "PlotSpec",
    "Scenario",
    "SlidingBoxScenario",
    "SCENARIOS",
    "SCENARIO_REGISTRY",
]
