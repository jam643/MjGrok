"""Scenario registry."""

from mjgrok.scenarios.actuated_arm import ActuatedArmScenario
from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario
from mjgrok.scenarios.parallel_jaw_grasp import ParallelJawGraspScenario
from mjgrok.scenarios.penetrating_sphere import PenetratingSphereScenario
from mjgrok.scenarios.sliding_box import SlidingBoxScenario

SCENARIOS: list[Scenario] = [
    SlidingBoxScenario(),
    PenetratingSphereScenario(),
    ParallelJawGraspScenario(),
    ActuatedArmScenario(),
]

# Keyed by Scenario.name — used by viewer worker to reconstruct model in subprocess
SCENARIO_REGISTRY: dict[str, Scenario] = {s.name: s for s in SCENARIOS}

__all__ = [
    "ParamSpec",
    "PlotSpec",
    "Scenario",
    "ActuatedArmScenario",
    "ParallelJawGraspScenario",
    "PenetratingSphereScenario",
    "SlidingBoxScenario",
    "SCENARIOS",
    "SCENARIO_REGISTRY",
]
