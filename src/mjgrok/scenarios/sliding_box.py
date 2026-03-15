"""Sliding box scenario: box on floor under friction + constant external force."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario


class SlidingBoxScenario(Scenario):
    name = "Sliding Box"
    description = (
        "Box on a flat floor under constant external force. Explore friction and solver parameters."
    )

    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                "force_x",
                "Force X (N)",
                "float",
                5.0,
                min_val=-20.0,
                max_val=20.0,
                step=0.1,
                tooltip="External force applied along X axis each step",
                group="Force",
            ),
            ParamSpec(
                "friction_slide",
                "Slide Friction",
                "float",
                0.5,
                min_val=0.0,
                max_val=2.0,
                step=0.01,
                tooltip="Sliding friction coefficient (MuJoCo friction[0])",
                group="Friction",
            ),
            ParamSpec(
                "friction_spin",
                "Spin Friction",
                "float",
                0.005,
                min_val=0.0,
                max_val=0.1,
                step=0.001,
                tooltip="Torsional friction coefficient (MuJoCo friction[1])",
                group="Friction",
            ),
            ParamSpec(
                "friction_roll",
                "Roll Friction",
                "float",
                0.0001,
                min_val=0.0,
                max_val=0.01,
                step=0.0001,
                tooltip="Rolling friction coefficient (MuJoCo friction[2])",
                group="Friction",
            ),
            ParamSpec(
                "cone",
                "Contact Cone",
                "enum",
                "pyramidal",
                choices=["pyramidal", "elliptic"],
                sweepable=False,
                tooltip="Contact friction cone type",
                group="Contact Solver",
            ),
            ParamSpec(
                "solimp_0",
                "solimp dmin",
                "float",
                0.9,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                tooltip="Constraint impedance: minimum (dmin)",
                group="Contact Solver",
            ),
            ParamSpec(
                "solimp_1",
                "solimp dmax",
                "float",
                0.95,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                tooltip="Constraint impedance: maximum (dmax)",
                group="Contact Solver",
            ),
            ParamSpec(
                "solimp_2",
                "solimp width",
                "float",
                0.001,
                min_val=0.0,
                max_val=0.01,
                step=0.0001,
                tooltip="Constraint impedance: transition width",
                group="Contact Solver",
            ),
            ParamSpec(
                "solref_0",
                "solref timeconst",
                "float",
                0.02,
                min_val=0.001,
                max_val=0.5,
                step=0.001,
                tooltip="Constraint reference: time constant (s)",
                group="Contact Solver",
            ),
            ParamSpec(
                "solref_1",
                "solref dampratio",
                "float",
                1.0,
                min_val=0.1,
                max_val=2.0,
                step=0.05,
                tooltip="Constraint reference: damping ratio",
                group="Contact Solver",
            ),
            ParamSpec(
                "noslip_iterations",
                "No-slip Iters",
                "int",
                0,
                min_val=0,
                max_val=20,
                step=1,
                tooltip="Number of no-slip friction constraint iterations",
                group="Contact Solver",
            ),
            ParamSpec(
                "impratio",
                "impratio",
                "float",
                1.0,
                min_val=0.1,
                max_val=100.0,
                step=0.1,
                tooltip="Ratio of impedance to constraint reference (impratio)",
                group="Contact Solver",
            ),
        ]

    def plot_specs(self) -> list[PlotSpec]:
        return [
            PlotSpec("pos_x", "Position X", "time (s)", "x (m)", ["pos_x"]),
            PlotSpec("vel_x", "Velocity X", "time (s)", "vel (m/s)", ["vel_x"]),
            PlotSpec("fn", "Normal Contact Force", "time (s)", "Fn (N)", ["fn"]),
            PlotSpec("ft", "Tangential Friction Force", "time (s)", "Ft (N)", ["ft"]),
        ]

    def build_spec(self, params: dict[str, Any]) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()

        # ── Solver options ────────────────────────────────────────────────────
        spec.option.timestep = 0.002
        spec.option.cone = (
            mujoco.mjtCone.mjCONE_ELLIPTIC
            if params["cone"] == "elliptic"
            else mujoco.mjtCone.mjCONE_PYRAMIDAL
        )
        spec.option.noslip_iterations = int(params["noslip_iterations"])
        spec.option.impratio = float(params["impratio"])

        friction = [
            float(params["friction_slide"]),
            float(params["friction_spin"]),
            float(params["friction_roll"]),
        ]
        solimp = [
            float(params["solimp_0"]),
            float(params["solimp_1"]),
            float(params["solimp_2"]),
            0.5,  # midpoint (fixed)
            2.0,  # power (fixed)
        ]
        solref = [float(params["solref_0"]), float(params["solref_1"])]

        # ── Floor ─────────────────────────────────────────────────────────────
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [10.0, 10.0, 0.1]
        floor.friction = friction
        floor.solimp = solimp
        floor.solref = solref
        floor.rgba = [0.8, 0.8, 0.8, 1.0]

        # ── Box body ──────────────────────────────────────────────────────────
        box_body = spec.worldbody.add_body()
        box_body.name = "box"
        box_body.pos = [0.0, 0.0, 0.25]

        slide = box_body.add_joint()
        slide.name = "slide_x"
        slide.type = mujoco.mjtJoint.mjJNT_SLIDE
        slide.axis = [1, 0, 0]

        box_geom = box_body.add_geom()
        box_geom.name = "box_geom"
        box_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        box_geom.size = [0.25, 0.25, 0.25]
        box_geom.mass = 1.0
        box_geom.friction = friction
        box_geom.solimp = solimp
        box_geom.solref = solref
        box_geom.rgba = [0.2, 0.6, 0.9, 1.0]

        # ── Motor actuator for external force ─────────────────────────────────
        act = spec.add_actuator()
        act.set_to_motor()
        act.name = "force_x"
        act.target = "slide_x"
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT

        return spec

    def apply_ctrl(
        self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]
    ) -> None:
        data.ctrl[0] = float(params["force_x"])

    def extract_series(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
    ) -> dict[str, float]:
        pos_x = float(data.qpos[0])
        vel_x = float(data.qvel[0])

        fn = 0.0
        ft = 0.0
        force_buf = np.zeros(6)
        for i in range(data.ncon):
            mujoco.mj_contactForce(model, data, i, force_buf)
            fn += abs(force_buf[0])
            ft += float(np.linalg.norm(force_buf[1:3]))

        return {"pos_x": pos_x, "vel_x": vel_x, "fn": fn, "ft": ft}
