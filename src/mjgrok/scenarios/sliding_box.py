"""Sliding box scenario: box on floor under friction + constant external force."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario


class SlidingBoxScenario(Scenario):
    name = "Sliding Box"
    description = (
        "Box on a flat floor under constant external force. "
        "Gravity = 10 m/s². Explore box geometry, mass, friction, and solver parameters."
    )

    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                "box_half_width",
                "Half-width (m)",
                "float",
                0.25,
                min_val=0.05,
                max_val=1.0,
                step=0.05,
                tooltip="Half-extent of the box in X and Y",
                group="Box",
            ),
            ParamSpec(
                "box_half_height",
                "Half-height (m)",
                "float",
                0.25,
                min_val=0.05,
                max_val=1.0,
                step=0.05,
                tooltip="Half-extent of the box in Z",
                group="Box",
            ),
            ParamSpec(
                "box_mass",
                "Mass (kg)",
                "float",
                1.0,
                min_val=0.1,
                max_val=50.0,
                step=0.1,
                tooltip="Mass of the box",
                group="Box",
            ),
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
                "timestep",
                "Timestep (s)",
                "float",
                0.002,
                min_val=0.0001,
                max_val=0.02,
                step=0.0001,
                tooltip="Simulation timestep",
                group="Simulation",
            ),
            ParamSpec(
                "integrator",
                "Integrator",
                "enum",
                "Euler",
                choices=["Euler", "RK4", "implicit", "implicitfast"],
                sweepable=False,
                tooltip="Numerical integrator for the equations of motion",
                group="Simulation",
            ),
            ParamSpec(
                "solver",
                "Solver",
                "enum",
                "Newton",
                choices=["PGS", "CG", "Newton"],
                sweepable=False,
                tooltip="Constraint solver algorithm",
                group="Simulation",
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
        spec.option.timestep = float(params["timestep"])
        spec.option.gravity = [0.0, 0.0, -10.0]
        _integrator_map = {
            "Euler": mujoco.mjtIntegrator.mjINT_EULER,
            "RK4": mujoco.mjtIntegrator.mjINT_RK4,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }
        spec.option.integrator = _integrator_map[params["integrator"]]
        _solver_map = {
            "PGS": mujoco.mjtSolver.mjSOL_PGS,
            "CG": mujoco.mjtSolver.mjSOL_CG,
            "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }
        spec.option.solver = _solver_map[params["solver"]]
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
        box_hw = float(params["box_half_width"])
        box_hh = float(params["box_half_height"])
        box_body = spec.worldbody.add_body()
        box_body.name = "box"
        box_body.pos = [0.0, 0.0, box_hh]
        free = box_body.add_joint()
        free.name = "free"
        free.type = mujoco.mjtJoint.mjJNT_FREE

        box_geom = box_body.add_geom()
        box_geom.name = "box_geom"
        box_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        box_geom.size = [box_hw, box_hw, box_hh]
        box_geom.mass = float(params["box_mass"])
        box_geom.friction = friction
        box_geom.solimp = solimp
        box_geom.solref = solref
        box_geom.rgba = [0.2, 0.6, 0.9, 1.0]

        return spec

    def apply_ctrl(
        self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]
    ) -> None:
        # Apply force directly on the box body in X — no actuator needed with free joint
        body_id = model.body("box").id
        data.xfrc_applied[body_id, 0] = float(params["force_x"])

    def extract_series(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
    ) -> dict[str, float]:
        # Free joint: qpos = [x, y, z, qw, qx, qy, qz], qvel = [vx, vy, vz, wx, wy, wz]
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
