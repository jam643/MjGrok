"""Penetrating sphere scenario: sphere dropped into floor to study contact constraint behavior."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario


class PenetratingSphereScenario(Scenario):
    name = "Penetrating Sphere"
    description = (
        "Sphere initially embedded in the floor. "
        "Explore how solimp/solref/impratio govern constraint stiffness and penetration resolution."
    )

    @property
    def sim_duration(self) -> float:
        return 1.0

    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                "sphere_radius",
                "Sphere Radius (m)",
                "float",
                0.1,
                min_val=0.01,
                max_val=0.5,
                step=0.01,
                tooltip="Radius of the sphere",
                group="Geometry",
            ),
            ParamSpec(
                "penetration_depth",
                "Penetration Depth (x radius)",
                "float",
                1.0,
                min_val=0.0,
                max_val=2.0,
                step=0.01,
                tooltip=(
                    "Initial penetration depth as a fraction of sphere radius. "
                    "0 = resting on floor."
                ),
                group="Geometry",
            ),
            ParamSpec(
                "sphere_mass",
                "Mass (kg)",
                "float",
                1.0,
                min_val=0.01,
                max_val=10.0,
                step=0.01,
                tooltip="Mass of the sphere",
                group="Physics",
            ),
            ParamSpec(
                "timestep",
                "Timestep (s)",
                "float",
                0.002,
                min_val=0.0001,
                max_val=0.01,
                step=0.0001,
                tooltip="Simulation timestep",
                group="Solver",
            ),
            ParamSpec(
                "integrator",
                "Integrator",
                "enum",
                "Euler",
                choices=["Euler", "RK4", "implicit", "implicitfast"],
                sweepable=False,
                tooltip="Numerical integrator for the equations of motion",
                group="Solver",
            ),
            ParamSpec(
                "solver",
                "Solver",
                "enum",
                "Newton",
                choices=["CG", "Newton", "PGS"],
                sweepable=False,
                tooltip="Constraint solver algorithm",
                group="Solver",
            ),
            ParamSpec(
                "solimp_dmin",
                "solimp dmin",
                "float",
                0.9,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                tooltip="Constraint impedance: minimum (dmin)",
                group="Constraint",
            ),
            ParamSpec(
                "solimp_dmax",
                "solimp dmax",
                "float",
                0.95,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                tooltip="Constraint impedance: maximum (dmax)",
                group="Constraint",
            ),
            ParamSpec(
                "solimp_width",
                "solimp width",
                "float",
                0.001,
                min_val=0.0001,
                max_val=0.05,
                step=0.0001,
                tooltip="Constraint impedance: transition width",
                group="Constraint",
            ),
            ParamSpec(
                "solimp_midpoint",
                "solimp midpoint",
                "float",
                0.5,
                min_val=0.1,
                max_val=0.9,
                step=0.05,
                tooltip="Constraint impedance: midpoint of transition",
                group="Constraint",
            ),
            ParamSpec(
                "solimp_power",
                "solimp power",
                "float",
                2.0,
                min_val=1.0,
                max_val=5.0,
                step=0.1,
                tooltip="Constraint impedance: power of transition curve",
                group="Constraint",
            ),
            ParamSpec(
                "solref_timeconst",
                "solref timeconst (s)",
                "float",
                0.02,
                min_val=0.001,
                max_val=0.5,
                step=0.001,
                tooltip="Constraint reference: time constant (s)",
                group="Constraint",
            ),
            ParamSpec(
                "solref_dampratio",
                "solref dampratio",
                "float",
                1.0,
                min_val=0.1,
                max_val=2.0,
                step=0.05,
                tooltip="Constraint reference: damping ratio",
                group="Constraint",
            ),
            ParamSpec(
                "impratio",
                "impratio",
                "float",
                1.0,
                min_val=0.0,
                max_val=100.0,
                step=0.1,
                tooltip="Ratio of impedance to constraint reference (impratio)",
                group="Constraint",
            ),
        ]

    def plot_specs(self) -> list[PlotSpec]:
        return [
            PlotSpec(
                "z_pos",
                "Z Position",
                "time (s)",
                "Z Position (m)",
                ["z_pos"],
                group="Kinematics",
            ),
            PlotSpec(
                "z_vel",
                "Z Velocity",
                "time (s)",
                "Z Velocity (m/s)",
                ["z_vel"],
                group="Kinematics",
            ),
            PlotSpec(
                "fn",
                "Normal Contact Force",
                "time (s)",
                "Normal Force (N)",
                ["fn"],
                group="Contact",
            ),
            PlotSpec(
                "ncon",
                "Contact Points",
                "time (s)",
                "Contact Points",
                ["ncon"],
                group="Contact",
            ),
            PlotSpec(
                "solver_iter",
                "Solver Iterations",
                "time (s)",
                "Solver Iterations",
                ["solver_niter"],
                group="Solver",
            ),
            PlotSpec(
                "max_pen",
                "Max Penetration Depth",
                "time (s)",
                "Max Penetration Depth (m)",
                ["max_pen"],
                group="Solver",
            ),
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
        spec.option.impratio = float(params["impratio"])

        solimp = [
            float(params["solimp_dmin"]),
            float(params["solimp_dmax"]),
            float(params["solimp_width"]),
            float(params["solimp_midpoint"]),
            float(params["solimp_power"]),
        ]
        solref = [float(params["solref_timeconst"]), float(params["solref_dampratio"])]

        # ── Floor ─────────────────────────────────────────────────────────────
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [10.0, 10.0, 0.1]
        floor.solimp = solimp
        floor.solref = solref
        floor.rgba = [0.8, 0.8, 0.8, 1.0]

        # ── Sphere body ───────────────────────────────────────────────────────
        radius = float(params["sphere_radius"])
        pen_depth = float(params["penetration_depth"])
        # Initial Z: sphere_radius * (1 - penetration_depth) puts center below floor level
        # when penetration_depth > 0, creating initial penetration
        init_z = radius * (1.0 - pen_depth)

        sphere_body = spec.worldbody.add_body()
        sphere_body.name = "sphere"
        sphere_body.pos = [0.0, 0.0, init_z]

        free = sphere_body.add_joint()
        free.name = "free"
        free.type = mujoco.mjtJoint.mjJNT_FREE

        sphere_geom = sphere_body.add_geom()
        sphere_geom.name = "sphere_geom"
        sphere_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        sphere_geom.size = [radius, 0.0, 0.0]
        sphere_geom.mass = float(params["sphere_mass"])
        sphere_geom.friction = [0.0, 0.0, 0.0]
        sphere_geom.solimp = solimp
        sphere_geom.solref = solref
        sphere_geom.rgba = [0.2, 0.6, 0.9, 1.0]

        return spec

    def extract_series(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
    ) -> dict[str, float]:
        # Free joint: qpos = [x, y, z, qw, qx, qy, qz], qvel = [vx, vy, vz, wx, wy, wz]
        z_pos = float(data.qpos[2])
        z_vel = float(data.qvel[2])

        fn = 0.0
        force_buf = np.zeros(6)
        for i in range(data.ncon):
            mujoco.mj_contactForce(model, data, i, force_buf)
            fn += abs(force_buf[0])

        solver_niter = int(data.solver_niter[0])
        max_pen = max((max(0.0, -data.contact[i].dist) for i in range(data.ncon)), default=0.0)

        return {
            "z_pos": z_pos,
            "z_vel": z_vel,
            "fn": fn,
            "ncon": float(data.ncon),
            "solver_niter": float(solver_niter),
            "max_pen": max_pen,
        }
