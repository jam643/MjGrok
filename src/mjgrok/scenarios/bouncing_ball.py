"""Bouncing Ball scenario: sphere dropped above a floor under 10 m/s² gravity.

Designed to build intuition for collision dynamics and energy conservation —
specifically how solimp/solref parameters shape the effective coefficient of
restitution and penetration depth at impact.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario

_INTEGRATOR_MAP = {
    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}

_SOLVER_MAP = {
    "PGS": mujoco.mjtSolver.mjSOL_PGS,
    "CG": mujoco.mjtSolver.mjSOL_CG,
    "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
}

_CONE_MAP = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}


class BouncingBallScenario(Scenario):
    name = "Bouncing Ball"
    description = (
        "A sphere dropped from height onto a rigid floor under 10 m/s^2 gravity. "
        "Explore how solimp/solref tune the effective restitution, energy loss per bounce, "
        "and contact penetration depth."
    )

    @property
    def sim_duration(self) -> float:
        return 4.0

    def param_specs(self) -> list[ParamSpec]:
        return [
            # Geometry group
            ParamSpec(
                "drop_height",
                "Drop Height (m)",
                "float",
                1.0,
                min_val=0.1,
                max_val=5.0,
                step=0.1,
                sweepable=True,
                tooltip="Initial height of the sphere's center above the floor",
                group="Geometry",
            ),
            ParamSpec(
                "sphere_radius",
                "Sphere Radius (m)",
                "float",
                0.05,
                min_val=0.01,
                max_val=0.3,
                step=0.005,
                sweepable=True,
                tooltip="Radius of the bouncing sphere",
                group="Geometry",
            ),
            # Physics group
            ParamSpec(
                "sphere_mass",
                "Sphere Mass (kg)",
                "float",
                0.1,
                min_val=0.01,
                max_val=5.0,
                step=0.01,
                sweepable=True,
                tooltip="Mass of the bouncing sphere",
                group="Physics",
            ),
            # Simulation group
            ParamSpec(
                "timestep",
                "Timestep (s)",
                "float",
                0.002,
                min_val=0.0001,
                max_val=0.02,
                step=0.0001,
                sweepable=True,
                tooltip="Simulation timestep - smaller values increase accuracy at impact",
                group="Simulation",
            ),
            ParamSpec(
                "integrator",
                "Integrator",
                "enum",
                "implicitfast",
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
                "Friction Cone",
                "enum",
                "elliptic",
                choices=["pyramidal", "elliptic"],
                sweepable=False,
                tooltip="Friction cone approximation type",
                group="Simulation",
            ),
            # Contact Solver group
            ParamSpec(
                "solimp_0",
                "solimp dmin",
                "float",
                0.9,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                sweepable=True,
                tooltip=(
                    "Constraint impedance dmin: minimum impedance in the contact zone. "
                    "Higher values → stiffer contact → less penetration but more bounce."
                ),
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
                sweepable=True,
                tooltip="Constraint impedance dmax: maximum impedance at full penetration",
                group="Contact Solver",
            ),
            ParamSpec(
                "solimp_2",
                "solimp width",
                "float",
                0.001,
                min_val=0.0001,
                max_val=0.1,
                step=0.0001,
                sweepable=True,
                tooltip="Constraint impedance width: penetration depth where transition occurs",
                group="Contact Solver",
            ),
            ParamSpec(
                "solimp_3",
                "solimp midpoint",
                "float",
                0.5,
                min_val=0.1,
                max_val=0.9,
                step=0.05,
                sweepable=True,
                tooltip="Constraint impedance midpoint: fraction of width at transition midpoint",
                group="Contact Solver",
            ),
            ParamSpec(
                "solimp_4",
                "solimp power",
                "float",
                2.0,
                min_val=1.0,
                max_val=5.0,
                step=0.1,
                sweepable=True,
                tooltip="Constraint impedance power: shape of impedance transition curve",
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
                sweepable=True,
                tooltip=(
                    "Constraint reference time constant (s): controls contact spring stiffness. "
                    "Smaller = stiffer spring = higher restitution. "
                    "Must be > 2x timestep to remain stable."
                ),
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
                sweepable=True,
                tooltip=(
                    "Constraint reference damping ratio: 1.0 = critically damped (no bounce), "
                    "<1.0 = under-damped (more bounce), >1.0 = over-damped."
                ),
                group="Contact Solver",
            ),
            ParamSpec(
                "impratio",
                "Impedance Ratio",
                "float",
                1.0,
                min_val=0.0,
                max_val=100.0,
                step=1.0,
                sweepable=True,
                tooltip="Ratio of frictional to normal constraint impedance",
                group="Contact Solver",
            ),
        ]

    def plot_specs(self) -> list[PlotSpec]:
        return [
            PlotSpec(
                "z_pos",
                "Ball Height",
                "time (s)",
                "Z Position (m)",
                ["z_pos"],
                group="Kinematics",
            ),
            PlotSpec(
                "z_vel",
                "Vertical Velocity",
                "time (s)",
                "Z Velocity (m/s)",
                ["z_vel"],
                group="Kinematics",
            ),
            PlotSpec(
                "energy",
                "Mechanical Energy",
                "time (s)",
                "Energy (J)",
                ["total_energy", "kinetic_energy", "potential_energy"],
                group="Energy",
            ),
            PlotSpec(
                "energy_loss",
                "Energy Loss per Bounce",
                "time (s)",
                "Energy Fraction Remaining",
                ["energy_fraction"],
                group="Energy",
            ),
            PlotSpec(
                "fn",
                "Contact Normal Force",
                "time (s)",
                "Normal Force (N)",
                ["fn"],
                group="Contact",
            ),
            PlotSpec(
                "max_pen",
                "Penetration Depth",
                "time (s)",
                "Penetration (m)",
                ["max_pen"],
                group="Contact",
            ),
            PlotSpec(
                "ncon",
                "Contact Count",
                "time (s)",
                "Contact Points",
                ["ncon"],
                group="Contact",
            ),
            PlotSpec(
                "solver_iter",
                "Solver Iterations",
                "time (s)",
                "Iterations",
                ["solver_niter"],
                group="Solver",
            ),
        ]

    def build_spec(self, params: dict[str, Any]) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()

        spec.option.gravity = [0.0, 0.0, -10.0]
        spec.option.timestep = float(params["timestep"])
        spec.option.integrator = _INTEGRATOR_MAP[params["integrator"]]
        spec.option.solver = _SOLVER_MAP[params["solver"]]
        spec.option.cone = _CONE_MAP[params["cone"]]
        spec.option.impratio = float(params["impratio"])

        solimp = [
            float(params["solimp_0"]),
            float(params["solimp_1"]),
            float(params["solimp_2"]),
            float(params["solimp_3"]),
            float(params["solimp_4"]),
        ]
        solref = [float(params["solref_0"]), float(params["solref_1"])]

        radius = float(params["sphere_radius"])
        drop_height = float(params["drop_height"])

        # ── Floor (static plane attached to worldbody) ────────────────────────
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [10.0, 10.0, 0.1]
        floor.pos = [0.0, 0.0, 0.0]
        floor.solimp = solimp
        floor.solref = solref
        floor.rgba = [0.4, 0.4, 0.4, 1.0]

        # ── Ball (freejoint) ──────────────────────────────────────────────────
        ball = spec.worldbody.add_body()
        ball.name = "ball"
        ball.pos = [0.0, 0.0, drop_height + radius]
        fj = ball.add_joint()
        fj.type = mujoco.mjtJoint.mjJNT_FREE
        bg = ball.add_geom()
        bg.type = mujoco.mjtGeom.mjGEOM_SPHERE
        bg.size = [radius, 0.0, 0.0]
        bg.mass = float(params["sphere_mass"])
        bg.solimp = solimp
        bg.solref = solref
        bg.rgba = [0.85, 0.3, 0.2, 1.0]

        return spec

    def extract_series(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
    ) -> dict[str, float]:
        # freejoint qpos: [x, y, z, qw, qx, qy, qz]
        # freejoint qvel: [vx, vy, vz, wx, wy, wz]
        z_pos = float(data.qpos[2])
        vx = float(data.qvel[0])
        vy = float(data.qvel[1])
        vz = float(data.qvel[2])

        mass = float(model.body_mass[1])  # ball is body index 1 (worldbody=0)
        g = 10.0

        kinetic_energy = 0.5 * mass * (vx * vx + vy * vy + vz * vz)
        potential_energy = mass * g * max(z_pos, 0.0)
        total_energy = kinetic_energy + potential_energy

        # Energy fraction relative to initial potential energy (drop height + radius)
        initial_z = float(model.body_pos[1][2])  # ball body pos[z] in default config
        initial_e = mass * g * initial_z
        energy_fraction = total_energy / initial_e if initial_e > 0.0 else 1.0

        fn = 0.0
        force_buf = np.zeros(6)
        for i in range(data.ncon):
            mujoco.mj_contactForce(model, data, i, force_buf)
            fn += abs(force_buf[0])

        max_pen = max((max(0.0, -data.contact[i].dist) for i in range(data.ncon)), default=0.0)
        solver_niter = int(data.solver_niter[0])

        return {
            "z_pos": z_pos,
            "z_vel": vz,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "total_energy": total_energy,
            "energy_fraction": energy_fraction,
            "fn": fn,
            "max_pen": max_pen,
            "ncon": float(data.ncon),
            "solver_niter": float(solver_niter),
        }
