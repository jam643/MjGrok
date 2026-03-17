"""Parallel Jaw Grasp scenario: two finger geoms gripping an object against gravity."""

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

_GEOM_TYPE = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
}


def _finger_size_arr(geom: str, s: float) -> list[float]:
    if geom == "sphere":
        return [s, 0.0, 0.0]
    if geom == "cylinder":
        return [s, s * 2, 0.0]
    # box
    return [s, s, s * 2]


def _object_size_arr(geom: str, s: float) -> list[float]:
    if geom == "sphere":
        return [s, 0.0, 0.0]
    if geom == "cylinder":
        return [s, s, 0.0]
    # box
    return [s, s, s]


class ParallelJawGraspScenario(Scenario):
    name = "Parallel Jaw Grasp"
    description = (
        "Two parallel finger geoms grip an object that gravity tries to pull out. "
        "Explore how friction, grasp force, and contact solver parameters determine stability."
    )

    @property
    def sim_duration(self) -> float:
        return 5.0

    def param_specs(self) -> list[ParamSpec]:
        return [
            # Geometry group
            ParamSpec(
                "finger_geom",
                "Finger Geom",
                "enum",
                "sphere",
                choices=["sphere", "box", "cylinder"],
                sweepable=False,
                tooltip="Geometry type for the finger contact surfaces",
                group="Geometry",
            ),
            ParamSpec(
                "object_geom",
                "Object Geom",
                "enum",
                "sphere",
                choices=["sphere", "box", "cylinder"],
                sweepable=False,
                tooltip="Geometry type for the grasped object",
                group="Geometry",
            ),
            ParamSpec(
                "finger_size",
                "Finger Size (m)",
                "float",
                0.05,
                min_val=0.01,
                max_val=0.15,
                step=0.005,
                sweepable=True,
                tooltip="Finger contact extent in x (radius or half-width)",
                group="Geometry",
            ),
            ParamSpec(
                "object_size",
                "Object Size (m)",
                "float",
                0.04,
                min_val=0.01,
                max_val=0.12,
                step=0.005,
                sweepable=True,
                tooltip="Object contact extent in x (radius or half-width)",
                group="Geometry",
            ),
            # Physics group
            ParamSpec(
                "object_mass",
                "Object Mass (kg)",
                "float",
                0.1,
                min_val=0.01,
                max_val=5.0,
                step=0.01,
                sweepable=True,
                tooltip="Mass of the grasped object",
                group="Physics",
            ),
            ParamSpec(
                "grasp_force",
                "Grasp Force (N)",
                "float",
                5.0,
                min_val=0.0,
                max_val=50.0,
                step=0.5,
                sweepable=True,
                tooltip="Force applied by the movable finger pressing inward",
                group="Physics",
            ),
            # Friction group
            ParamSpec(
                "friction_slide",
                "Sliding Friction",
                "float",
                0.8,
                min_val=0.0,
                max_val=3.0,
                step=0.05,
                sweepable=True,
                tooltip="Sliding friction coefficient between fingers and object",
                group="Friction",
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
                tooltip="Simulation timestep",
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
            # Contact Solver group
            ParamSpec(
                "cone",
                "Friction Cone",
                "enum",
                "elliptic",
                choices=["pyramidal", "elliptic"],
                sweepable=False,
                tooltip="Friction cone approximation type",
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
                sweepable=True,
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
                sweepable=True,
                tooltip="Constraint impedance: maximum (dmax)",
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
                sweepable=True,
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
                sweepable=True,
                tooltip="Constraint reference: damping ratio",
                group="Contact Solver",
            ),
            ParamSpec(
                "impratio",
                "Impedance Ratio",
                "float",
                1.0,
                min_val=1.0,
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
                "Object Z Position",
                "time (s)",
                "Z Position (m)",
                ["z_pos"],
                group="Kinematics",
            ),
            PlotSpec(
                "z_vel",
                "Object Z Velocity",
                "time (s)",
                "Z Velocity (m/s)",
                ["z_vel"],
                group="Kinematics",
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
                "Solver Iterations",
                ["solver_niter"],
                group="Solver",
            ),
            PlotSpec(
                "max_pen",
                "Max Penetration Depth",
                "time (s)",
                "Max Penetration (m)",
                ["max_pen"],
                group="Solver",
            ),
        ]

    def build_spec(self, params: dict[str, Any]) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()

        spec.option.gravity = [0.0, 0.0, -9.81]
        spec.option.timestep = float(params["timestep"])
        spec.option.integrator = _INTEGRATOR_MAP[params["integrator"]]
        spec.option.solver = _SOLVER_MAP[params["solver"]]
        spec.option.cone = _CONE_MAP[params["cone"]]
        spec.option.impratio = float(params["impratio"])

        solimp = [
            float(params["solimp_0"]),
            float(params["solimp_1"]),
            float(params["solimp_2"]),
            0.5,
            2.0,
        ]
        solref = [float(params["solref_0"]), float(params["solref_1"])]
        friction = [float(params["friction_slide"]), 0.005, 0.0001]

        finger_geom = params["finger_geom"]
        object_geom = params["object_geom"]
        fs = float(params["finger_size"])
        os_ = float(params["object_size"])

        finger_pos_x = fs + os_
        height = 1.0

        # ── Fixed finger (attached to world body, no joint) ───────────────────
        fb = spec.worldbody.add_body()
        fb.name = "fixed_finger"
        fb.pos = [-finger_pos_x, 0.0, height]
        fg = fb.add_geom()
        fg.type = _GEOM_TYPE[finger_geom]
        fg.size = _finger_size_arr(finger_geom, fs)
        fg.friction = friction
        fg.solimp = solimp
        fg.solref = solref
        fg.rgba = [0.3, 0.6, 0.9, 1.0]

        # ── Movable finger (slide joint axis [-1,0,0] → positive ctrl = inward) ──
        mb = spec.worldbody.add_body()
        mb.name = "movable_finger"
        mb.pos = [finger_pos_x, 0.0, height]
        mj = mb.add_joint()
        mj.name = "finger_slide"
        mj.type = mujoco.mjtJoint.mjJNT_SLIDE
        mj.axis = [-1.0, 0.0, 0.0]
        mg = mb.add_geom()
        mg.type = _GEOM_TYPE[finger_geom]
        mg.size = _finger_size_arr(finger_geom, fs)
        mg.friction = friction
        mg.solimp = solimp
        mg.solref = solref
        mg.rgba = [0.3, 0.6, 0.9, 1.0]

        # ── Actuator on slide joint ───────────────────────────────────────────
        act = spec.add_actuator()
        act.name = "grasp_actuator"
        act.target = "finger_slide"
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm = np.array([1.0] + [0.0] * 9)
        act.biastype = mujoco.mjtBias.mjBIAS_NONE

        # ── Object (freejoint) ────────────────────────────────────────────────
        ob = spec.worldbody.add_body()
        ob.name = "object"
        ob.pos = [0.0, 0.0, height]
        fj = ob.add_joint()
        fj.type = mujoco.mjtJoint.mjJNT_FREE
        og = ob.add_geom()
        og.type = _GEOM_TYPE[object_geom]
        og.size = _object_size_arr(object_geom, os_)
        og.mass = float(params["object_mass"])
        og.friction = friction
        og.solimp = solimp
        og.solref = solref
        og.rgba = [0.9, 0.5, 0.2, 1.0]

        return spec

    def apply_ctrl(
        self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]
    ) -> None:
        data.ctrl[0] = float(params["grasp_force"])

    def extract_series(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
    ) -> dict[str, float]:
        # DOF layout:
        #   qpos[0]   = finger_slide displacement
        #   qpos[1:8] = object freejoint [x, y, z, qw, qx, qy, qz]
        #   qvel[0]   = finger_slide velocity
        #   qvel[1:7] = object freejoint velocities [vx, vy, vz, wx, wy, wz]
        z_pos = float(data.qpos[3])  # object z
        z_vel = float(data.qvel[3])  # object vz

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
