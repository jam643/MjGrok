"""Actuated Arm scenario: single-link arm on a hinge joint driven by PD control."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario

_INTEGRATOR_MAP = {
    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
}


class ActuatedArmScenario(Scenario):
    name = "Actuated Arm"
    description = (
        "Single-link arm on a hinge joint driven by a step-input position command. "
        "Explore joint damping, armature, PD gains, gravity effects, and timestep stability."
    )

    @property
    def sim_duration(self) -> float:
        return 5.0

    def param_specs(self) -> list[ParamSpec]:
        return [
            # Link Geometry
            ParamSpec(
                "link_length",
                "Link Length (m)",
                "float",
                0.5,
                min_val=0.1,
                max_val=1.0,
                step=0.05,
                sweepable=True,
                tooltip="Length of the arm link",
                group="Link Geometry",
            ),
            ParamSpec(
                "link_radius",
                "Link Radius (m)",
                "float",
                0.03,
                min_val=0.01,
                max_val=0.1,
                step=0.005,
                sweepable=False,
                tooltip="Radius of the capsule link",
                group="Link Geometry",
            ),
            ParamSpec(
                "link_mass",
                "Link Mass (kg)",
                "float",
                1.0,
                min_val=0.01,
                max_val=5.0,
                step=0.1,
                sweepable=True,
                tooltip="Mass of the arm link",
                group="Link Geometry",
            ),
            ParamSpec(
                "tip_mass",
                "Tip Mass (kg)",
                "float",
                0.0,
                min_val=0.0,
                max_val=2.0,
                step=0.1,
                sweepable=True,
                tooltip="Optional point mass at the tip of the arm (0 = none)",
                group="Link Geometry",
            ),
            # Joint Dynamics
            ParamSpec(
                "damping",
                "Damping",
                "float",
                0.5,
                min_val=0.0,
                max_val=5.0,
                step=0.05,
                sweepable=True,
                tooltip="Joint viscous damping coefficient (N·m·s/rad)",
                group="Joint Dynamics",
            ),
            ParamSpec(
                "armature",
                "Armature",
                "float",
                0.01,
                min_val=0.0,
                max_val=0.2,
                step=0.005,
                sweepable=True,
                tooltip="Rotor inertia added to joint (kg·m²)",
                group="Joint Dynamics",
            ),
            ParamSpec(
                "stiffness",
                "Stiffness",
                "float",
                0.0,
                min_val=0.0,
                max_val=10.0,
                step=0.1,
                sweepable=True,
                tooltip="Joint spring stiffness (N·m/rad) — restores to q=0",
                group="Joint Dynamics",
            ),
            ParamSpec(
                "frictionloss",
                "Friction Loss",
                "float",
                0.0,
                min_val=0.0,
                max_val=1.0,
                step=0.05,
                sweepable=True,
                tooltip="Dry friction torque (N·m) opposing joint motion",
                group="Joint Dynamics",
            ),
            # Actuator
            ParamSpec(
                "actuator_type",
                "Actuator Type",
                "enum",
                "position",
                choices=["position", "velocity", "motor"],
                sweepable=False,
                tooltip=(
                    "position: MuJoCo position servo (kp*(target-q) - kv*qdot); "
                    "velocity: velocity servo (damps toward rest); "
                    "motor: manual PD computed in apply_ctrl"
                ),
                group="Actuator",
            ),
            ParamSpec(
                "kp",
                "kp (position gain)",
                "float",
                50.0,
                min_val=0.1,
                max_val=500.0,
                step=1.0,
                sweepable=True,
                tooltip="Proportional gain for position or velocity actuator",
                group="Actuator",
            ),
            ParamSpec(
                "kv",
                "kv (velocity gain)",
                "float",
                5.0,
                min_val=0.0,
                max_val=50.0,
                step=0.5,
                sweepable=True,
                tooltip="Derivative gain (used by position and motor actuator types)",
                group="Actuator",
            ),
            ParamSpec(
                "target_angle_deg",
                "Target Angle (deg)",
                "float",
                90.0,
                min_val=-150.0,
                max_val=150.0,
                step=5.0,
                sweepable=False,
                tooltip="Step-input target joint angle (0° = hanging down, 90° = horizontal)",
                group="Actuator",
            ),
            # Joint Limits
            ParamSpec(
                "use_limits",
                "Use Joint Limits",
                "enum",
                "on",
                choices=["on", "off"],
                sweepable=False,
                tooltip="Enable joint angle limits",
                group="Joint Limits",
            ),
            ParamSpec(
                "limit_range_deg",
                "Limit Range (deg)",
                "float",
                150.0,
                min_val=10.0,
                max_val=175.0,
                step=5.0,
                sweepable=False,
                tooltip="Symmetric joint limit ±N degrees from q=0",
                group="Joint Limits",
            ),
            ParamSpec(
                "limit_margin_deg",
                "Limit Margin (deg)",
                "float",
                1.0,
                min_val=0.0,
                max_val=10.0,
                step=0.5,
                sweepable=False,
                tooltip="Soft constraint activation margin near the limit (degrees)",
                group="Joint Limits",
            ),
            # Simulation
            ParamSpec(
                "timestep",
                "Timestep (s)",
                "float",
                0.002,
                min_val=0.0001,
                max_val=0.01,
                step=0.0001,
                sweepable=True,
                tooltip="Simulation timestep",
                group="Simulation",
            ),
            ParamSpec(
                "integrator",
                "Integrator",
                "enum",
                "implicit",
                choices=["Euler", "RK4", "implicit"],
                sweepable=False,
                tooltip="Numerical integrator for the equations of motion",
                group="Simulation",
            ),
        ]

    def plot_specs(self) -> list[PlotSpec]:
        return [
            PlotSpec(
                "angle",
                "Joint Angle",
                "time (s)",
                "Angle (deg)",
                ["angle_deg", "target_angle_deg"],
                group="Kinematics",
            ),
            PlotSpec(
                "velocity",
                "Angular Velocity",
                "time (s)",
                "Angular Velocity (deg/s)",
                ["velocity_degs"],
                group="Kinematics",
            ),
            PlotSpec(
                "ctrl_torque",
                "Actuator Torque",
                "time (s)",
                "Torque (N·m)",
                ["ctrl_torque"],
                group="Control",
            ),
            PlotSpec(
                "torque_components",
                "Torque Components",
                "time (s)",
                "Torque (N·m)",
                ["gravity_torque", "ctrl_torque"],
                group="Control",
            ),
            PlotSpec(
                "ke",
                "Kinetic Energy",
                "time (s)",
                "Energy (J)",
                ["ke"],
                group="Energy",
            ),
            PlotSpec(
                "total_energy",
                "Total Energy",
                "time (s)",
                "Energy (J)",
                ["ke", "pe", "total_energy"],
                group="Energy",
            ),
        ]

    def build_spec(self, params: dict[str, Any]) -> mujoco.MjSpec:
        L = float(params["link_length"])
        r = float(params["link_radius"])
        link_mass = float(params["link_mass"])
        tip_mass = float(params["tip_mass"])
        damping = float(params["damping"])
        armature = float(params["armature"])
        stiffness = float(params["stiffness"])
        frictionloss = float(params["frictionloss"])
        actuator_type = params["actuator_type"]
        kp = float(params["kp"])
        kv = float(params["kv"])
        use_limits = params["use_limits"] == "on"
        limit_range_rad = math.radians(float(params["limit_range_deg"]))
        limit_margin_rad = math.radians(float(params["limit_margin_deg"]))

        spec = mujoco.MjSpec()
        spec.option.gravity = [0.0, 0.0, -9.81]
        spec.option.timestep = float(params["timestep"])
        spec.option.integrator = _INTEGRATOR_MAP[params["integrator"]]

        # Arm body at world origin
        arm_body = spec.worldbody.add_body()
        arm_body.name = "arm"
        arm_body.pos = [0.0, 0.0, 0.0]

        # Hinge joint (Y-axis: rotation in the X-Z plane, so gravity acts on the arm)
        hinge = arm_body.add_joint()
        hinge.name = "hinge"
        hinge.type = mujoco.mjtJoint.mjJNT_HINGE
        hinge.axis = [0.0, 1.0, 0.0]
        hinge.pos = [0.0, 0.0, 0.0]
        hinge.damping = damping
        hinge.armature = armature
        hinge.stiffness = stiffness
        hinge.frictionloss = frictionloss
        if use_limits:
            hinge.limited = True
            hinge.range = [-limit_range_rad, limit_range_rad]
            hinge.margin = limit_margin_rad

        # Capsule link: hangs downward at q=0
        # Center at [0, 0, -L/2], size=[radius, half_length]
        link_geom = arm_body.add_geom()
        link_geom.name = "link"
        link_geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        link_geom.pos = [0.0, 0.0, -L / 2.0]
        link_geom.size = [r, L / 2.0, 0.0]
        link_geom.mass = link_mass
        link_geom.rgba = [0.4, 0.6, 0.8, 1.0]

        # Optional tip mass
        if tip_mass > 0.0:
            tip_body = arm_body.add_body()
            tip_body.name = "tip"
            tip_body.pos = [0.0, 0.0, -L]
            tip_geom = tip_body.add_geom()
            tip_geom.name = "tip_geom"
            tip_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            tip_geom.size = [r * 1.5, 0.0, 0.0]
            tip_geom.mass = tip_mass
            tip_geom.rgba = [0.9, 0.5, 0.2, 1.0]

        # Actuator
        act = spec.add_actuator()
        act.name = "arm_actuator"
        act.target = "hinge"
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT

        if actuator_type == "position":
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm = np.array([kp] + [0.0] * 9)
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.biasprm = np.array([0.0, -kp, -kv] + [0.0] * 7)
        elif actuator_type == "velocity":
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm = np.array([kp] + [0.0] * 9)
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.biasprm = np.array([0.0, 0.0, -kp] + [0.0] * 7)
        else:  # motor
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm = np.array([1.0] + [0.0] * 9)
            act.biastype = mujoco.mjtBias.mjBIAS_NONE

        # Cache derived quantities for extract_series
        m_eff = link_mass + tip_mass
        L_eff = (link_mass * L / 2.0 + tip_mass * L) / m_eff if m_eff > 0 else L / 2.0
        I_eff = (1.0 / 3.0) * link_mass * L**2 + tip_mass * L**2 + armature

        self._m_eff = m_eff
        self._L_eff = L_eff
        self._I_eff = I_eff
        self._target_angle_rad = math.radians(float(params["target_angle_deg"]))

        return spec

    def apply_ctrl(
        self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]
    ) -> None:
        actuator_type = params["actuator_type"]
        target_rad = math.radians(float(params["target_angle_deg"]))

        if actuator_type == "position":
            data.ctrl[0] = target_rad
        elif actuator_type == "velocity":
            data.ctrl[0] = 0.0
        else:  # motor: manual PD
            kp = float(params["kp"])
            kv = float(params["kv"])
            joint_id = model.joint("hinge").id
            q = float(data.qpos[model.jnt_qposadr[joint_id]])
            qdot = float(data.qvel[model.jnt_dofadr[joint_id]])
            data.ctrl[0] = kp * (target_rad - q) - kv * qdot

    def extract_series(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
    ) -> dict[str, float]:
        joint_id = model.joint("hinge").id
        q = float(data.qpos[model.jnt_qposadr[joint_id]])
        qdot = float(data.qvel[model.jnt_dofadr[joint_id]])

        angle_deg = math.degrees(q)
        velocity_degs = math.degrees(qdot)
        ctrl_torque = float(data.actuator_force[0])
        gravity_torque = -self._m_eff * 9.81 * self._L_eff * math.sin(q)
        ke = 0.5 * self._I_eff * qdot**2
        pe = self._m_eff * 9.81 * self._L_eff * (1.0 - math.cos(q))
        total_energy = ke + pe

        return {
            "angle_deg": angle_deg,
            "target_angle_deg": math.degrees(self._target_angle_rad),
            "velocity_degs": velocity_degs,
            "ctrl_torque": ctrl_torque,
            "gravity_torque": gravity_torque,
            "ke": ke,
            "pe": pe,
            "total_energy": total_energy,
        }
