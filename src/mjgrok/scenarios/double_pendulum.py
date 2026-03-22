"""Double Pendulum scenario: serial 2-link kinematic chain on hinge joints.

Teaches: multi-body kinematics, energy conservation vs. integrator drift,
sensitivity to initial conditions (chaos), and joint damping dissipation.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario

_INTEGRATOR_MAP = {
    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


class DoublePendulumScenario(Scenario):
    name = "Double Pendulum"
    description = (
        "Two-link passive pendulum on hinge joints. "
        "Explore energy conservation, integrator drift, chaos at large angles, "
        "and how joint damping dissipates energy."
    )

    @property
    def sim_duration(self) -> float:
        return 10.0

    def param_specs(self) -> list[ParamSpec]:
        return [
            # ── Link Geometry ────────────────────────────────────────────────
            ParamSpec(
                "link1_length", "Link 1 Length (m)", "float", 0.5,
                min_val=0.1, max_val=1.5, step=0.05, sweepable=False,
                tooltip="Length of the upper pendulum link.",
                group="Link Geometry",
            ),
            ParamSpec(
                "link1_mass", "Link 1 Mass (kg)", "float", 1.0,
                min_val=0.1, max_val=5.0, step=0.1, sweepable=False,
                tooltip="Mass of the upper link.",
                group="Link Geometry",
            ),
            ParamSpec(
                "link2_length", "Link 2 Length (m)", "float", 0.5,
                min_val=0.1, max_val=1.5, step=0.05, sweepable=False,
                tooltip="Length of the lower pendulum link.",
                group="Link Geometry",
            ),
            ParamSpec(
                "link2_mass", "Link 2 Mass (kg)", "float", 1.0,
                min_val=0.1, max_val=5.0, step=0.1, sweepable=False,
                tooltip="Mass of the lower link.",
                group="Link Geometry",
            ),
            ParamSpec(
                "link_radius", "Link Radius (m)", "float", 0.02,
                min_val=0.005, max_val=0.05, step=0.005, sweepable=False,
                tooltip="Capsule radius for both links (visual only).",
                group="Link Geometry",
            ),
            # ── Initial Conditions ───────────────────────────────────────────
            ParamSpec(
                "theta1_deg", "theta1 Initial Angle (deg)", "float", 10.0,
                min_val=-180.0, max_val=180.0, step=1.0, sweepable=True,
                tooltip=(
                    "Initial angle of link 1 from vertical. 0 deg = hanging down. "
                    "Sweep from ~5 deg to ~120 deg to observe the transition from "
                    "regular oscillation to chaotic motion."
                ),
                group="Initial Conditions",
            ),
            ParamSpec(
                "theta2_deg", "theta2 Initial Angle (deg)", "float", 10.0,
                min_val=-180.0, max_val=180.0, step=1.0, sweepable=True,
                tooltip=(
                    "Initial angle of link 2 relative to link 1. "
                    "Sweep alongside theta1 to show exponential divergence of nearby trajectories."
                ),
                group="Initial Conditions",
            ),
            # ── Joint Dynamics ───────────────────────────────────────────────
            ParamSpec(
                "joint_damping", "Joint Damping (N*m*s/rad)", "float", 0.0,
                min_val=0.0, max_val=2.0, step=0.01, sweepable=True,
                tooltip=(
                    "Viscous damping on both hinge joints. "
                    "0 = energy-conserving (Hamiltonian). "
                    "Sweep to observe energy dissipation rate."
                ),
                group="Joint Dynamics",
            ),
            # ── Simulation ───────────────────────────────────────────────────
            ParamSpec(
                "gravity_scale", "Gravity Scale", "float", 1.0,
                min_val=0.0, max_val=3.0, step=0.05, sweepable=True,
                tooltip=(
                    "Multiplier on Earth gravity (9.81 m/s^2). "
                    "Oscillation period scales as 1/sqrt(g). "
                    "Try 0.38 for Mars, 0.17 for Moon."
                ),
                group="Simulation",
            ),
            ParamSpec(
                "timestep", "Timestep (s)", "float", 0.002,
                min_val=0.0001, max_val=0.02, step=0.0001, sweepable=True,
                tooltip=(
                    "Simulation timestep. Sweep to compare energy drift per integrator. "
                    "Euler requires very small dt; RK4 is accurate at dt=0.002; "
                    "implicit/implicitfast are stable but numerically dissipative."
                ),
                group="Simulation",
            ),
            ParamSpec(
                "integrator", "Integrator", "enum", "RK4",
                choices=["Euler", "RK4", "implicit", "implicitfast"],
                sweepable=False,
                tooltip=(
                    "Numerical integrator. "
                    "Euler: 1st-order, energy drifts upward with large dt. "
                    "RK4: 4th-order, near energy-conserving for small dt. "
                    "implicit/implicitfast: unconditionally stable but numerically dissipative."
                ),
                group="Simulation",
            ),
        ]

    def plot_specs(self) -> list[PlotSpec]:
        return [
            # ── Kinematics ───────────────────────────────────────────────────
            PlotSpec("theta1", "Link 1 Angle", "time (s)", "theta1 (deg)",
                     ["theta1_deg"], group="Kinematics"),
            PlotSpec("theta2", "Link 2 Angle", "time (s)", "theta2 (deg)",
                     ["theta2_deg"], group="Kinematics"),
            PlotSpec("dtheta1", "Link 1 Angular Velocity", "time (s)", "dtheta1/dt (deg/s)",
                     ["dtheta1_degs"], group="Kinematics"),
            PlotSpec("dtheta2", "Link 2 Angular Velocity", "time (s)", "dtheta2/dt (deg/s)",
                     ["dtheta2_degs"], group="Kinematics"),
            # ── Energy ───────────────────────────────────────────────────────
            PlotSpec("energy", "Mechanical Energy", "time (s)", "Energy (J)",
                     ["ke", "pe", "total_energy"], group="Energy"),
            PlotSpec("energy_drift", "Energy Drift", "time (s)", "(E - E0) / E0",
                     ["energy_drift"], group="Energy"),
            # ── Phase Portraits ──────────────────────────────────────────────
            PlotSpec("phase1", "Phase Portrait - Link 1", "theta1 (deg)", "dtheta1/dt (deg/s)",
                     ["theta1_deg", "dtheta1_degs"], mode="phase_portrait",
                     group="Phase Portraits"),
            PlotSpec("phase2", "Phase Portrait - Link 2", "theta2 (deg)", "dtheta2/dt (deg/s)",
                     ["theta2_deg", "dtheta2_degs"], mode="phase_portrait",
                     group="Phase Portraits"),
            # ── Tip Trajectory ───────────────────────────────────────────────
            PlotSpec("tip_xy", "Tip Position (X-Z plane)", "x (m)", "z (m)",
                     ["tip_x", "tip_z"], mode="phase_portrait",
                     group="Tip Trajectory"),
        ]

    def build_spec(self, params: dict[str, Any]) -> mujoco.MjSpec:
        L1 = float(params["link1_length"])
        L2 = float(params["link2_length"])
        m1 = float(params["link1_mass"])
        m2 = float(params["link2_mass"])
        r = float(params["link_radius"])
        damping = float(params["joint_damping"])
        gravity_scale = float(params["gravity_scale"])

        spec = mujoco.MjSpec()
        spec.option.gravity = [0.0, 0.0, -9.81 * gravity_scale]
        spec.option.timestep = float(params["timestep"])
        spec.option.integrator = _INTEGRATOR_MAP[params["integrator"]]
        spec.option.disableflags = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

        # ── Link 1 ───────────────────────────────────────────────────────────
        link1 = spec.worldbody.add_body()
        link1.name = "link1"
        link1.pos = [0.0, 0.0, 0.0]

        j1 = link1.add_joint()
        j1.name = "joint1"
        j1.type = mujoco.mjtJoint.mjJNT_HINGE
        j1.axis = [0.0, 1.0, 0.0]
        j1.pos = [0.0, 0.0, 0.0]
        j1.damping = damping

        g1 = link1.add_geom()
        g1.name = "geom1"
        g1.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g1.pos = [0.0, 0.0, -L1 / 2.0]
        g1.size = [r, L1 / 2.0, 0.0]
        g1.mass = m1
        g1.rgba = [0.3, 0.55, 0.85, 1.0]

        # ── Link 2 (pivot at distal tip of link 1) ───────────────────────────
        link2 = link1.add_body()
        link2.name = "link2"
        link2.pos = [0.0, 0.0, -L1]  # local to link1 frame

        j2 = link2.add_joint()
        j2.name = "joint2"
        j2.type = mujoco.mjtJoint.mjJNT_HINGE
        j2.axis = [0.0, 1.0, 0.0]
        j2.pos = [0.0, 0.0, 0.0]
        j2.damping = damping

        g2 = link2.add_geom()
        g2.name = "geom2"
        g2.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g2.pos = [0.0, 0.0, -L2 / 2.0]
        g2.size = [r, L2 / 2.0, 0.0]
        g2.mass = m2
        g2.rgba = [0.9, 0.45, 0.15, 1.0]

        # ── Tip site ─────────────────────────────────────────────────────────
        tip = link2.add_site()
        tip.name = "tip"
        tip.pos = [0.0, 0.0, -L2]
        tip.size = [0.015, 0.0, 0.0]
        tip.rgba = [1.0, 1.0, 0.0, 1.0]

        return spec

    def setup_data(
        self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]
    ) -> None:
        theta1_rad = math.radians(float(params["theta1_deg"]))
        theta2_rad = math.radians(float(params["theta2_deg"]))
        j1_id = model.joint("joint1").id
        j2_id = model.joint("joint2").id
        data.qpos[model.jnt_qposadr[j1_id]] = theta1_rad
        data.qpos[model.jnt_qposadr[j2_id]] = theta2_rad
        self._E0: float | None = None

    def extract_series(
        self, model: mujoco.MjModel, data: mujoco.MjData, t: float
    ) -> dict[str, float]:
        j1_id = model.joint("joint1").id
        j2_id = model.joint("joint2").id
        q1 = float(data.qpos[model.jnt_qposadr[j1_id]])
        q2 = float(data.qpos[model.jnt_qposadr[j2_id]])
        dq1 = float(data.qvel[model.jnt_dofadr[j1_id]])
        dq2 = float(data.qvel[model.jnt_dofadr[j2_id]])

        site_id = model.site("tip").id
        tip_pos = data.site_xpos[site_id]
        tip_x = float(tip_pos[0])
        tip_z = float(tip_pos[2])

        mujoco.mj_energyPos(model, data)
        mujoco.mj_energyVel(model, data)
        pe = float(data.energy[0])
        ke = float(data.energy[1])
        total_energy = ke + pe

        if not hasattr(self, "_E0") or self._E0 is None:
            self._E0 = total_energy if total_energy != 0.0 else 1.0
        energy_drift = (total_energy - self._E0) / abs(self._E0)

        return {
            "theta1_deg": math.degrees(q1),
            "theta2_deg": math.degrees(q2),
            "dtheta1_degs": math.degrees(dq1),
            "dtheta2_degs": math.degrees(dq2),
            "tip_x": tip_x,
            "tip_z": tip_z,
            "ke": ke,
            "pe": pe,
            "total_energy": total_energy,
            "energy_drift": energy_drift,
        }
