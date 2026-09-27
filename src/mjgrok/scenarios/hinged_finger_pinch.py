"""Hinged Finger Pinch scenario: two position-controlled hinge fingers pinch an object from above.

Everything lives in the X-Z plane: finger hinges and the object's rotation are about Y, and the
object is restricted to planar motion (slide x, slide z, hinge y), so the grasp cannot escape
out of plane. The object starts resting on a table that retracts on a slide joint partway through
the run, after which only the fingers hold the object.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario

_DOCS = "https://mujoco.readthedocs.io/en/stable/XMLreference.html"
_COMP_DOCS = "https://mujoco.readthedocs.io/en/stable/computation/index.html"

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

_SIDES = ("left", "right")

# Fixed geometry not worth a slider. Depths are the Y half-extents; the scene is planar, so they
# only set how many contact points a face-face contact spreads over.
_FINGER_DEPTH = 0.015
_OBJECT_DEPTH = 0.02
_TABLE_HALF_SIZE = (0.15, 0.05, 0.02)
_TABLE_MASS = 1.0
_TABLE_KP = 2000.0
_FLOOR_Z = -0.2

# Keep a cylinder's axis along Y so it is a disc in the X-Z plane.
_QUAT_Z_TO_Y = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]


def _ramp(t: float, t_start: float, duration: float, start: float, end: float) -> float:
    """Linear ramp from `start` to `end` over [t_start, t_start + duration], held after."""
    if t <= t_start:
        return start
    if duration <= 0.0 or t >= t_start + duration:
        return end
    return start + (end - start) * (t - t_start) / duration


def _deadzone(e: float, width: float) -> float:
    """Zero inside a band of total `width` centered on 0, shifted linear outside it."""
    half = 0.5 * width
    if abs(e) <= half:
        return 0.0
    return e - math.copysign(half, e)


class HingedFingerPinchScenario(Scenario):
    name = "Hinged Finger Pinch"
    description = (
        "Two position-servo hinge fingers ramp closed onto an object resting on a table, then the "
        "table retracts so only the grasp holds it. Explore how servo gain, effort limit, joint "
        "friction, backlash, encoder bias, contact friction/softness, and contact angle (hinge "
        "spacing) decide whether the object slips."
    )

    @property
    def sim_duration(self) -> float:
        return 4.0

    def param_specs(self) -> list[ParamSpec]:
        return [
            # ── Command ──────────────────────────────────────────────────────
            ParamSpec(
                "open_angle_deg",
                "Open Angle (deg)",
                "float",
                -20.0,
                min_val=-60.0,
                max_val=0.0,
                step=1.0,
                sweepable=True,
                tooltip=(
                    "Initial finger angle and ramp start. 0 = finger hanging straight down; "
                    "negative swings the tip outward (open), positive inward (closing)."
                ),
                group="Command",
            ),
            ParamSpec(
                "close_angle_deg",
                "Close Target (deg)",
                "float",
                10.0,
                min_val=-30.0,
                max_val=45.0,
                step=0.5,
                sweepable=True,
                tooltip=(
                    "Setpoint the ramp ends at and holds. Squeeze torque ~ kp * (target - angle "
                    "where the finger stalls on the object), so commanding past contact is what "
                    "produces grip force."
                ),
                group="Command",
            ),
            ParamSpec(
                "ramp_start",
                "Ramp Start (s)",
                "float",
                0.2,
                min_val=0.0,
                max_val=3.0,
                step=0.05,
                sweepable=True,
                tooltip="Time the closing ramp begins",
                group="Command",
            ),
            ParamSpec(
                "ramp_duration",
                "Ramp Duration (s)",
                "float",
                0.5,
                min_val=0.0,
                max_val=3.0,
                step=0.05,
                sweepable=True,
                tooltip="Time to go from open to close target; 0 = step command",
                group="Command",
            ),
            # ── Finger Actuator ──────────────────────────────────────────────
            ParamSpec(
                "actuator_mode",
                "Actuator Mode",
                "enum",
                "position servo",
                choices=["position servo", "motor PD + backlash"],
                sweepable=False,
                tooltip=(
                    "position servo: MuJoCo built-in position actuator, kp*(ctrl-q) - kv*qdot "
                    "(kv integrated implicitly by implicit/implicitfast). "
                    "motor PD + backlash: explicit PD torque computed each step with a dead-zone "
                    "of `Backlash` on the position error, a proxy for gear play."
                ),
                group="Finger Actuator",
                doc_url=f"{_DOCS}#actuator-position",
            ),
            ParamSpec(
                "kp",
                "kp (Nm/rad)",
                "float",
                5.0,
                min_val=0.1,
                max_val=50.0,
                step=0.1,
                sweepable=True,
                tooltip="Finger servo position gain",
                group="Finger Actuator",
                doc_url=f"{_DOCS}#actuator-position-kp",
            ),
            ParamSpec(
                "kv",
                "kv (Nm*s/rad)",
                "float",
                0.12,
                min_val=0.0,
                max_val=2.0,
                step=0.01,
                sweepable=True,
                tooltip="Finger servo velocity (damping) gain",
                group="Finger Actuator",
                doc_url=f"{_DOCS}#actuator-position-kv",
            ),
            ParamSpec(
                "effort_limit",
                "Effort Limit (Nm)",
                "float",
                1.15,
                min_val=0.0,
                max_val=10.0,
                step=0.05,
                sweepable=True,
                tooltip=(
                    "Actuator torque clip (forcerange); 0 = unlimited. Past kp * error = limit, "
                    "commanding a deeper closure adds no squeeze."
                ),
                group="Finger Actuator",
                doc_url=f"{_DOCS}#actuator-general-forcerange",
            ),
            ParamSpec(
                "backlash_deg",
                "Backlash (deg)",
                "float",
                0.0,
                min_val=0.0,
                max_val=10.0,
                step=0.25,
                sweepable=True,
                tooltip=(
                    "Total dead-zone width on the position error (motor PD mode only). No torque "
                    "until the error exceeds half of this, so the stall torque drops by "
                    "kp * backlash / 2."
                ),
                group="Finger Actuator",
            ),
            ParamSpec(
                "encoder_bias_deg",
                "Encoder Bias (deg)",
                "float",
                0.0,
                min_val=-10.0,
                max_val=10.0,
                step=0.25,
                sweepable=True,
                tooltip=(
                    "Calibration offset: measured = true + bias, and the servo is sent "
                    "target - bias. In free space the measured angle still tracks the target, but "
                    "the true finger sits `bias` short of it, so the squeeze changes."
                ),
                group="Finger Actuator",
            ),
            # ── Finger Joint ─────────────────────────────────────────────────
            ParamSpec(
                "frictionloss",
                "Friction Loss (Nm)",
                "float",
                0.0,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                sweepable=True,
                tooltip=(
                    "Dry (Coulomb) joint friction, solved as a soft constraint. While moving it "
                    "adds a lag of about frictionloss / kp; once stopped the regularized "
                    "constraint lets the joint creep, so the error decays (see friction solref)."
                ),
                group="Finger Joint",
                doc_url=f"{_DOCS}#body-joint-frictionloss",
            ),
            ParamSpec(
                "friction_solref_0",
                "Friction-Loss solref timeconst (s)",
                "float",
                0.02,
                min_val=0.001,
                max_val=0.2,
                step=0.001,
                sweepable=True,
                tooltip=(
                    "solreffriction time constant of the joint friction-loss constraint. MuJoCo's "
                    "dry friction is regularized, so under a steady sub-threshold torque the "
                    "joint creeps and the steady-state error bleeds away; a smaller timeconst "
                    "(clamped to >= 2 * timestep) slows the creep."
                ),
                group="Finger Joint",
                doc_url=f"{_DOCS}#body-joint-solreffriction",
            ),
            ParamSpec(
                "damping",
                "Damping (Nm*s/rad)",
                "float",
                0.008,
                min_val=0.0,
                max_val=1.0,
                step=0.002,
                sweepable=True,
                tooltip="Passive viscous joint damping (adds to the servo kv)",
                group="Finger Joint",
                doc_url=f"{_DOCS}#body-joint-damping",
            ),
            ParamSpec(
                "armature",
                "Armature (kg*m^2)",
                "float",
                0.00044,
                min_val=0.0,
                max_val=0.01,
                step=0.0001,
                sweepable=True,
                tooltip="Reflected rotor inertia; sets servo bandwidth, not static grip",
                group="Finger Joint",
                doc_url=f"{_DOCS}#body-joint-armature",
            ),
            ParamSpec(
                "stiffness",
                "Spring Stiffness (Nm/rad)",
                "float",
                0.0,
                min_val=0.0,
                max_val=5.0,
                step=0.05,
                sweepable=True,
                tooltip=(
                    "Passive return spring toward the open angle (springref). Models cable "
                    "preload or return springs: the error grows with how far the finger closes."
                ),
                group="Finger Joint",
                doc_url=f"{_DOCS}#body-joint-stiffness",
            ),
            # ── Geometry ─────────────────────────────────────────────────────
            ParamSpec(
                "hinge_spacing",
                "Hinge Spacing (m)",
                "float",
                0.052,
                min_val=0.02,
                max_val=0.12,
                step=0.002,
                sweepable=True,
                tooltip=(
                    "Horizontal distance between the finger hinges; sets the contact angle. At "
                    "2 * (object half-width + finger half-thickness) the fingers touch while "
                    "vertical. Wider: fingers meet the object tilted inward (a V that cradles it, "
                    "normals push up). Narrower: tilted outward, normals squeeze it down and out."
                ),
                group="Geometry",
            ),
            ParamSpec(
                "hinge_height",
                "Hinge Height (m)",
                "float",
                0.09,
                min_val=0.03,
                max_val=0.2,
                step=0.005,
                sweepable=True,
                tooltip="Hinge height above the tabletop surface",
                group="Geometry",
            ),
            ParamSpec(
                "finger_length",
                "Finger Length (m)",
                "float",
                0.08,
                min_val=0.02,
                max_val=0.2,
                step=0.005,
                sweepable=True,
                tooltip="Length of each box finger, hinge to tip",
                group="Geometry",
            ),
            ParamSpec(
                "finger_thickness",
                "Finger Half-Thickness (m)",
                "float",
                0.006,
                min_val=0.002,
                max_val=0.02,
                step=0.001,
                sweepable=True,
                tooltip="Half-thickness of each box finger in the plane",
                group="Geometry",
            ),
            ParamSpec(
                "finger_mass",
                "Finger Mass (kg)",
                "float",
                0.03,
                min_val=0.001,
                max_val=0.5,
                step=0.005,
                sweepable=True,
                tooltip="Mass of each finger link",
                group="Geometry",
            ),
            ParamSpec(
                "object_geom",
                "Object Geom",
                "enum",
                "box",
                choices=["box", "cylinder", "sphere"],
                sweepable=False,
                tooltip="Object primitive; the cylinder's axis is along Y (a disc in the plane)",
                group="Geometry",
            ),
            ParamSpec(
                "object_half_width",
                "Object Half-Width (m)",
                "float",
                0.02,
                min_val=0.005,
                max_val=0.06,
                step=0.001,
                sweepable=True,
                tooltip="Box half-width in x, or cylinder/sphere radius",
                group="Geometry",
            ),
            ParamSpec(
                "object_half_height",
                "Object Half-Height (m)",
                "float",
                0.02,
                min_val=0.005,
                max_val=0.08,
                step=0.001,
                sweepable=True,
                tooltip="Box half-height in z (ignored for cylinder/sphere)",
                group="Geometry",
            ),
            ParamSpec(
                "object_mass",
                "Object Mass (kg)",
                "float",
                0.15,
                min_val=0.01,
                max_val=2.0,
                step=0.01,
                sweepable=True,
                tooltip="Object mass; its weight is what friction must hold once the table leaves",
                group="Geometry",
            ),
            # ── Table ────────────────────────────────────────────────────────
            ParamSpec(
                "table_retract_time",
                "Retract Start (s)",
                "float",
                1.2,
                min_val=0.0,
                max_val=4.0,
                step=0.05,
                sweepable=True,
                tooltip="Time the table starts moving away",
                group="Table",
            ),
            ParamSpec(
                "table_retract_duration",
                "Retract Duration (s)",
                "float",
                0.2,
                min_val=0.0,
                max_val=2.0,
                step=0.05,
                sweepable=True,
                tooltip="Time the table takes to move away; 0 = as fast as its servo allows",
                group="Table",
            ),
            ParamSpec(
                "table_retract_distance",
                "Retract Distance (m)",
                "float",
                0.08,
                min_val=0.0,
                max_val=0.15,
                step=0.005,
                sweepable=True,
                tooltip=(
                    "down: drop distance. sideways: clearance between the table edge and the "
                    "object once it has slid past. 0 with down = table never leaves."
                ),
                group="Table",
            ),
            ParamSpec(
                "table_retract_direction",
                "Retract Direction",
                "enum",
                "down",
                choices=["down", "sideways"],
                sweepable=False,
                tooltip=(
                    "down: table drops away. sideways: table slides out in +x, dragging the "
                    "object by table friction as it goes."
                ),
                group="Table",
            ),
            ParamSpec(
                "table_friction",
                "Table Friction",
                "float",
                0.8,
                min_val=0.0,
                max_val=2.0,
                step=0.05,
                sweepable=True,
                tooltip=(
                    "Table slide friction. Table and object share priority 0, so their contact "
                    "uses max(table, object) friction."
                ),
                group="Table",
                doc_url=f"{_DOCS}#body-geom-friction",
            ),
            # ── Finger Contact ───────────────────────────────────────────────
            ParamSpec(
                "finger_friction",
                "Finger Slide Friction",
                "float",
                0.6,
                min_val=0.0,
                max_val=3.0,
                step=0.05,
                sweepable=True,
                tooltip=(
                    "Finger pad slide friction. With finger priority 1 this alone sets the "
                    "finger-object friction; with priority 0 it is max'd with the object's."
                ),
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-friction",
            ),
            ParamSpec(
                "finger_torsional",
                "Finger Torsional Friction (m)",
                "float",
                0.005,
                min_val=0.0,
                max_val=0.05,
                step=0.001,
                sweepable=True,
                tooltip=(
                    "Torsional friction (condim >= 4), resisting spin about the contact normal. "
                    "The normal here lies in the plane and spin about it is out of plane, so this "
                    "is mostly inert in this 2D scene."
                ),
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-friction",
            ),
            ParamSpec(
                "finger_condim",
                "Finger condim",
                "enum",
                "3",
                choices=["1", "3", "4", "6"],
                sweepable=False,
                tooltip=(
                    "Contact dimensionality: 1 frictionless, 3 slide friction, 4 + torsional, "
                    "6 + rolling"
                ),
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-condim",
            ),
            ParamSpec(
                "finger_priority",
                "Finger Priority",
                "int",
                1,
                min_val=0,
                max_val=1,
                step=1,
                sweepable=False,
                tooltip=(
                    "1: finger contact params win outright over the object (priority 0). "
                    "0: equal priority, so friction = max of the two, condim = max, and "
                    "solref/solimp are mixed by solmix."
                ),
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-priority",
            ),
            ParamSpec(
                "object_friction",
                "Object Slide Friction",
                "float",
                0.3,
                min_val=0.0,
                max_val=3.0,
                step=0.05,
                sweepable=True,
                tooltip="Object slide friction; only reaches the grasp when finger priority is 0",
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-friction",
            ),
            ParamSpec(
                "solref_0",
                "solref timeconst (s)",
                "float",
                0.005,
                min_val=0.001,
                max_val=0.1,
                step=0.001,
                sweepable=True,
                tooltip=(
                    "Finger contact spring time constant. Larger = softer pad: more penetration "
                    "and finger travel before force builds, like a lower kp in series. Keep >= "
                    "2 * timestep."
                ),
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-solref",
            ),
            ParamSpec(
                "solref_1",
                "solref dampratio",
                "float",
                1.0,
                min_val=0.1,
                max_val=3.0,
                step=0.05,
                sweepable=True,
                tooltip="Finger contact damping ratio",
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-solref",
            ),
            ParamSpec(
                "solimp_0",
                "solimp dmin",
                "float",
                0.9,
                min_val=0.0,
                max_val=0.9999,
                step=0.01,
                sweepable=True,
                tooltip="Finger contact impedance at zero penetration",
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-solimp",
            ),
            ParamSpec(
                "solimp_1",
                "solimp dmax",
                "float",
                0.95,
                min_val=0.0,
                max_val=0.9999,
                step=0.01,
                sweepable=True,
                tooltip="Finger contact impedance once penetration reaches `width`",
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-solimp",
            ),
            ParamSpec(
                "solimp_2",
                "solimp width (m)",
                "float",
                0.001,
                min_val=0.0001,
                max_val=0.02,
                step=0.0001,
                sweepable=True,
                tooltip="Penetration over which impedance ramps from dmin to dmax",
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-solimp",
            ),
            ParamSpec(
                "finger_margin",
                "Finger Margin (m)",
                "float",
                0.0,
                min_val=0.0,
                max_val=0.005,
                step=0.0005,
                sweepable=True,
                tooltip=(
                    "Distance at which finger contacts activate; acts like extra pad thickness"
                ),
                group="Finger Contact",
                doc_url=f"{_DOCS}#body-geom-margin",
            ),
            # ── Simulation ───────────────────────────────────────────────────
            ParamSpec(
                "timestep",
                "Timestep (s)",
                "float",
                0.002,
                min_val=0.0002,
                max_val=0.01,
                step=0.0001,
                sweepable=True,
                tooltip="Simulation timestep",
                group="Simulation",
                doc_url=f"{_DOCS}#option-timestep",
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
                doc_url=f"{_DOCS}#option-integrator",
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
                doc_url=f"{_DOCS}#option-solver",
            ),
            ParamSpec(
                "iterations",
                "Solver Iterations",
                "int",
                100,
                min_val=1,
                max_val=200,
                step=1,
                sweepable=True,
                tooltip="Max main-solver iterations per step",
                group="Simulation",
                doc_url=f"{_DOCS}#option-iterations",
            ),
            ParamSpec(
                "cone",
                "Friction Cone",
                "enum",
                "elliptic",
                choices=["pyramidal", "elliptic"],
                sweepable=False,
                tooltip="Friction cone approximation",
                group="Simulation",
                doc_url=f"{_DOCS}#option-cone",
            ),
            ParamSpec(
                "impratio",
                "Impedance Ratio",
                "float",
                1.0,
                min_val=0.1,
                max_val=100.0,
                step=0.5,
                sweepable=True,
                tooltip=(
                    "Frictional vs normal constraint impedance. Higher = friction resisted more "
                    "stiffly, so a grasp creeps less under constant load (elliptic cone)."
                ),
                group="Simulation",
                doc_url=f"{_DOCS}#option-impratio",
            ),
            ParamSpec(
                "noslip_iterations",
                "No-slip Iterations",
                "int",
                0,
                min_val=0,
                max_val=20,
                step=1,
                sweepable=True,
                tooltip="Extra no-slip post-solve iterations; 0 = disabled. Suppresses creep.",
                group="Simulation",
                doc_url=f"{_DOCS}#option-noslip-iterations",
            ),
        ]

    def plot_specs(self) -> list[PlotSpec]:
        return [
            PlotSpec(
                "finger_angle",
                "Finger Angle: Target vs Measured vs True",
                "time (s)",
                "Angle (deg)",
                ["target_deg", "left_meas_deg", "left_true_deg", "right_true_deg"],
                group="Finger Tracking",
            ),
            PlotSpec(
                "tracking_error",
                "Tracking Error (target - measured)",
                "time (s)",
                "Error (deg)",
                ["left_err_deg", "right_err_deg"],
                group="Finger Tracking",
            ),
            PlotSpec(
                "finger_vel",
                "Finger Angular Velocity",
                "time (s)",
                "Velocity (deg/s)",
                ["left_vel_degs", "right_vel_degs"],
                group="Finger Tracking",
            ),
            PlotSpec(
                "actuator_torque",
                "Actuator Torque vs Effort Limit",
                "time (s)",
                "Torque (Nm)",
                ["left_torque", "right_torque", "effort_limit"],
                group="Finger Tracking",
            ),
            PlotSpec(
                "joint_friction",
                "Joint Friction-Loss Torque",
                "time (s)",
                "Torque (Nm)",
                ["left_frictionloss_torque", "right_frictionloss_torque"],
                group="Finger Tracking",
            ),
            PlotSpec(
                "normal_force",
                "Contact Normal Force",
                "time (s)",
                "Force (N)",
                ["left_fn", "right_fn", "table_fn", "object_weight"],
                group="Contact Forces",
            ),
            PlotSpec(
                "friction_force",
                "Finger Friction (Tangential) Force",
                "time (s)",
                "Force (N)",
                ["left_ft", "right_ft"],
                group="Contact Forces",
            ),
            PlotSpec(
                "friction_util",
                "Friction Utilization |Ft| / (mu * Fn)  (1 = sliding)",
                "time (s)",
                "Utilization (-)",
                ["left_util", "right_util", "util_limit"],
                group="Contact Forces",
            ),
            PlotSpec(
                "slip_speed",
                "Contact Slip Speed (object vs finger, tangential)",
                "time (s)",
                "Speed (m/s)",
                ["left_slip", "right_slip"],
                group="Contact Forces",
            ),
            PlotSpec(
                "object_pos",
                "Object Position",
                "time (s)",
                "Position (m)",
                ["obj_x", "obj_z", "table_pos"],
                group="Object",
            ),
            PlotSpec(
                "object_vel",
                "Object Velocity",
                "time (s)",
                "Velocity (m/s)",
                ["obj_vx", "obj_vz"],
                group="Object",
            ),
            PlotSpec(
                "object_angle",
                "Object Rotation (about Y)",
                "time (s)",
                "Angle (deg)",
                ["obj_theta_deg"],
                group="Object",
            ),
            PlotSpec(
                "max_pen",
                "Max Finger-Object Penetration",
                "time (s)",
                "Penetration (m)",
                ["max_pen"],
                group="Solver",
            ),
            PlotSpec(
                "ncon",
                "Contact Count",
                "time (s)",
                "Contacts",
                ["ncon"],
                group="Solver",
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
        spacing = float(params["hinge_spacing"])
        hinge_h = float(params["hinge_height"])
        length = float(params["finger_length"])
        thick = float(params["finger_thickness"])
        obj_geom = params["object_geom"]
        obj_w = float(params["object_half_width"])
        obj_h = float(params["object_half_height"]) if obj_geom == "box" else obj_w
        effort = float(params["effort_limit"])
        motor_mode = params["actuator_mode"] == "motor PD + backlash"

        spec = mujoco.MjSpec()
        spec.option.gravity = [0.0, 0.0, -9.81]
        spec.option.timestep = float(params["timestep"])
        spec.option.integrator = _INTEGRATOR_MAP[params["integrator"]]
        spec.option.solver = _SOLVER_MAP[params["solver"]]
        spec.option.iterations = int(params["iterations"])
        spec.option.cone = _CONE_MAP[params["cone"]]
        spec.option.impratio = float(params["impratio"])
        spec.option.noslip_iterations = int(params["noslip_iterations"])
        # Side-on view of the X-Z plane.
        spec.visual.global_.azimuth = 90.0
        spec.visual.global_.elevation = -10.0

        # Catches a dropped object so the plots stay readable.
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.pos = [0.0, 0.0, _FLOOR_Z]
        floor.size = [0.5, 0.5, 0.01]
        floor.rgba = [0.85, 0.85, 0.85, 1.0]

        # ── Table on a slide joint, held by a stiff position servo ────────────
        tx, ty, tz = _TABLE_HALF_SIZE
        table = spec.worldbody.add_body()
        table.name = "table"
        table.pos = [0.0, 0.0, -tz]  # tabletop surface at z = 0
        table.gravcomp = 1.0
        tj = table.add_joint()
        tj.name = "table_slide"
        tj.type = mujoco.mjtJoint.mjJNT_SLIDE
        tj.axis = [0.0, 0.0, 1.0] if params["table_retract_direction"] == "down" else [1, 0, 0]
        tg = table.add_geom()
        tg.name = "table_geom"
        tg.type = mujoco.mjtGeom.mjGEOM_BOX
        tg.size = [tx, ty, tz]
        tg.mass = _TABLE_MASS
        tg.friction = [float(params["table_friction"]), 0.005, 0.0001]
        tg.rgba = [0.55, 0.4, 0.3, 1.0]

        # ── Planar object resting on the table ────────────────────────────────
        obj = spec.worldbody.add_body()
        obj.name = "object"
        obj.pos = [0.0, 0.0, obj_h]
        for jname, jtype, axis in (
            ("obj_x", mujoco.mjtJoint.mjJNT_SLIDE, [1.0, 0.0, 0.0]),
            ("obj_z", mujoco.mjtJoint.mjJNT_SLIDE, [0.0, 0.0, 1.0]),
            ("obj_theta", mujoco.mjtJoint.mjJNT_HINGE, [0.0, 1.0, 0.0]),
        ):
            j = obj.add_joint()
            j.name = jname
            j.type = jtype
            j.axis = axis
        og = obj.add_geom()
        og.name = "object_geom"
        if obj_geom == "box":
            og.type = mujoco.mjtGeom.mjGEOM_BOX
            og.size = [obj_w, _OBJECT_DEPTH, obj_h]
        elif obj_geom == "cylinder":
            og.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            og.size = [obj_w, _OBJECT_DEPTH, 0.0]
            og.quat = _QUAT_Z_TO_Y
        else:
            og.type = mujoco.mjtGeom.mjGEOM_SPHERE
            og.size = [obj_w, 0.0, 0.0]
        og.mass = float(params["object_mass"])
        og.friction = [float(params["object_friction"]), 0.005, 0.0001]
        og.condim = 3
        og.rgba = [0.9, 0.5, 0.2, 1.0]

        # ── Fingers: hinge at the top, box link hanging down along -z at q = 0 ──
        # Left hinge axis is -y and right is +y, so positive q closes both tips inward.
        solref = [float(params["solref_0"]), float(params["solref_1"])]
        solimp = [
            float(params["solimp_0"]),
            float(params["solimp_1"]),
            float(params["solimp_2"]),
            0.5,
            2.0,
        ]
        finger_friction = [
            float(params["finger_friction"]),
            float(params["finger_torsional"]),
            0.0001,
        ]
        for side, sign in zip(_SIDES, (-1.0, 1.0), strict=True):
            fb = spec.worldbody.add_body()
            fb.name = f"{side}_finger"
            fb.pos = [sign * spacing / 2.0, 0.0, hinge_h]
            fj = fb.add_joint()
            fj.name = f"{side}_hinge"
            fj.type = mujoco.mjtJoint.mjJNT_HINGE
            fj.axis = [0.0, sign, 0.0]
            fj.damping = float(params["damping"])
            fj.armature = float(params["armature"])
            fj.frictionloss = float(params["frictionloss"])
            fj.solref_friction = [float(params["friction_solref_0"]), 1.0]
            fj.stiffness = float(params["stiffness"])
            # MjSpec follows the XML convention: hinge range/springref are in degrees.
            fj.springref = float(params["open_angle_deg"])
            fj.limited = True
            fj.range = [-90.0, 90.0]
            fg = fb.add_geom()
            fg.name = f"{side}_finger_geom"
            fg.type = mujoco.mjtGeom.mjGEOM_BOX
            fg.pos = [0.0, 0.0, -length / 2.0]
            fg.size = [thick, _FINGER_DEPTH, length / 2.0]
            fg.mass = float(params["finger_mass"])
            fg.friction = finger_friction
            fg.condim = int(params["finger_condim"])
            fg.priority = int(params["finger_priority"])
            fg.solref = solref
            fg.solimp = solimp
            fg.margin = float(params["finger_margin"])
            fg.rgba = [0.3, 0.6, 0.9, 1.0]

            act = spec.add_actuator()
            act.name = f"{side}_actuator"
            act.target = f"{side}_hinge"
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            if effort > 0.0:
                act.forcelimited = True
                act.forcerange = np.array([-effort, effort])
            if motor_mode:
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                act.gainprm = np.array([1.0] + [0.0] * 9)
                act.biastype = mujoco.mjtBias.mjBIAS_NONE
            else:
                kp, kv = float(params["kp"]), float(params["kv"])
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                act.gainprm = np.array([kp] + [0.0] * 9)
                act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                act.biasprm = np.array([0.0, -kp, -kv] + [0.0] * 7)

        # Fingers only interact with the object.
        for side in _SIDES:
            spec.add_exclude(bodyname1=f"{side}_finger", bodyname2="table")
        spec.add_exclude(bodyname1="left_finger", bodyname2="right_finger")

        # Critically damped table servo.
        table_act = spec.add_actuator()
        table_act.name = "table_actuator"
        table_act.target = "table_slide"
        table_act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        table_act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        table_act.gainprm = np.array([_TABLE_KP] + [0.0] * 9)
        table_act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        table_act.biasprm = np.array(
            [0.0, -_TABLE_KP, -2.0 * math.sqrt(_TABLE_KP * _TABLE_MASS)] + [0.0] * 7
        )

        # Cached for extract_series, which only receives (model, data, t).
        self._params = dict(params)
        return spec

    def setup_data(
        self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]
    ) -> None:
        q_open = math.radians(float(params["open_angle_deg"]))
        for side in _SIDES:
            data.qpos[model.jnt_qposadr[model.joint(f"{side}_hinge").id]] = q_open

    def _finger_target(self, t: float, params: dict[str, Any]) -> float:
        """Commanded finger angle (rad) at time `t`."""
        return math.radians(
            _ramp(
                t,
                float(params["ramp_start"]),
                float(params["ramp_duration"]),
                float(params["open_angle_deg"]),
                float(params["close_angle_deg"]),
            )
        )

    def _table_target(self, t: float, params: dict[str, Any]) -> float:
        """Commanded table slide position (m) at time `t`."""
        dist = float(params["table_retract_distance"])
        if params["table_retract_direction"] == "down":
            end = -dist
        else:
            # Slide the table's -x edge `dist` past the object's +x side so it fully clears.
            end = _TABLE_HALF_SIZE[0] + float(params["object_half_width"]) + dist
        return _ramp(
            t,
            float(params["table_retract_time"]),
            float(params["table_retract_duration"]),
            0.0,
            end,
        )

    def apply_ctrl(
        self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]
    ) -> None:
        t = float(data.time)
        # The servo sees `target - bias`, so the true finger settles `bias` away from the target.
        servo_target = self._finger_target(t, params) - math.radians(
            float(params["encoder_bias_deg"])
        )
        motor_mode = params["actuator_mode"] == "motor PD + backlash"
        kp, kv = float(params["kp"]), float(params["kv"])
        backlash = math.radians(float(params["backlash_deg"]))
        for i, side in enumerate(_SIDES):
            if motor_mode:
                jid = model.joint(f"{side}_hinge").id
                q = float(data.qpos[model.jnt_qposadr[jid]])
                qdot = float(data.qvel[model.jnt_dofadr[jid]])
                # forcerange clips this to the effort limit.
                data.ctrl[i] = kp * _deadzone(servo_target - q, backlash) - kv * qdot
            else:
                data.ctrl[i] = servo_target
        data.ctrl[2] = self._table_target(t, params)

    def _finger_contact_stats(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        finger_geom: int,
        object_geom: int,
        object_body: int,
        finger_body: int,
    ) -> tuple[float, float, float, float, float]:
        """Aggregate the contacts between one finger and the object.

        Returns:
            (normal force sum, |tangential force sum|, effective mu, normal-force-weighted
            tangential slip speed, max penetration).
        """
        fn_sum = 0.0
        ft_world = np.zeros(3)
        slip_weighted = 0.0
        mu = 0.0
        max_pen = 0.0
        force = np.zeros(6)
        jacp_o = np.zeros((3, model.nv))
        jacp_f = np.zeros((3, model.nv))
        for i in range(data.ncon):
            con = data.contact[i]
            if {con.geom1, con.geom2} != {finger_geom, object_geom}:
                continue
            mujoco.mj_contactForce(model, data, i, force)
            # Rows of `frame` are the contact normal then the two tangents, in world coordinates.
            frame = con.frame.reshape(3, 3)
            fn = float(force[0])
            fn_sum += fn
            ft_world += force[1] * frame[1] + force[2] * frame[2]
            mu = float(con.friction[0])
            max_pen = max(max_pen, -float(con.dist))

            mujoco.mj_jac(model, data, jacp_o, None, con.pos, object_body)
            mujoco.mj_jac(model, data, jacp_f, None, con.pos, finger_body)
            v_rel = (jacp_o - jacp_f) @ data.qvel
            v_t = v_rel - np.dot(v_rel, frame[0]) * frame[0]
            slip_weighted += fn * float(np.linalg.norm(v_t))

        ft = float(np.linalg.norm(ft_world))
        slip = slip_weighted / fn_sum if fn_sum > 1e-9 else 0.0
        return fn_sum, ft, mu, slip, max_pen

    def extract_series(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
    ) -> dict[str, float]:
        params = self._params
        bias = math.radians(float(params["encoder_bias_deg"]))
        target = self._finger_target(t, params)
        object_geom = model.geom("object_geom").id
        object_body = model.body("object").id

        out: dict[str, float] = {"target_deg": math.degrees(target)}
        max_pen = 0.0
        for i, side in enumerate(_SIDES):
            jid = model.joint(f"{side}_hinge").id
            dof = model.jnt_dofadr[jid]
            q = float(data.qpos[model.jnt_qposadr[jid]])
            meas = q + bias
            out[f"{side}_true_deg"] = math.degrees(q)
            out[f"{side}_meas_deg"] = math.degrees(meas)
            out[f"{side}_err_deg"] = math.degrees(target - meas)
            out[f"{side}_vel_degs"] = math.degrees(float(data.qvel[dof]))
            out[f"{side}_torque"] = float(data.actuator_force[i])
            # Friction-loss constraint torque on this dof (zero when frictionloss = 0).
            out[f"{side}_frictionloss_torque"] = float(
                sum(
                    data.efc_force[k]
                    for k in range(data.nefc)
                    if data.efc_type[k] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
                    and data.efc_id[k] == dof
                )
            )

            fn, ft, mu, slip, pen = self._finger_contact_stats(
                model,
                data,
                model.geom(f"{side}_finger_geom").id,
                object_geom,
                object_body,
                model.body(f"{side}_finger").id,
            )
            out[f"{side}_fn"] = fn
            out[f"{side}_ft"] = ft
            out[f"{side}_util"] = ft / (mu * fn) if mu > 0.0 and fn > 1e-6 else 0.0
            out[f"{side}_slip"] = slip
            max_pen = max(max_pen, pen)

        table_geom = model.geom("table_geom").id
        table_fn = 0.0
        force = np.zeros(6)
        for i in range(data.ncon):
            con = data.contact[i]
            if {con.geom1, con.geom2} == {table_geom, object_geom}:
                mujoco.mj_contactForce(model, data, i, force)
                table_fn += float(force[0])

        obj_x = float(data.qpos[model.jnt_qposadr[model.joint("obj_x").id]])
        obj_z = float(data.qpos[model.jnt_qposadr[model.joint("obj_z").id]])
        obj_theta = float(data.qpos[model.jnt_qposadr[model.joint("obj_theta").id]])
        table_jid = model.joint("table_slide").id

        out |= {
            "table_fn": table_fn,
            "object_weight": float(params["object_mass"]) * 9.81,
            "effort_limit": float(params["effort_limit"]),
            "util_limit": 1.0,
            # Object joint positions are offsets from its resting pose on the table.
            "obj_x": obj_x,
            "obj_z": obj_z,
            "obj_theta_deg": math.degrees(obj_theta),
            "obj_vx": float(data.qvel[model.jnt_dofadr[model.joint("obj_x").id]]),
            "obj_vz": float(data.qvel[model.jnt_dofadr[model.joint("obj_z").id]]),
            "table_pos": float(data.qpos[model.jnt_qposadr[table_jid]]),
            "max_pen": max_pen,
            "ncon": float(data.ncon),
            "solver_niter": float(data.solver_niter[0]),
        }
        return out
