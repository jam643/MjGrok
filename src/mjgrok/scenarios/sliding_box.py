"""Sliding box scenario: box on floor under friction + constant external force."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mjgrok.scenarios.base import ParamSpec, PlotSpec, Scenario

XML_TEMPLATE = """\
<mujoco model="sliding_box">
  <option timestep="0.002" cone="{cone}" noslip_iterations="{noslip_iterations}"
          impratio="{impratio}"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"
          friction="{friction_slide} {friction_spin} {friction_roll}"
          solimp="{solimp_0} {solimp_1} {solimp_2} 0.5 2"
          solref="{solref_0} {solref_1}"
          rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.25 0.25 0.25" mass="1.0"
            friction="{friction_slide} {friction_spin} {friction_roll}"
            solimp="{solimp_0} {solimp_1} {solimp_2} 0.5 2"
            solref="{solref_0} {solref_1}"
            rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


class SlidingBoxScenario(Scenario):
    name = "Sliding Box"
    description = (
        "Box on a flat floor under constant external force. Explore friction and solver parameters."
    )

    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                "friction_slide",
                "Slide Friction",
                "float",
                0.5,
                min_val=0.0,
                max_val=2.0,
                step=0.01,
                tooltip="Sliding friction coefficient (MuJoCo friction[0])",
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
            ),
            ParamSpec(
                "cone",
                "Contact Cone",
                "enum",
                "pyramidal",
                choices=["pyramidal", "elliptic"],
                sweepable=False,
                tooltip="Contact friction cone type",
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
            ),
        ]

    def plot_specs(self) -> list[PlotSpec]:
        return [
            PlotSpec("pos_x", "Position X", "time (s)", "x (m)", ["pos_x"]),
            PlotSpec("vel_x", "Velocity X", "time (s)", "vel (m/s)", ["vel_x"]),
            PlotSpec("fn", "Normal Contact Force", "time (s)", "Fn (N)", ["fn"]),
            PlotSpec("ft", "Tangential Friction Force", "time (s)", "Ft (N)", ["ft"]),
        ]

    def build_model(self, params: dict[str, Any]) -> mujoco.MjModel:
        xml = XML_TEMPLATE.format(**params)
        return mujoco.MjModel.from_xml_string(xml)

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

        return {
            "pos_x": pos_x,
            "vel_x": vel_x,
            "fn": fn,
            "ft": ft,
        }
