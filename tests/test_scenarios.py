"""Scenario tests — runnable with: uv run pytest tests/

These tests verify that each scenario:
  1. Builds a valid MjModel from default params
  2. Runs a full simulation without errors
  3. Produces the expected extract_series keys
  4. Exhibits the correct physics for its key behaviors

Run a single test class:
    uv run pytest tests/test_scenarios.py::TestActuatedArm -v
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from mjgrok.scenarios.actuated_arm import ActuatedArmScenario
from mjgrok.scenarios.penetrating_sphere import PenetratingSphereScenario
from mjgrok.scenarios.sliding_box import SlidingBoxScenario
from mjgrok.scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_scenario(scenario, params, n_steps=None):
    """Build model, run n_steps (default: 1 second worth), return final data."""
    model = scenario.build_model(params)
    data = mujoco.MjData(model)
    scenario.setup_data(model, data, params)
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    steps = n_steps if n_steps is not None else max(1, int(1.0 / dt))

    for _ in range(steps):
        scenario.apply_ctrl(model, data, params)
        mujoco.mj_step(model, data)

    return model, data


def run_to_steady_state(scenario, params):
    """Run for the full sim_duration of the scenario."""
    model = scenario.build_model(params)
    data = mujoco.MjData(model)
    scenario.setup_data(model, data, params)
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    steps = int(scenario.sim_duration / dt)

    for _ in range(steps):
        scenario.apply_ctrl(model, data, params)
        mujoco.mj_step(model, data)

    return model, data


# ---------------------------------------------------------------------------
# Generic smoke tests — every scenario must pass these
# ---------------------------------------------------------------------------

class TestAllScenarios:
    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_build_model_succeeds(self, scenario):
        params = scenario.default_params()
        model = scenario.build_model(params)
        assert isinstance(model, mujoco.MjModel)

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_param_specs_have_required_fields(self, scenario):
        for spec in scenario.param_specs():
            assert spec.name, "ParamSpec must have a name"
            assert spec.label, "ParamSpec must have a label"
            assert spec.dtype in ("float", "int", "enum")
            assert spec.default is not None

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_plot_specs_have_required_fields(self, scenario):
        for spec in scenario.plot_specs():
            assert spec.plot_id
            assert spec.title
            assert spec.series_keys

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_extract_series_returns_all_plot_keys(self, scenario):
        """Every key referenced in plot_specs must appear in extract_series output."""
        params = scenario.default_params()
        model, data = run_scenario(scenario, params, n_steps=1)
        series = scenario.extract_series(model, data, data.time)

        expected_keys = {
            key
            for spec in scenario.plot_specs()
            for key in spec.series_keys
        }
        missing = expected_keys - set(series.keys())
        assert not missing, f"Missing series keys: {missing}"

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_simulation_runs_without_nan(self, scenario):
        """Full simulation should not produce NaN values in extract_series."""
        params = scenario.default_params()
        model = scenario.build_model(params)
        data = mujoco.MjData(model)
        scenario.setup_data(model, data, params)
        mujoco.mj_forward(model, data)

        dt = model.opt.timestep
        steps = int(scenario.sim_duration / dt)

        for _ in range(steps):
            scenario.apply_ctrl(model, data, params)
            mujoco.mj_step(model, data)

        series = scenario.extract_series(model, data, data.time)
        for key, val in series.items():
            assert math.isfinite(val), f"series['{key}'] = {val} is not finite"


# ---------------------------------------------------------------------------
# Actuated Arm
# ---------------------------------------------------------------------------

class TestActuatedArm:
    def setup_method(self):
        self.scenario = ActuatedArmScenario()
        self.params = self.scenario.default_params()

    # ── Model structure ──────────────────────────────────────────────────

    def test_joint_range_is_in_radians_in_compiled_model(self):
        """Joint limits must be compiled as ±150° = ±2.618 rad (not ±2.618°)."""
        params = {**self.params, "limit_range_deg": 150.0, "use_limits": "on"}
        model = self.scenario.build_model(params)
        jnt_id = model.joint("hinge").id
        lo, hi = model.jnt_range[jnt_id]
        expected = math.radians(150.0)
        assert abs(hi - expected) < 0.01, (
            f"Joint upper limit is {math.degrees(hi):.2f}° in model, expected 150°. "
            f"Likely passing radians where MjSpec expects degrees."
        )

    def test_gear_is_set_in_compiled_model(self):
        """Gear ratio must be reflected in the compiled model's actuator_gear."""
        for gear in [1.0, 2.0, 5.0]:
            params = {**self.params, "gear_ratio": gear}
            model = self.scenario.build_model(params)
            act_id = model.actuator("arm_actuator").id
            compiled_gear = model.actuator_gear[act_id, 0]
            assert abs(compiled_gear - gear) < 1e-6, (
                f"Expected gear={gear}, got {compiled_gear}"
            )

    def test_torque_limit_is_set_in_compiled_model(self):
        params = {**self.params, "max_torque": 5.0}
        model = self.scenario.build_model(params)
        act_id = model.actuator("arm_actuator").id
        assert model.actuator_forcelimited[act_id], "Force limit should be enabled"
        lo, hi = model.actuator_forcerange[act_id]
        assert abs(hi - 5.0) < 1e-6 and abs(lo + 5.0) < 1e-6

    def test_no_torque_limit_when_max_torque_zero(self):
        params = {**self.params, "max_torque": 0.0}
        model = self.scenario.build_model(params)
        act_id = model.actuator("arm_actuator").id
        assert not model.actuator_forcelimited[act_id], "Force limit should not be enabled"

    # ── Initial conditions ───────────────────────────────────────────────

    def test_initial_angle_sets_starting_position(self):
        for angle_deg in [0.0, 45.0, -90.0]:
            params = {**self.params, "initial_angle_deg": angle_deg}
            model = self.scenario.build_model(params)
            data = mujoco.MjData(model)
            self.scenario.setup_data(model, data, params)
            mujoco.mj_forward(model, data)
            jnt_id = model.joint("hinge").id
            q0 = data.qpos[model.jnt_qposadr[jnt_id]]
            assert abs(math.degrees(q0) - angle_deg) < 0.01, (
                f"initial_angle_deg={angle_deg}: got q0={math.degrees(q0):.2f}°"
            )

    # ── Physics correctness ──────────────────────────────────────────────

    def test_position_servo_reaches_target(self):
        """With high kp and no torque limit, arm must reach within 3° of target."""
        params = {
            **self.params,
            "kp": 300.0,
            "kv": 20.0,
            "target_angle_deg": 90.0,
            "use_limits": "off",
            "max_torque": 0.0,
            "gear_ratio": 1.0,
        }
        model, data = run_to_steady_state(self.scenario, params)
        jnt_id = model.joint("hinge").id
        q_final = math.degrees(data.qpos[model.jnt_qposadr[jnt_id]])
        assert abs(q_final - 90.0) < 3.0, (
            f"Position servo should reach ~90°, got {q_final:.2f}°. "
            f"Check joint range units or gear scaling in apply_ctrl."
        )

    def test_position_servo_zero_gravity_reaches_target_exactly(self):
        """Without gravity, position servo should reach target within 0.5°."""
        params = {
            **self.params,
            "kp": 200.0,
            "kv": 10.0,
            "target_angle_deg": 90.0,
            "gravity_scale": 0.0,
            "use_limits": "off",
            "max_torque": 0.0,
            "gear_ratio": 1.0,
        }
        model, data = run_to_steady_state(self.scenario, params)
        jnt_id = model.joint("hinge").id
        q_final = math.degrees(data.qpos[model.jnt_qposadr[jnt_id]])
        assert abs(q_final - 90.0) < 0.5, (
            f"Zero-gravity position servo should reach target exactly, got {q_final:.2f}°"
        )

    def test_torque_limit_prevents_reaching_target(self):
        """Very low torque limit (< gravity torque at target) should prevent reaching 90°."""
        # Gravity torque at 90° = m*g*L_eff = 1.0*9.81*0.25 ≈ 2.45 N·m
        # With max_torque=1.0, arm can't hold against gravity at 90°
        params = {
            **self.params,
            "kp": 300.0,
            "kv": 20.0,
            "target_angle_deg": 90.0,
            "max_torque": 1.0,
            "gear_ratio": 1.0,
            "use_limits": "off",
        }
        model, data = run_to_steady_state(self.scenario, params)
        jnt_id = model.joint("hinge").id
        q_final = math.degrees(data.qpos[model.jnt_qposadr[jnt_id]])
        assert q_final < 50.0, (
            f"With max_torque=1 N·m, arm should be blocked far from 90°, got {q_final:.2f}°"
        )

    def test_gear_ratio_multiplies_effective_torque(self):
        """Higher gear means the arm can overcome gravity at a larger angle with same kp."""
        # With torque limit active, gear ratio multiplies effective torque at joint.
        # gear=4 should let the arm reach a larger angle than gear=1 with same raw kp.
        base = {
            **self.params,
            "kp": 10.0,
            "kv": 5.0,
            "target_angle_deg": 90.0,
            "max_torque": 0.0,
            "use_limits": "off",
        }
        _, data1 = run_to_steady_state(self.scenario, {**base, "gear_ratio": 1.0})
        _, data2 = run_to_steady_state(self.scenario, {**base, "gear_ratio": 4.0})

        s = self.scenario
        params1 = {**base, "gear_ratio": 1.0}
        params2 = {**base, "gear_ratio": 4.0}
        m1 = s.build_model(params1)
        m2 = s.build_model(params2)
        jnt_id = m1.joint("hinge").id
        q1 = math.degrees(data1.qpos[m1.jnt_qposadr[jnt_id]])
        q2 = math.degrees(data2.qpos[m2.jnt_qposadr[jnt_id]])
        assert q2 > q1 + 5.0, (
            f"gear=4 should reach a larger angle than gear=1 (got {q2:.1f}° vs {q1:.1f}°)"
        )

    def test_gravity_scale_zero_allows_frictionless_oscillation(self):
        """With gravity=0 and no damping, arm started at 45° should oscillate (KE > 0)."""
        params = {
            **self.params,
            "gravity_scale": 0.0,
            "damping": 0.0,
            "kp": 0.0,
            "kv": 0.0,
            "actuator_type": "motor",
            "initial_angle_deg": 45.0,
            "use_limits": "off",
            "stiffness": 10.0,  # spring restores toward 0 so it oscillates
        }
        model = self.scenario.build_model(params)
        data = mujoco.MjData(model)
        self.scenario.setup_data(model, data, params)
        mujoco.mj_forward(model, data)
        ke_initial = 0.0

        max_ke = 0.0
        steps = int(self.scenario.sim_duration / model.opt.timestep)
        for _ in range(steps):
            self.scenario.apply_ctrl(model, data, params)
            mujoco.mj_step(model, data)
            series = self.scenario.extract_series(model, data, data.time)
            max_ke = max(max_ke, series["ke"])

        assert max_ke > 0.01, (
            f"With stiffness spring + no gravity + initial displacement, KE should be non-zero. "
            f"Got max_ke={max_ke:.4f}"
        )

    def test_zero_damping_causes_overshoot(self):
        """With zero damping and zero velocity gain, the arm should overshoot the target."""
        base = {
            **self.params,
            "kp": 200.0,
            "target_angle_deg": 45.0,
            "use_limits": "off",
            "gear_ratio": 1.0,
            "max_torque": 0.0,
        }

        def max_angle_in_sim(damping, kv, n_steps=500):
            params = {**base, "damping": damping, "kv": kv}
            model = self.scenario.build_model(params)
            data = mujoco.MjData(model)
            self.scenario.setup_data(model, data, params)
            mujoco.mj_forward(model, data)
            jnt_id = model.joint("hinge").id
            max_q = 0.0
            for _ in range(n_steps):
                self.scenario.apply_ctrl(model, data, params)
                mujoco.mj_step(model, data)
                max_q = max(max_q, math.degrees(data.qpos[model.jnt_qposadr[jnt_id]]))
            return max_q

        # No damping → significant overshoot past 45°
        max_undamped = max_angle_in_sim(damping=0.0, kv=0.0)
        # Heavy damping → approaches without overshoot
        max_damped = max_angle_in_sim(damping=5.0, kv=10.0)

        assert max_undamped > 45.0 + 5.0, (
            f"Zero-damping arm should overshoot 45° target, max was {max_undamped:.1f}°"
        )
        assert max_damped < 45.0 + 5.0, (
            f"Heavily-damped arm should not overshoot 45° target, max was {max_damped:.1f}°"
        )


# ---------------------------------------------------------------------------
# Sliding Box
# ---------------------------------------------------------------------------

class TestSlidingBox:
    def setup_method(self):
        self.scenario = SlidingBoxScenario()
        self.params = self.scenario.default_params()

    def test_static_case_box_does_not_move(self):
        """When applied force < friction force, box velocity stays near zero."""
        params = {
            **self.params,
            "friction_slide": 1.0,
            "force_x_normalized": 0.5,  # F < friction → static
        }
        model, data = run_to_steady_state(self.scenario, params)
        vel_x = data.qvel[0]
        assert abs(vel_x) < 0.01, f"Box should be stationary, got vel_x={vel_x:.4f} m/s"

    def test_sliding_case_box_moves(self):
        """When applied force > friction force, box should slide and reach non-zero velocity."""
        params = {
            **self.params,
            "friction_slide": 0.1,
            "force_x_normalized": 1.0,  # F >> friction → slides
        }
        model, data = run_to_steady_state(self.scenario, params)
        vel_x = data.qvel[0]
        assert vel_x > 0.5, f"Box should be sliding, got vel_x={vel_x:.3f} m/s"

    def test_sliding_case_matches_analytical(self):
        """Simulated position should match analytical Coulomb friction solution within 5%."""
        params = {
            **self.params,
            "friction_slide": 0.3,
            "force_x_normalized": 1.0,
            "box_mass": 1.0,
        }
        model, data = run_to_steady_state(self.scenario, params)

        g = 10.0
        mass = float(params["box_mass"])
        mu = float(params["friction_slide"])
        f_norm = float(params["force_x_normalized"])
        N = mass * g
        F_x = f_norm * N
        F_fric = mu * N
        a = (F_x - F_fric) / mass
        t = self.scenario.sim_duration
        x_analytical = 0.5 * a * t**2

        x_sim = data.qpos[0]
        relative_err = abs(x_sim - x_analytical) / max(abs(x_analytical), 1e-3)
        assert relative_err < 0.05, (
            f"Simulated x={x_sim:.3f} m, analytical x={x_analytical:.3f} m "
            f"(relative error={relative_err:.1%})"
        )

    def test_higher_friction_reduces_velocity(self):
        """Doubling friction should result in lower final velocity."""
        def final_vel(mu):
            params = {**self.params, "friction_slide": mu, "force_x_normalized": 2.0}
            _, data = run_to_steady_state(self.scenario, params)
            return data.qvel[0]

        v_low = final_vel(0.1)
        v_high = final_vel(0.5)
        assert v_low > v_high, (
            f"Higher friction should reduce velocity: mu=0.1 → {v_low:.2f}, mu=0.5 → {v_high:.2f}"
        )

    def test_static_box_has_contact_with_floor(self):
        """A static box resting on the floor should have at least one contact point."""
        params = {
            **self.params,
            "friction_slide": 2.0,
            "force_x_normalized": 0.0,  # no applied force → box just rests
        }
        model = self.scenario.build_model(params)
        data = mujoco.MjData(model)
        self.scenario.setup_data(model, data, params)
        mujoco.mj_forward(model, data)

        # Let box settle
        for _ in range(50):
            self.scenario.apply_ctrl(model, data, params)
            mujoco.mj_step(model, data)

        assert data.ncon > 0, "Resting box should be in contact with the floor"


# ---------------------------------------------------------------------------
# Penetrating Sphere
# ---------------------------------------------------------------------------

class TestPenetratingSphere:
    def setup_method(self):
        self.scenario = PenetratingSphereScenario()
        self.params = self.scenario.default_params()

    def test_sphere_is_ejected_from_floor(self):
        """A penetrating sphere should be pushed out of the floor and settle above z=0."""
        params = {**self.params, "penetration_depth": 1.0}  # center at z=0, half inside
        model, data = run_to_steady_state(self.scenario, params)
        z_center = data.qpos[2]
        radius = float(params["sphere_radius"])
        assert z_center >= radius * 0.9, (
            f"Sphere center should be at or above radius ({radius} m), got z={z_center:.4f} m"
        )

    def test_no_penetration_sphere_rests_on_floor(self):
        """Sphere placed at rest on floor (no penetration) should stay near z=radius."""
        params = {**self.params, "penetration_depth": 0.0}
        model, data = run_to_steady_state(self.scenario, params)
        z_center = data.qpos[2]
        radius = float(params["sphere_radius"])
        assert abs(z_center - radius) < radius * 0.1, (
            f"Sphere should rest at z≈{radius} m, got z={z_center:.4f} m"
        )

    def test_longer_solref_timeconst_allows_more_penetration(self):
        """A longer solref time constant means a softer constraint → more residual penetration."""
        def steady_state_z(timeconst):
            params = {
                **self.params,
                "penetration_depth": 0.0,  # start resting on floor
                "solref_timeconst": timeconst,
            }
            model = self.scenario.build_model(params)
            data = mujoco.MjData(model)
            self.scenario.setup_data(model, data, params)
            mujoco.mj_forward(model, data)
            dt = model.opt.timestep
            for _ in range(int(self.scenario.sim_duration / dt)):
                mujoco.mj_step(model, data)
            return data.qpos[2]  # z-position of sphere center

        # Stiffer constraint (short time constant) → sphere sits higher (less penetration)
        # Softer constraint (long time constant) → sphere sinks more
        z_stiff = steady_state_z(0.002)
        z_soft = steady_state_z(0.2)
        assert z_soft < z_stiff, (
            f"Softer constraint (timeconst=0.2) should allow the sphere to sink lower "
            f"than stiff (timeconst=0.002): z_soft={z_soft:.5f}, z_stiff={z_stiff:.5f}"
        )

    def test_normal_force_is_positive_when_in_contact(self):
        """Normal contact force should be positive (upward) when sphere is on floor."""
        params = {**self.params, "penetration_depth": 0.0}
        model = self.scenario.build_model(params)
        data = mujoco.MjData(model)
        self.scenario.setup_data(model, data, params)
        mujoco.mj_forward(model, data)

        # Settle first
        for _ in range(50):
            mujoco.mj_step(model, data)

        series = self.scenario.extract_series(model, data, data.time)
        assert series["fn"] >= 0.0, f"Normal force must be non-negative, got {series['fn']}"
