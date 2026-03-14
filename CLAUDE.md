# MjGrok — MuJoCo Interactive Physics Learning Tool

## Purpose

Interactive sandbox for developing deep intuition for MuJoCo's physics solver. Users select canonical physics scenarios, tune/sweep solver parameters, run simulations, and visualize results via DearPyGUI plots + MuJoCo viewer with playback controls.

## Launch Command

```bash
uv run mjpython -m mjgrok
```

**CRITICAL**: Always use `mjpython` (not plain `python`). On macOS, `mujoco.viewer.launch_passive` requires `mjpython` to dispatch OpenGL renders to the main thread. Without it, the viewer silently fails or crashes.

## Architecture Overview

- `scenarios/base.py` — `ParamSpec`, `PlotSpec`, `Scenario` ABC: foundation for all scenarios
- `scenarios/sliding_box.py` — MVP scenario: box sliding on floor under friction + external force
- `simulation/runner.py` — Background simulation thread with cancel support
- `simulation/trajectory.py` — `TrajectoryCache`: stores time-series data per run
- `viewer/playback.py` — `ViewerController`: launch/seek/play/pause MuJoCo viewer
- `gui/app.py` — DearPyGUI main loop and integration hub
- `gui/param_panel.py` — Auto-generated parameter widgets from `ParamSpec`
- `gui/plot_panel.py` — Pre-created plots updated via `dpg.set_value()` (thread-safe)
- `gui/playback_panel.py` — Scrub slider + play/pause/step/viewer controls

## Threading Model

```
Main Thread (mjpython)
  └─ dpg.start_dearpygui()  ←→  GUI callbacks

SimulationRunner Thread (daemon)
  └─ mj_step() loop → TrajectoryCache
     on_done → dpg.set_value() [thread-safe]

ViewerController Thread (daemon, spawned on-demand)
  └─ launch_passive() → handle
     seek/play: acquire handle.lock() → set qpos/qvel → mj_forward → sync()
```

**Thread safety rule**: Only call `dpg.set_value()` from background threads — never create new DPG items from outside the main thread.

## Key MuJoCo Patterns

**External force injection** (NOT via actuators):
```python
data.xfrc_applied[box_body_id][0] = params["force_x"]  # each step before mj_step
```

**Contact force extraction** (NOT raw `contact.pos`):
```python
force_buf = np.zeros(6)
for i in range(data.ncon):
    mujoco.mj_contactForce(model, data, i, force_buf)
    fn += abs(force_buf[0])           # normal
    ft += np.linalg.norm(force_buf[1:3])  # tangential
```

## GLFW Double-Init Safeguard

Both DearPyGUI and MuJoCo use GLFW. Always call `dpg.create_context()` + `dpg.create_viewport()` before spawning any viewer thread. A `time.sleep(0.5)` after viewport creation before first viewer launch is a pragmatic safeguard.

## Adding New Scenarios

1. Subclass `Scenario` in `scenarios/`
2. Implement `param_specs()`, `plot_specs()`, `build_model()`, `extract_series()`
3. Register in `scenarios/__init__.py`
4. The GUI auto-generates parameter widgets and plots from the specs — no GUI code changes needed.
