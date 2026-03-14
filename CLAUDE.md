# MjGrok — MuJoCo Interactive Physics Learning Tool

## Goal

An interactive sandbox for building deep intuition for MuJoCo's physics solver and its parameters (constraints, contacts, friction, penetration, solver impedance, etc.).

The application hosts a collection of simple **canonical scenarios** — each designed to excite a specific set of MuJoCo solver behaviors (e.g. sliding friction, actuated pendulum, elastic collision). A GUI lets the user:

1. Select a scenario
2. Tune or sweep over the relevant MuJoCo parameters for that scenario
3. Run the simulation to generate trajectories
4. Visualize results via scenario-specific plots and an interactive viewer with scrub/play/pause controls

Multiple trajectories from a parameter sweep are overlaid in plots, and a dropdown selects which trajectory to view in the playback viewer.

## Tech Stack

- **GUI + plots**: DearPyGUI — immediate-mode, GPU-accelerated, built-in plotting, no browser overhead, thread-safe `set_value()` from background threads
- **Simulation**: MuJoCo CPU (default), macOS
- **Viewer**: `mujoco.Renderer` (offscreen) → DPG texture displayed in a floating window inside the GUI
- **Package management**: `uv` | **Linting/formatting**: `ruff`

## macOS Main-Thread Constraint

**Launch with plain Python — do NOT use `mjpython`:**

```bash
uv run python -m mjgrok
```

DearPyGUI's `show_viewport()` creates an `NSWindow`, which macOS requires on the main thread. `mjpython` reserves that thread for its own OpenGL dispatch, causing an `NSInternalInconsistencyException` crash. The viewer instead uses `mujoco.Renderer` for offscreen rendering — no GLFW conflict, no separate viewer window process needed.

## Architecture Overview

```
src/mjgrok/
├── scenarios/
│   ├── base.py          # Scenario ABC, ParamSpec, PlotSpec
│   └── sliding_box.py   # MVP scenario
├── simulation/
│   ├── runner.py        # SimulationRunner: background thread, cancel support
│   └── trajectory.py    # TrajectoryCache: stores time-series per run
├── viewer/
│   └── playback.py      # ViewerController: offscreen render → DPG texture
└── gui/
    ├── app.py           # DearPyGUI main loop, integration hub
    ├── param_panel.py   # Auto-generated widgets from ParamSpec (+ sweep UI)
    ├── plot_panel.py    # Pre-created plots, updated via dpg.set_value()
    └── playback_panel.py # Scrub slider, play/pause/step, viewer controls
```

## Threading Model

```
Main Thread
  └─ dpg.start_dearpygui()  ←→  GUI callbacks (on_run, on_seek, on_play)

SimulationRunner Thread (daemon)
  └─ mj_step() loop → TrajectoryCache.append()
     on_done/on_progress → dpg.set_value() [thread-safe]
     Checks cancel_event every step

ViewerController Thread (daemon, spawned on-demand)
  └─ mujoco.Renderer → render frame → dpg.set_value(TEX_TAG, pixels)
     play(): loop rendering frames with frame-rate cap (~60 fps)
```

**Thread safety rule**: Only call `dpg.set_value()` from background threads — never create new DPG items outside the main thread.

## Scenario Interface

Each scenario implements:

```python
class Scenario(ABC):
    name: str
    description: str

    def param_specs(self) -> list[ParamSpec]: ...   # drives param panel widgets
    def plot_specs(self) -> list[PlotSpec]: ...      # drives plot panel layout
    def build_model(self, params) -> mujoco.MjModel: ...
    def extract_series(self, model, data, t) -> dict[str, float]: ...
```

`ParamSpec` includes `dtype` (float/int/enum), default, min/max, `sweepable` flag, and tooltip. The GUI auto-generates all parameter widgets and plots from these specs — no GUI code changes needed when adding a new scenario.

## Sweep Architecture

Sweepable params show a checkbox in the param panel. When checked, the slider is replaced by Min / Max / N inputs. `ParamPanel.get_sweep_configs()` returns the configured ranges. The runner will execute N sequential simulations and the plot panel will overlay all resulting trajectories. A dropdown in the playback panel selects which sweep trajectory to view.

## Adding New Scenarios

1. Subclass `Scenario` in `scenarios/`
2. Implement `param_specs()`, `plot_specs()`, `build_model()`, `extract_series()`
3. Register in `scenarios/__init__.py`

The GUI adapts automatically — no other changes needed.
