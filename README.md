# MjGrok

An interactive sandbox for building deep intuition for MuJoCo's physics solver and its parameters.

## What it does

MjGrok hosts a collection of **canonical scenarios** — simple simulations designed to isolate and excite specific MuJoCo solver behaviors (friction, contacts, constraints, penetration, etc.). For each scenario you can:

- Tune solver parameters via an auto-generated GUI
- Run single simulations or sweep across a parameter range
- Overlay multiple trajectories in real-time plots
- Scrub, play, and pause a rendered playback of any run

The goal is to make MuJoCo's solver behavior tangible and explorable.

## Running

```bash
uv run python -m mjgrok
```

> **macOS note:** Use plain Python, not `mjpython`. DearPyGUI requires `NSWindow` on the main thread; `mjpython` conflicts with that.

## Adding a Scenario

1. Subclass `Scenario` in `src/mjgrok/scenarios/`
2. Implement `param_specs()`, `plot_specs()`, `build_model()`, and `extract_series()`
3. Register it in `scenarios/__init__.py`

The GUI adapts automatically — no other changes needed.

## Tech Stack

| Layer | Library |
|---|---|
| GUI & plots | DearPyGUI |
| Physics | MuJoCo (CPU, offscreen renderer) |
| Package management | uv |
| Linting / formatting | ruff |
