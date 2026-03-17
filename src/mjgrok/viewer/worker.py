"""MuJoCo viewer worker — runs as a subprocess.

macOS: launched via `mjpython` (gets its own main thread for NSWindow).
Linux: launched via plain `python` (no main-thread restriction).

Receives a .npz with (scenario_name, params_json, qpos, qvel) and reconstructs
the model by calling scenario.build_model(params) directly — no XML round-trip.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from mjgrok.scenarios import SCENARIO_REGISTRY


def _apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    f: int,
) -> None:
    data.qpos[:] = qpos_traj[f]
    data.qvel[:] = qvel_traj[f]
    mujoco.mj_forward(model, data)


def _load_npz(npz_path: str) -> tuple:
    traj = np.load(npz_path, allow_pickle=True)
    scenario_name = str(traj["scenario_name"])
    params = json.loads(str(traj["params_json"]))
    qpos_traj: np.ndarray = traj["qpos"]  # (T, nq)
    qvel_traj: np.ndarray = traj["qvel"]  # (T, nv)
    scenario = SCENARIO_REGISTRY[scenario_name]
    model = scenario.build_model(params)
    data = mujoco.MjData(model)
    return model, data, qpos_traj, qvel_traj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Trajectory .npz file path")
    parser.add_argument("--ctrl", required=True, help="Control .json file path")
    args = parser.parse_args()

    model, data, qpos_traj, qvel_traj = _load_npz(args.npz)

    frame = 0
    version = 0
    last_ctrl_mtime = 0.0

    # Outer loop: re-enters launch_passive whenever a reload is signalled
    while True:
        n_frames = qpos_traj.shape[0]
        frame = max(0, min(frame, n_frames - 1))

        _apply_frame(model, data, qpos_traj, qvel_traj, frame)

        reload_needed = False
        with mujoco.viewer.launch_passive(model, data) as handle:
            while handle.is_running():
                # Poll control file for seek / reload commands from the GUI
                try:
                    mtime = os.path.getmtime(args.ctrl)
                    if mtime != last_ctrl_mtime:
                        last_ctrl_mtime = mtime
                        with open(args.ctrl) as f:
                            ctrl = json.load(f)
                        new_version = int(ctrl.get("version", 0))
                        if new_version != version:
                            version = new_version
                            reload_needed = True
                            break  # exit inner loop to reload model
                        frame = max(0, min(int(ctrl.get("frame", frame)), n_frames - 1))
                except Exception:
                    pass

                with handle.lock():
                    _apply_frame(model, data, qpos_traj, qvel_traj, frame)
                handle.sync()

                time.sleep(1.0 / 60.0)

        if reload_needed:
            model, data, qpos_traj, qvel_traj = _load_npz(args.npz)
            # Keep frame position so playback resumes where it left off
        else:
            break  # viewer closed by user


if __name__ == "__main__":
    main()
