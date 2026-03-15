"""MuJoCo viewer worker — runs as a subprocess.

macOS: launched via `mjpython` (gets its own main thread for NSWindow).
Linux: launched via plain `python` (no main-thread restriction).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import mujoco.viewer
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Trajectory .npz file path")
    parser.add_argument("--ctrl", required=True, help="Control .json file path")
    args = parser.parse_args()

    traj = np.load(args.npz, allow_pickle=True)
    model = mujoco.MjModel.from_xml_string(str(traj["xml"]))
    data = mujoco.MjData(model)
    qpos_traj: np.ndarray = traj["qpos"]  # (T, nq)
    qvel_traj: np.ndarray = traj["qvel"]  # (T, nv)
    n_frames = qpos_traj.shape[0]

    frame = 0
    last_ctrl_mtime = 0.0

    def apply_frame(f: int) -> None:
        data.qpos[:] = qpos_traj[f]
        data.qvel[:] = qvel_traj[f]
        mujoco.mj_forward(model, data)

    apply_frame(0)

    with mujoco.viewer.launch_passive(model, data) as handle:
        while handle.is_running():
            # Poll control file for seek/play commands from the GUI
            try:
                mtime = os.path.getmtime(args.ctrl)
                if mtime != last_ctrl_mtime:
                    last_ctrl_mtime = mtime
                    with open(args.ctrl) as f:
                        ctrl = json.load(f)
                    frame = max(0, min(int(ctrl.get("frame", frame)), n_frames - 1))
            except Exception:
                pass

            with handle.lock():
                apply_frame(frame)
            handle.sync()

            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
