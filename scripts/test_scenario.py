"""Headless smoke test: run 200 steps of SlidingBoxScenario, print qpos and forces."""

import sys

sys.path.insert(0, "src")

import mujoco

from mjgrok.scenarios.sliding_box import SlidingBoxScenario
from mjgrok.simulation.trajectory import TrajectoryCache

scenario = SlidingBoxScenario()
params = scenario.default_params()
print("Params:", params)

model = scenario.build_model(params)
data = mujoco.MjData(model)

box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
print(f"Box body id: {box_id}")

cache = TrajectoryCache(params=params)

for step in range(200):
    data.xfrc_applied[box_id][0] = params["force_x"]
    mujoco.mj_step(model, data)
    values = scenario.extract_series(model, data, data.time)
    cache.append(data.time, values)
    if step % 50 == 0:
        px, vx = values["pos_x"], values["vel_x"]
        fn, ft = values["fn"], values["ft"]
        print(f"t={data.time:.3f}  pos_x={px:.4f}  vel_x={vx:.4f}  fn={fn:.4f}  ft={ft:.4f}")

cache.finalize()
print(f"\nFinal frame count: {cache.frame_count()}")
print(f"Final pos_x: {cache.series_arr['pos_x'][-1]:.4f}")
print("Smoke test PASSED")
