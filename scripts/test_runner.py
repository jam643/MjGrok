"""Test SimulationRunner: run 2s, verify cancel works cleanly."""

import sys
import threading

sys.path.insert(0, "src")

from mjgrok.scenarios.sliding_box import SlidingBoxScenario
from mjgrok.simulation.runner import SimulationRunner

scenario = SlidingBoxScenario()
params = scenario.default_params()

done_event = threading.Event()
results = {}


def on_done(cache):
    results["cache"] = cache
    done_event.set()
    final_x = cache.series_arr["pos_x"][-1]
    print(f"[on_done] frames={cache.frame_count()}, final pos_x={final_x:.4f}")


def on_error(e):
    print(f"[on_error] {e}")
    done_event.set()


def on_progress(frac):
    pass  # suppress for cleaner output


runner = SimulationRunner(on_done=on_done, on_error=on_error, on_progress=on_progress)

# Test 1: normal run
print("=== Test 1: Normal 2s run ===")
runner.run(scenario, params, duration=2.0)
done_event.wait(timeout=30)
assert "cache" in results, "Did not receive on_done"
print(f"Test 1 PASSED — frames={results['cache'].frame_count()}")

# Test 2: cancel returns cleanly
print("\n=== Test 2: Cancel returns cleanly ===")
done_event.clear()
results.clear()
runner.run(scenario, params, duration=2.0)
runner.cancel()  # immediate cancel
print("Cancel returned cleanly (no hang)")
print("Test 2 PASSED")

print("\nAll runner tests PASSED")
