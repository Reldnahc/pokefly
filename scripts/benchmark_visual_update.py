"""ROM-free exactness/timing check of a visual kernel; no learned weights exported.

Original and fused update branches use the SAME snapshot, stimuli and rewards.
Timing is explicitly collected while other research jobs may share the GPU.
The production model is not monkey-patched outside this diagnostic process.
"""

import time

import numpy as np

from pokefly.cuda_visual import CUDAVisualUpdate
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json


def main():
    from pathlib import Path

    c = InternalBrain(device="cuda", config=load_config(Path("configs/visual-rate-v1.json")).brain)
    v, xp = c.hybrid.visual_circuit, c.brain.xp
    kernel = CUDAVisualUpdate(xp)

    patterns = test_patterns()
    c.observe(patterns["checker"])
    initial, state = c.snapshot()
    output = run_directory("visual-update-benchmark")
    report = {"scope": __doc__, "decisions_per_branch": 64, "rows": []}
    reference, reference_state, reference_choices = None, None, None
    for implementation in ("original", "fused", "fused", "original"):
        c.restore(initial, state)
        v.cuda_update = None if implementation == "original" else kernel
        actions, counts = [], []
        xp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        for index in range(64):
            frame = patterns[("left", "right", "checker", "black")[(index // 8) % 4]]
            observation = c.observe(frame)
            action = c.choose(observation)[0]
            # A fixed synthetic schedule solely for exact learning-state coverage.
            c.reinforce(1.0 if index % 7 == 0 else 0.0)
            actions.append(action)
            counts.append(observation.counts)
        xp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - started
        arrays, metadata = c.snapshot()
        if reference is None:
            reference, reference_state, reference_choices = arrays, metadata, (actions, counts)
        else:
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], reference[key], err_msg=key)
            assert metadata == reference_state
            assert actions == reference_choices[0]
            np.testing.assert_array_equal(counts, reference_choices[1])
        row = {
            "implementation": implementation,
            "milliseconds_per_decision": 1000 * elapsed / 64,
            "all_state_counts_actions_exact": True,
        }
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(row, flush=True)
    c.restore(initial, state)
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
