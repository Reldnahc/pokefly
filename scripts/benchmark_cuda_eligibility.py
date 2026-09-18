"""Check CUDA eligibility against the serial rule, including transfer overhead.

No game, weights, model parameters or production dispatch are changed. The
benchmark runs alongside active research jobs, not on an isolated GPU.
"""

import argparse
import time

import numpy as np

from pokefly.cuda_plasticity import CUDAEligibility
from pokefly.fast_plasticity import sensorimotor_eligibility
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=int, default=4300611)
    args = parser.parse_args()
    if not 1 <= args.edges <= 30_000_000:
        parser.error("Edge count must be within the simulator's practical range")
    configure_runtime()
    import cupy as cp

    output = run_directory("cuda-eligibility-benchmark")
    report = {"scope": __doc__, "rows": []}
    rng = np.random.default_rng(9393)
    n, edges = 166700, args.edges
    pre, post = [rng.integers(0, n, edges) for _ in range(2)]
    trace = rng.normal(0, 0.1, n).astype(np.float32)
    baseline = rng.uniform(0, 0.1, n).astype(np.float32)
    spike = (rng.random(n) < 0.024).astype(np.float32)
    initial = rng.normal(0, 4, edges).astype(np.float32)
    cuda = CUDAEligibility(pre, post, cp)
    for clipped, dual in ((False, False), (True, False), (False, True), (True, True)):
        results = []
        for label, kernel in (("serial", sensorimotor_eligibility), ("cuda", cuda)):
            values, slow = initial.copy(), initial.copy()
            options = {"clip_eligibility": clipped, "incoming_floor": -5.0}
            if dual:
                options.update(slow_eligibility=slow, slow_decay=np.exp(-0.02 / 30))
            args = (pre, post, trace, baseline, spike, values, 0.02, np.exp(-0.02 / 0.6))
            kernel(*args, **options)
            values[:], slow[:] = initial, initial
            start = time.perf_counter()
            for _ in range(8):
                kernel(*args, **options)
            elapsed = time.perf_counter() - start
            results.append((values, slow))
            row = {
                "implementation": label,
                "clip": clipped,
                "dual": dual,
                "edges": edges,
                "milliseconds_per_step": 1000 * elapsed / 8,
            }
            report["rows"].append(row)
            print(row, flush=True)
        for index in range(2):
            np.testing.assert_array_equal(results[0][index], results[1][index])
        report["rows"][-1]["all_edges_bit_exact"] = True
        write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
