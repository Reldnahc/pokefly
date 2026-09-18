"""Numerical/performance diagnostic only; no game, learned policy or weight edit."""

from __future__ import annotations

import time

import numba
import numpy as np

from pokefly.fast_plasticity import parallel_sensorimotor_eligibility, sensorimotor_eligibility
from pokefly.runner import run_directory, write_json


def main():
    output = run_directory("eligibility-thread-benchmark")
    rng = np.random.default_rng(4242)
    n, edges = 166700, 4300611
    pre = rng.integers(0, n, edges)
    post = rng.integers(0, n, edges)
    incoming = rng.normal(0, 0.04, n).astype(np.float32)
    baseline = rng.uniform(0, 0.1, n).astype(np.float32)
    spike = (rng.random(n) < 0.024).astype(np.float32)
    initial = rng.normal(0, 1, edges).astype(np.float32)
    expected = initial.copy()
    args = (pre, post, incoming, baseline, spike)
    options = {"clip_eligibility": False, "incoming_floor": -5.0}
    old_threads = numba.get_num_threads()
    report = {
        "scope": __doc__,
        "edges": edges,
        "neurons": n,
        "repetitions": 8,
        "caveat": "Concurrent research jobs are running; not an isolated machine benchmark",
        "rows": [],
    }
    try:
        for label, threads, kernel in [
            ("serial", 1, sensorimotor_eligibility),
            *[
                ("parallel", k, parallel_sensorimotor_eligibility)
                for k in (2, 4, 8)
                if k <= numba.config.NUMBA_NUM_THREADS
            ],
        ]:
            numba.set_num_threads(threads)
            values = initial.copy()
            kernel(*args, values, 0.02, np.exp(-0.02 / 0.6), **options)  # compile/warmup
            values[:] = initial
            start = time.perf_counter()
            for _ in range(8):
                kernel(*args, values, 0.02, np.exp(-0.02 / 0.6), **options)
            elapsed = time.perf_counter() - start
            if label == "serial":
                expected[:] = values
            else:
                np.testing.assert_array_equal(values, expected)
            row = {
                "implementation": label,
                "threads": threads,
                "milliseconds_per_step": 1000 * elapsed / 8,
                "all_edges_bit_exact": True,
            }
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(row, flush=True)
    finally:
        numba.set_num_threads(old_threads)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
