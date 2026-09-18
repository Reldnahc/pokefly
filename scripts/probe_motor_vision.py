"""Read-only matched-noise visual information reaching anatomical motor inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/intrinsic-v1.json"))
    parser.add_argument("--decisions", type=int, default=256)
    args = parser.parse_args()
    output = run_directory("motor-vision-probe")
    c = InternalBrain(device="cuda", config=load_config(args.config).brain)
    b = c.brain
    count_by_cue = {}
    motor_cells = np.concatenate(list(c.motors.values()))
    motor_edges = np.flatnonzero(np.isin(b.indices, motor_cells))
    input_cells = np.unique(np.searchsorted(b.indptr, motor_edges, side="right") - 1)
    series = {}
    for cue in ("black", "white", "left", "right"):
        c.reset_dynamics(801)
        counts = np.zeros(b.n, np.int64)
        samples = []
        frame = test_patterns()[cue]
        for _ in range(args.decisions):
            observed = c.observe(frame).counts
            counts += observed
            samples.append(observed[input_cells])
        count_by_cue[cue] = counts
        series[cue] = np.stack(samples)
    np.savez_compressed(
        output / "counts.npz",
        **count_by_cue,
        body_ids=c.body_ids,
        input_cells=input_cells,
        **{"series_" + key: value for key, value in series.items()},
    )
    report = {"protocol": "Same neural noise seed 801, no rewards or plastic updates", "rows": []}
    for first, second in (("black", "white"), ("left", "right")):
        a, z = count_by_cue[first], count_by_cue[second]
        delta = np.abs(a - z)
        for name, cells in c.motors.items():
            edges = np.flatnonzero(np.isin(b.indices, cells))
            pre = np.searchsorted(b.indptr, edges, side="right") - 1
            for sign, selected in (("exc", b.weights[edges] > 0), ("inh", b.weights[edges] < 0)):
                source, weight = pre[selected], np.abs(b.weights[edges[selected]])
                change = weight * delta[source]
                total = weight * (a[source] + z[source]) / 2
                positions = np.searchsorted(input_cells, source)
                first_late = series[first][args.decisions // 2 :].sum(axis=0)[positions]
                second_late = series[second][args.decisions // 2 :].sum(axis=0)[positions]
                late_contrast = (weight * np.abs(first_late - second_late)).sum()
                late_mean = (weight * (first_late + second_late) / 2).sum()
                best = np.argsort(change)[-8:][::-1]
                report["rows"].append(
                    {
                        "pair": [first, second],
                        "motor": name,
                        "sign": sign,
                        "weighted_input_contrast": float(change.sum() / max(total.sum(), 1e-12)),
                        "weighted_mean_spikes": float(total.sum()),
                        "last_half_weighted_contrast": float(late_contrast / max(late_mean, 1e-12)),
                        "distinct_presynaptic_cells": int(len(np.unique(source))),
                        "largest_differences": [
                            {
                                "type": str(b.cell_type[source[i]]),
                                "id": int(c.body_ids[source[i]]),
                                "first_spikes": int(a[source[i]]),
                                "second_spikes": int(z[source[i]]),
                                "weighted_difference": float(change[i]),
                            }
                            for i in best
                        ],
                    }
                )
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
