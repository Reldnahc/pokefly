"""Read-only sensory latency/decodability diagnostic, never a game policy.

An offline nearest-centroid measurement asks whether neural activity contains
the current cue across independent noise seeds. Its coefficients are NEVER fed
back to the fly, decoder, rewards or game. All neural weights stay frozen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/intrinsic-v1.json"))
    p.add_argument("--seeds", nargs="+", type=int, default=[801, 802, 803, 804])
    args = p.parse_args()
    if len(set(args.seeds)) < 3:
        p.error("At least three distinct noise seeds required")
    output = run_directory("visual-latency-probe")
    c = InternalBrain(device="cuda", config=load_config(args.config).brain)
    b = c.brain
    edges = np.flatnonzero(np.isin(b.indices, np.concatenate(list(c.motors.values()))))
    motor_input = np.unique(np.searchsorted(b.indptr, edges, side="right") - 1)
    groups = {
        name: c.groups[name]
        for name in ("visual_projection", "visual_Kenyon_candidates", "MBONs", "descending")
    }
    groups["motor_inputs"] = motor_input
    cells = np.unique(np.concatenate(list(groups.values())))
    groups = {name: np.searchsorted(cells, idx) for name, idx in groups.items()}
    original_weights = b.weights.copy()
    records = {}
    for seed in args.seeds:
        for condition in ("left", "right", "switching"):
            c.reset_dynamics(seed)
            rows = []
            for index in range(256 if condition == "switching" else 128):
                cue = (
                    ("left", "right")[(index // 64) % 2] if condition == "switching" else condition
                )
                observed = c.observe(test_patterns()[cue])
                rows.append(observed.counts[cells].astype(np.uint16))
            records[f"{seed}_{condition}"] = np.stack(rows)
        print("frozen sensory probe seed", seed, flush=True)
    np.testing.assert_array_equal(b.weights, original_weights)
    np.savez_compressed(output / "counts.npz", **records, body_ids=c.body_ids[cells])
    report = {
        "config": str(args.config),
        "seeds": args.seeds,
        "protocol": "Frozen brain; no game, rewards, weight fitting or action interventions. "
        "Offline leave-one-noise-seed-out nearest-centroid activity measurement only; "
        "never deployed as a policy. Correlated descriptive samples, not a significance test.",
        "neural_seconds_per_decision": c.brain_steps * b.dt,
        "constant_cue_fit_decisions": [64, 128],
        "switch_block_decisions": 64,
        "rows": [],
    }
    windows = ((0, 8), (8, 16), (16, 32), (32, 64))
    for name, indices in groups.items():
        correct = {str(window): [] for window in windows}
        constant = []
        for seed in args.seeds:
            centers = {
                cue: np.stack(
                    [
                        records[f"{other}_{cue}"][64:, indices].mean(axis=0)
                        for other in args.seeds
                        if other != seed
                    ]
                ).mean(axis=0)
                for cue in ("left", "right")
            }
            contrast = centers["left"] - centers["right"]
            midpoint = (centers["left"] + centers["right"]) / 2

            def predicts_left(activity, midpoint=midpoint, contrast=contrast):
                return float((activity - midpoint) @ contrast) > 0

            for cue in ("left", "right"):
                response = records[f"{seed}_{cue}"][64:, indices].mean(axis=0)
                constant.append(predicts_left(response) == (cue == "left"))
            for block in (1, 2, 3):  # Exclude initial onset; test actual cue switches.
                for start, end in windows:
                    response = records[f"{seed}_switching"][
                        block * 64 + start : block * 64 + end, indices
                    ].mean(axis=0)
                    correct[str((start, end))].append(predicts_left(response) == (block % 2 == 0))
        row = {
            "group": name,
            "neurons": len(indices),
            "constant_cue_accuracy": float(np.mean(constant)),
            "constant_samples": len(constant),
            "switch_accuracy_by_decision_window": {
                window: {"accuracy": float(np.mean(values)), "samples": len(values)}
                for window, values in correct.items()
            },
        }
        report["rows"].append(row)
        print(row, flush=True)
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
