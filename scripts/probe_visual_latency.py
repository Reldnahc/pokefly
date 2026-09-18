"""Read-only sensory latency/decodability diagnostic, never a game policy.

An offline nearest-centroid measurement asks whether neural activity contains
the current cue across independent noise seeds. Its coefficients are NEVER fed
back to the fly, decoder, rewards or game. All neural weights stay frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from assay_stimuli import observe_stimulus
from evaluate_learning_choices import score

from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/intrinsic-v1.json"))
    p.add_argument("--seeds", nargs="+", type=int, default=[801, 802, 803, 804])
    p.add_argument(
        "--stimulus", choices=("static-half-v1", "motion-grating-v1"), default="static-half-v1"
    )
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
    # Graded visual cells have no spikes. For this optional model, measure
    # their ACTUAL integrated release instead of mistaking zero spike counts
    # for zero activity. This accumulator only observes the neural simulation.
    graded = None
    if c.hybrid and c.hybrid.visual_circuit is not None:
        graded = c.hybrid.graded_host
        accumulated = b.xp.zeros(len(graded), b.xp.float32)
        original_step = b.step

        def measured_step(*args, **kwargs):
            fired = original_step(*args, **kwargs)
            accumulated[:] += c.hybrid.release[c.hybrid.graded]
            return fired

        b.step = measured_step
    records = {}
    choices = []
    for seed in args.seeds:
        for condition in ("left", "right", "switching"):
            c.reset_dynamics(seed)
            rows = []
            actions = []
            for index in range(256 if condition == "switching" else 128):
                cue = (
                    ("left", "right")[(index // 64) % 2] if condition == "switching" else condition
                )
                if graded is not None:
                    accumulated.fill(0)
                observed = observe_stimulus(c, cue, index, args.stimulus)
                if graded is None:
                    rows.append(observed.counts[cells].astype(np.uint16))
                else:
                    activity = observed.counts.astype(np.float32)
                    activity[graded] = accumulated if b.xp is np else accumulated.get()
                    rows.append(activity[cells])
                actions.append(c.choose(observed)[0])
            records[f"{seed}_{condition}"] = np.stack(rows)
            choices.append({"noise_seed": seed, "cue": condition, "actions": actions})
        print("frozen sensory probe seed", seed, flush=True)
    np.testing.assert_array_equal(b.weights, original_weights)
    np.savez_compressed(output / "counts.npz", **records, body_ids=c.body_ids[cells])
    report = {
        "config": str(args.config),
        "stimulus": args.stimulus,
        "recorded_activity": (
            "spike counts; calibrated graded cells: actual summed transmitter release"
            if graded is not None else "spike counts"
        ),
        "seeds": args.seeds,
        "protocol": "Frozen brain; no game, rewards, weight fitting or action interventions. "
        "Offline leave-one-noise-seed-out nearest-centroid activity measurement only; "
        "never deployed as a policy. Correlated descriptive samples, not a significance test.",
        "neural_seconds_per_decision": c.brain_steps * b.dt,
        "constant_cue_fit_decisions": [64, 128],
        "switch_block_decisions": 64,
        "rows": [],
        "fixed_decoder_records": choices,
    }
    constants = [
        {**row, "actions": dict(Counter(row["actions"][64:]))}
        for row in choices if row["cue"] in ("left", "right")
    ]
    report["fixed_decoder_same_direction_score"] = score(constants, False)
    report["fixed_decoder_opposite_direction_score"] = score(constants, True)
    print("Frozen actual choices", report["fixed_decoder_same_direction_score"], flush=True)
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
