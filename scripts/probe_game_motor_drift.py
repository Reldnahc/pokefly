"""Neutral-screen motor-rate audit of final ACTUAL-game synapses.

Original and retained conditions share fixed calibration, raw gray pixels,
decoder and reset noise. No rewards, fitted targets, learning, game actions or
neural-state export. This tests tonic drift, not gameplay skill or intention.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from train_game_series import completed_game_source

from pokefly.actions import count_buttons
from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig
from pokefly.internal_brain import InternalBrain
from pokefly.runner import run_directory, write_json


def probe_rates(c, frame, seed, *, warmup=128, decisions=128):
    weights = c.plasticity.weights.copy()
    c.reset_dynamics(seed)
    for _ in range(warmup):
        c.observe(frame)
    counts = np.zeros(c.brain.n, np.int64)
    actions = Counter()
    for _ in range(decisions):
        observed = c.observe(frame)
        counts += observed.counts
        actions[c.choose(observed)[0]] += 1
    np.testing.assert_array_equal(weights, c.plasticity.weights)
    seconds = decisions * c.brain_steps * c.brain.dt
    return {
        "noise_seed": seed, "scored_neural_seconds": seconds,
        "motor_hz": {name: float(counts[indices].mean() / seconds)
                     for name, indices in c.motors.items()},
        "motor_counts": {name: counts[indices].tolist() for name, indices in c.motors.items()},
        "actions": dict(actions), "button_counts": count_buttons(actions),
        "weights_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3201, 3202, 3203])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3:
        parser.error("At least three distinct noise seeds required")
    sources = [completed_game_source(path) for path in args.source_runs]
    if len({str(source[0]) for source in sources}) != len(sources):
        parser.error("Duplicate source checkpoint")
    if any(source[1] != sources[0][1] for source in sources):
        parser.error("Source brains must share the same model settings")
    if any(source[2]["source_launch_seed"] in args.seeds for source in sources):
        parser.error("Use held-out noise, not the game's training seed")
    c = InternalBrain(device="cuda", config=ExperimentConfig.from_dict(sources[0][1]).brain)
    original, original_state = c.snapshot()
    output = run_directory("game-motor-drift-probe")
    report = {
        "scope": __doc__, "sources": [s[2] for s in sources], "seeds": args.seeds,
        "warmup_decisions": 128, "scored_decisions": 128, "screen": "uniform RGB128",
        "one_shared_original_control": True, "status": "running", "rows": [],
    }
    write_json(output / "report.json", report)
    frame = np.full((144, 160, 3), 128, np.uint8)
    for name, checkpoint in [("original", None), *[(str(s[0]), s[0]) for s in sources]]:
        if checkpoint is None:
            c.restore(original, original_state, weights_only=True)
        else:
            arrays, saved = read_checkpoint(checkpoint)
            c.restore(arrays, saved["neural"], weights_only=True)
        for seed in args.seeds:
            row = {"condition": name, **probe_rates(c, frame, seed)}
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(name, seed, row["motor_hz"], row["button_counts"], flush=True)
    for checkpoint, _, provenance in sources:
        assert sha256(checkpoint / "brain.npz") == provenance["brain_sha256"]
    report["status"] = "completed"
    report["source_neural_files_unchanged"] = True
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
