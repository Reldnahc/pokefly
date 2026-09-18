"""Post-discovery control: shuffled reversal from the SAME acquired neural state.

Reuses completed paired reversal measurements, explicitly not new trials.
Only this control is run. ROM-free; no diagnostic weights enter Pokemon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_learning_choices import score, test_choices

from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquisition", type=Path, required=True)
    p.add_argument("--reversal", type=Path, required=True)
    args = p.parse_args()
    acquired = json.loads((args.acquisition / "report.json").read_text())
    reversed_report = json.loads((args.reversal / "report.json").read_text())
    start = max(acquired["checkpoints"])
    ends = reversed_report["checkpoints"]
    if (
        reversed_report["previous_report_sha256"] != sha256(args.acquisition / "report.json")
        or reversed_report["starting_step"] != start
        or reversed_report["reverse_mapping"] == acquired["reverse_mapping"]
        or reversed_report["seed"] != acquired["seed"]
    ):
        p.error("Matching acquired source and explicitly reversed continuation required")
    rows = json.loads((args.reversal / "paired-training.json").read_text())
    if len(rows) != max(ends):
        p.error("Paired reversal training must be complete")
    buttons = tuple(reversed_report.get("rewarded_button_pair", ["left", "right"]))
    cues = tuple(reversed_report.get("cue_images", ["left", "right"]))
    rng = np.random.default_rng(acquired["seed"] + 712345)
    schedule = np.array([r["reward"] for r in rows])
    presented = np.array([r["cue"] for r in rows])
    previous = start
    for end in ends:
        for cue in cues:
            indices = np.flatnonzero(presented[previous:end] == cue) + previous
            schedule[indices] = rng.permutation(schedule[indices])
        previous = end
    c = InternalBrain(device="cuda", config=load_config(Path(acquired["config"])).brain)
    source = args.acquisition / f"paired-{start}"
    with np.load(source.with_suffix(".npz"), allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    c.restore(arrays, json.loads(source.with_suffix(".json").read_text()))
    output = run_directory("matched-visual-reversal")
    report = {
        "scope": __doc__,
        "acquisition": str(args.acquisition),
        "reversal": str(args.reversal),
        "starting_arrays_sha256": sha256(source.with_suffix(".npz")),
        "paired_training_sha256": sha256(args.reversal / "paired-training.json"),
        "before": reversed_report["pre_reversal_scores"]["paired"],
        "reused_paired": [r for r in reversed_report["rows"] if r["arm"] == "paired"],
        "rows": [],
    }
    patterns, actions = test_patterns(), []
    for index in range(start, max(ends)):
        action = c.choose(c.observe(patterns[presented[index]]))[0]
        c.reinforce(float(schedule[index]))
        actions.append(
            {
                "cue": str(presented[index]),
                "action": action,
                "shuffled_reward": float(schedule[index]),
            }
        )
        if index + 1 in ends:
            learned, state = c.snapshot()
            retention = test_choices(c, seed=acquired["seed"], cues=cues)
            row = {
                "training_decisions": index + 1,
                "retention": retention,
                "score": score(
                    retention, reversed_report["reverse_mapping"], buttons=buttons, cues=cues
                ),
            }
            report["rows"].append(row)
            write_json(output / "report.json", report)
            np.savez_compressed(output / f"shuffled-{index + 1}.npz", **learned)
            write_json(output / f"shuffled-{index + 1}.json", state)
            print(index + 1, row["score"], flush=True)
            c.restore(learned, state)
    write_json(output / "training.json", actions)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
