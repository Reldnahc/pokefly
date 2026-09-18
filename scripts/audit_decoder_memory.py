"""Read-only fixed-decoder timing audit on previously recorded neural spikes.

No training, new neural trials, classifier fitting, or game interventions.
The seven actual motor populations must all be present in the archived counts.
This diagnoses whether rate integration changes raw cue-specific readout; all
preselected timing alternatives and time bins are reported, not a fitted policy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from evaluate_learning_choices import score

from pokefly.checkpoint import sha256
from pokefly.motors import MOTOR_TYPES, MotorConfig, MotorDecoder
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True)
    args = p.parse_args()
    source = json.loads((args.probe / "report.json").read_text())
    with np.load(configure_runtime() / "brain.npz", allow_pickle=False) as z:
        ids, kinds, sides = z["ids"], z["cell_type"].astype(str), z["side"]
    with np.load(args.probe / "counts.npz", allow_pickle=False) as z:
        records = {key: z[key].copy() for key in z.files}
    lookup = {int(body): index for index, body in enumerate(records["body_ids"])}
    groups = {}
    for button, (kind, side) in MOTOR_TYPES.items():
        selected = np.flatnonzero((kinds == kind) & ((sides == side) if side else True))
        groups[button] = np.array([lookup[int(body)] for body in ids[selected]])
    rows = []
    for name, config in (
        ("parallel_0.15s", MotorConfig(arbitration="parallel-v2")),
        ("sustained_1s", MotorConfig(arbitration="sustained-v3", direction_trace_seconds=1)),
        ("sustained_3s", MotorConfig(arbitration="sustained-v3", direction_trace_seconds=3)),
    ):
        actions = []
        for seed in source["seeds"]:
            for condition in ("left", "right", "switching"):
                decoder = MotorDecoder(groups, config)
                for index, counts in enumerate(records[f"{seed}_{condition}"]):
                    cue = (
                        ("left", "right")[(index // 64) % 2]
                        if condition == "switching"
                        else condition
                    )
                    action = decoder.choose(counts, source["neural_seconds_per_decision"])[0]
                    actions.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "index": index,
                            "cue": cue,
                            "action": action,
                        }
                    )

        def scored(selected):
            return score(
                [
                    {
                        "cue": cue,
                        "actions": dict(Counter(r["action"] for r in selected if r["cue"] == cue)),
                    }
                    for cue in ("left", "right")
                ],
                False,
            )

        row = {
            "decoder": name,
            "constant_all": scored([r for r in actions if r["condition"] != "switching"]),
            "constant_late_64_128": scored(
                [r for r in actions if r["condition"] != "switching" and r["index"] >= 64]
            ),
            "switch_time_bins": [
                {
                    "start": lo,
                    "end": hi,
                    **scored(
                        [
                            r
                            for r in actions
                            if r["condition"] == "switching"
                            and r["index"] >= 64
                            and lo <= r["index"] % 64 < hi
                        ]
                    ),
                }
                for lo, hi in ((0, 8), (8, 16), (16, 32), (32, 64))
            ],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = run_directory("decoder-memory-audit")
    write_json(
        output / "report.json",
        {
            "scope": __doc__,
            "source_probe": str(args.probe),
            "source_counts_sha256": sha256(args.probe / "counts.npz"),
            "seeds": source["seeds"],
            "rows": rows,
        },
    )
    print("Report:", output)


if __name__ == "__main__":
    main()
