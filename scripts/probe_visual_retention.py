"""Independent frozen visual retention with time-resolved, unfiltered outcomes.

The curve's trained synapses remain confined to this ROM-free assay. Reports
every time bin, not just the best one. No classifier or output mapping is fitted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from evaluate_learning_choices import score

from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curve", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1311, 1312])
    p.add_argument(
        "--baseline-state", type=Path, help="Explicit acquired .npz for reversal baseline"
    )
    p.add_argument("--unpaired-state", type=Path, help="Explicit matched-control .npz")
    args = p.parse_args()
    source = json.loads((args.curve / "report.json").read_text())
    step = max(source["checkpoints"])
    cues = tuple(source.get("cue_images", ["left", "right"]))
    buttons = tuple(source.get("rewarded_button_pair", ["left", "right"]))
    output = run_directory("visual-retention-probe")
    c = InternalBrain(device="cuda", config=load_config(Path(source["config"])).brain)
    original, state = c.snapshot()
    gray = np.full((144, 160, 3), 128, np.uint8)
    patterns = test_patterns()
    report = {
        "scope": __doc__,
        "curve": str(args.curve),
        "checkpoint": step,
        "source_report_sha256": sha256(args.curve / "report.json"),
        "seeds": args.seeds,
        "neutral_warmup": 128,
        "test_decisions_per_cue": 128,
        "bins": [[0, 8], [8, 16], [16, 48], [48, 128]],
        "rows": [],
        "baseline_state": str(args.baseline_state) if args.baseline_state else None,
        "unpaired_state": str(args.unpaired_state) if args.unpaired_state else None,
    }
    for arm in ("original", "paired", "unpaired_within_cue"):
        explicit = (
            args.baseline_state
            if arm == "original"
            else args.unpaired_state
            if arm == "unpaired_within_cue"
            else None
        )
        if arm == "original" and explicit is None:
            c.restore(original, state, weights_only=True)
        else:
            path = explicit or args.curve / f"{arm}-{step}"
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            c.restore(arrays, json.loads(path.with_suffix(".json").read_text()), weights_only=True)
        weights = c.plasticity.weights.copy()
        records = []
        for seed in args.seeds:
            for cue in cues:
                c.reset_dynamics(seed)
                for _ in range(128):
                    c.observe(gray)
                actions = [c.choose(c.observe(patterns[cue]))[0] for _ in range(128)]
                records.append({"seed": seed, "cue": cue, "actions": actions})

        def scored(lo, hi, records=records):
            return score(
                [{"cue": r["cue"], "actions": dict(Counter(r["actions"][lo:hi]))} for r in records],
                source["reverse_mapping"],
                buttons=buttons,
                cues=cues,
            )

        row = {
            "arm": arm,
            "overall": scored(0, 128),
            "state_sha256": sha256(path.with_suffix(".npz"))
            if arm != "original" or explicit
            else None,
            "time_bins": [{"start": lo, "end": hi, **scored(lo, hi)} for lo, hi in report["bins"]],
            "raw": records,
        }
        np.testing.assert_array_equal(weights, c.plasticity.weights)
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(arm, row["overall"], row["time_bins"], flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
