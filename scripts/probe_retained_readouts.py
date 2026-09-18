"""ROM-free frozen-neural readout audit; not a newly trained controller.

Replay the SAME actual motor spike counts through two predeclared fixed
decoders: the saved parallel readout and the existing one-second sustained
readout. No weight, threshold, direction-specific gain or mapping is fitted.
Diagnostic weights never enter Pokemon. Report all arms, modes and time bins.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from evaluate_learning_choices import score

from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.motors import MotorDecoder
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curve", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1701, 1702])
    args = p.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        p.error("At least two distinct retention noise streams required")
    source = json.loads((args.curve / "report.json").read_text())
    step = max(source["checkpoints"])
    config = load_config(Path(source["config"])).brain
    if config.motor.arbitration != "parallel-v2":
        p.error("This one-factor audit expects a saved parallel-v2 model")
    modes = {
        "parallel": config.motor,
        "sustained_1s": replace(
            config.motor, arbitration="sustained-v3", direction_trace_seconds=1
        ),
    }
    cues = tuple(source.get("cue_images", ["left", "right"]))
    buttons = tuple(source.get("rewarded_button_pair", ["left", "right"]))
    output = run_directory("retained-readout-audit")
    c = InternalBrain(device="cuda", config=config)
    original, original_state = c.snapshot()
    motor_cells = np.unique(np.concatenate(list(c.motors.values())))
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
        "modes": {key: asdict(value) for key, value in modes.items()},
        "bins": [[0, 8], [8, 16], [16, 48], [48, 128]],
        "rows": [],
        "caveat": "Weights were trained with the original decoder. A rescore does not prove "
        "that training with the alternative decoder would produce these weights.",
    }
    recorded_counts = {}
    for arm in ("original", "paired", "unpaired_within_cue"):
        path = None
        if arm == "original":
            c.restore(original, original_state, weights_only=True)
        else:
            path = args.curve / f"{arm}-{step}.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            c.restore(arrays, json.loads(path.with_suffix(".json").read_text()), weights_only=True)
        weights = c.plasticity.weights.copy()
        records = {mode: [] for mode in modes}
        for seed in args.seeds:
            for cue in cues:
                c.reset_dynamics(seed)
                for _ in range(128):
                    c.observe(gray)
                decoders = {key: MotorDecoder(c.motors, value) for key, value in modes.items()}
                actions, counts = {key: [] for key in modes}, []
                for _ in range(128):
                    observed = c.observe(patterns[cue])
                    original_action = c.choose(observed)[0]
                    for mode, decoder in decoders.items():
                        action = decoder.choose(observed.counts, c.brain_steps * c.brain.dt)[0]
                        if mode == "parallel":
                            assert action == original_action
                        actions[mode].append(action)
                    counts.append(observed.counts[motor_cells].astype(np.uint16))
                for mode in modes:
                    records[mode].append({"seed": seed, "cue": cue, "actions": actions[mode]})
                recorded_counts[f"{arm}_{seed}_{cue}"] = np.stack(counts)
        np.testing.assert_array_equal(weights, c.plasticity.weights)
        for mode, data in records.items():

            def scored(lo, hi, data=data):
                return score(
                    [
                        {"cue": r["cue"], "actions": dict(Counter(r["actions"][lo:hi]))}
                        for r in data
                    ],
                    source["reverse_mapping"],
                    buttons=buttons,
                    cues=cues,
                )

            row = {
                "arm": arm,
                "decoder": mode,
                "overall": scored(0, 128),
                "state_sha256": sha256(path) if path else None,
                "time_bins": [
                    {"start": lo, "end": hi, **scored(lo, hi)} for lo, hi in report["bins"]
                ],
                "raw": data,
            }
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(arm, mode, row["overall"], flush=True)
    np.savez_compressed(
        output / "motor-counts.npz", body_ids=c.body_ids[motor_cells], **recorded_counts
    )
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
