"""Audit the preselected visual-retention gate from complete, immutable panels.

Descriptive engineering thresholds, NOT a statistical significance test or
evidence of Pokemon competence. Partial report files may be combined only when
they cover distinct arms of the SAME final checkpoint and held-out protocol.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from evaluate_learning_choices import score

from pokefly.checkpoint import sha256
from pokefly.runner import run_directory, write_json

ARMS = {"original", "paired", "unpaired_within_cue"}


def evaluate_panel(rows, reverse=False, buttons=("left", "right"), cues=("left", "right")):
    if set(rows) != ARMS:
        raise ValueError("All original, paired and shuffled arms are required")
    scores = {}
    for arm, row in rows.items():
        scored = score(
            [{"cue": r["cue"], "actions": dict(Counter(r["actions"]))} for r in row["raw"]],
            reverse, buttons=buttons, cues=cues,
        )
        if scored != row["overall"]:
            raise ValueError("Reported score does not match the full raw choices")
        scores[arm] = scored
    paired = scores["paired"]["balanced_conditional_accuracy"]
    cue_accuracy = {}
    for i, cue in enumerate(cues):
        counts = scores["paired"]["per_cue_button_counts"][cue]
        total = sum(counts.values())
        cue_accuracy[cue] = counts[buttons[i ^ int(reverse)]] / total if total else None
    advantages = {
        arm: paired - scores[arm]["balanced_conditional_accuracy"]
        if paired is not None and scores[arm]["balanced_conditional_accuracy"] is not None
        else None
        for arm in ("original", "unpaired_within_cue")
    }

    def at_least(value, threshold):
        return value is not None and value + 1e-12 >= threshold

    return {
        "scores": scores, "paired_per_cue_accuracy": cue_accuracy,
        "paired_balanced_advantages": advantages,
        "passed": at_least(paired, 0.60)
        and all(at_least(value, 0.55) for value in cue_accuracy.values())
        and all(at_least(value, 0.05) for value in advantages.values()),
    }


def audit_reports(paths):
    groups = {}
    common_protocol = None
    for path in paths:
        report_file = path / "report.json"
        report = json.loads(report_file.read_text())
        curve = Path(report["curve"]).resolve()
        source = json.loads((curve / "report.json").read_text())
        if report.get("status") != "completed" or source.get("status") != "completed":
            raise ValueError("Every source and retention panel needs explicit completed status")
        seeds = report["seeds"]
        if len(set(seeds)) < 4 or len(set(seeds)) != len(seeds) or source["seed"] in seeds:
            raise ValueError("At least four distinct held-out noise seeds are required")
        step = max(source["checkpoints"])
        if (report["checkpoint"] != step or report["neutral_warmup"] != 128
                or report["test_decisions_per_cue"] != 128):
            raise ValueError("Final checkpoint and registered neutral-warmup protocol required")
        cues = tuple(source.get("cue_images", ["left", "right"]))
        buttons = tuple(source.get("rewarded_button_pair", ["left", "right"]))
        correct_reward = source.get("correct_reward", 1.0)
        incorrect_reward = source.get("incorrect_reward", 0.0)
        if (not math.isfinite(correct_reward) or correct_reward <= 0
                or not -1 <= incorrect_reward <= 0):
            raise ValueError("Invalid registered assay reward strengths")
        reversal = "pre_reversal_scores" in source
        protocol = (json.loads(Path(source["config"]).read_text()), seeds,
                    cues, buttons, source["reverse_mapping"], reversal,
                    correct_reward, incorrect_reward)
        if common_protocol is not None and protocol != common_protocol:
            raise ValueError("Cannot pool different circuits or retention protocols")
        common_protocol = protocol
        if reversal and not source.get("same_acquired_start_for_both_reversal_arms"):
            raise ValueError("Reversal must start both arms from the same acquired state")
        group = groups.setdefault(str(curve), {
            "seed": source["seed"], "rows": {}, "reports": [], "checkpoint": step,
        })
        group["reports"].append({"path": str(path), "sha256": sha256(report_file)})
        if set(report["arms"]) != {row["arm"] for row in report["rows"]}:
            raise ValueError("Declared and recorded arms differ")
        for row in report["rows"]:
            arm = row["arm"]
            if arm not in ARMS or arm in group["rows"]:
                raise ValueError("Duplicate or unknown arm; do not count repeated controls twice")
            raw = row["raw"]
            if (len(raw) != len(seeds) * len(cues)
                    or {(r["seed"], r["cue"]) for r in raw}
                    != {(seed, cue) for seed in seeds for cue in cues}
                    or any(len(r["actions"]) != 128 for r in raw)):
                raise ValueError("Incomplete raw retention choices")
            if arm != "original":
                expected = sha256(curve / f"{arm}-{step}.npz")
            elif reversal:
                expected = source["shared_reversal_initial_state_sha256"][".npz"]
            else:
                expected = None
            if row["state_sha256"] != expected:
                raise ValueError("Retention must use the registered final neural state")
            group["rows"][arm] = row
    if not groups:
        raise ValueError("Retention reports are required")
    if len({g["seed"] for g in groups.values()}) != len(groups):
        raise ValueError("Training seeds must be independent, not duplicate source runs")
    results = [{
        "curve": curve, "training_seed": group["seed"],
        "checkpoint": group["checkpoint"], "reports": group["reports"],
        **evaluate_panel(group["rows"], common_protocol[4], common_protocol[3], common_protocol[2]),
    } for curve, group in groups.items()]
    return {
        "scope": __doc__, "status": "completed", "rows": results,
        "held_out_noise_seeds": common_protocol[1],
        "assay_rewards": {"correct": common_protocol[6], "incorrect": common_protocol[7]},
        "thresholds": {"balanced": 0.60, "each_cue": 0.55, "advantage_each_control": 0.05},
        "minimum_independent_training_seeds": 2,
        "passed": len(results) >= 2 and all(r["passed"] for r in results),
        "shared_original_control_is_not_an_independent_replication": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    report = audit_reports(args.reports)
    output = run_directory("visual-retention-gate")
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
