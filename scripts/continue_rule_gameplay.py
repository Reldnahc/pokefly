"""Finish an interrupted one-factor game trial without restarting or changing it.

Preserve the source artifacts, pin its last complete checkpoint, and identify
the two non-overlapping trajectory segments explicitly. Old post-checkpoint
samples are replay checks, not extra trials. No config/reward/policy changes.
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from pokefly.actions import count_buttons
from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def read_rows(path, *, interrupted=False):
    lines = (path / "trajectory.jsonl").read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not interrupted or i != len(lines) - 1:
                raise
            print("Ignoring only the interrupted log's incomplete final line", flush=True)
    return rows


def validate_source(report, game, saved, fixed):
    options, progress = game["options"], saved["experiment"]
    if report.get("status") != "running" or report.get("rows"):
        raise ValueError("An interrupted, unfinished single-candidate report is required")
    if not report.get("prefix_verification", {}).get("all_fields_exact"):
        raise ValueError("Missing original frozen-control verification")
    if (
        options["mode"] != "learn" or progress["mode"] != "learn"
        or not options["intro"] or options.get("resume") or options.get("weights")
        or options.get("load_state") or options["seed"] != report["seed"]
        or options["steps"] != report["steps"]
        or asdict(ExperimentConfig.from_dict(game["config"])) != fixed
        or asdict(ExperimentConfig.from_dict(progress["config"])) != fixed
        or not 0 < progress["sample"] < report["steps"]
        or game["rom_sha1"] != saved["rom_sha1"]
    ):
        raise ValueError("Candidate seed, model, budget, checkpoint or ROM changed")


def stitch(source, continuation, boundary, total):
    prefix = [row for row in source if row["sample"] <= boundary]
    if [r["sample"] for r in prefix] != list(range(1, boundary + 1)):
        raise ValueError("Source prefix is incomplete or duplicated")
    if [r["sample"] for r in continuation] != list(range(boundary + 1, total + 1)):
        raise ValueError("Resumed segment is incomplete or duplicated")
    tail = [row for row in source if boundary < row["sample"] <= total][:20]
    ignored = {"run_id", "compute_ms", "pacing"}
    for old, new in zip(tail, continuation, strict=False):
        if {k: v for k, v in old.items() if k not in ignored} != {
            k: v for k, v in new.items() if k not in ignored
        }:
            raise ValueError(f"Exact replay differs at decision {old['sample']}")
    return prefix + continuation, len(tail)


def measure_complete(path, summary, rows):
    if summary["samples"] != len(rows):
        raise ValueError("Full trajectory does not match cumulative summary")
    buttons = count_buttons(Counter(row["action"] for row in rows))
    if buttons != summary["button_counts"]:
        raise ValueError("Cumulative buttons disagree with the joined trajectory")
    town = [r for r in rows if r["telemetry"]["map"] == 0 and not r["telemetry"]["battle"]]
    route = [r["telemetry"]["y"] for r in rows if r["telemetry"]["map"] == 12]
    return {
        "run": str(path), "samples": len(rows), "tiles": summary["tiles"],
        "maps": summary["maps"], "reward": summary["reward"],
        "reward_counts": summary["reward_counts"], "buttons": buttons,
        "first_starter": next((r["sample"] for r in rows
                               if r["telemetry"]["party"] and r["telemetry"]["levels"]), None),
        "first_house_exit": summary["first_house_exit"], "final_state": summary["final_state"],
        "town_samples": len(town),
        "town_lower_edge_fraction": sum(r["telemetry"]["y"] >= 16 for r in town) / len(town)
        if town else None,
        "route1_minimum_y": min(route) if route else None,
        "weight_updates_this_evaluation": sum(r["learning"]["changed_this_reward"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--game-run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if sha256(Path(report["config"])) != report["config_sha256"]:
        raise ValueError("Candidate config changed")
    if sha256(Path(report["baseline_report"])) != report["baseline_report_sha256"]:
        raise ValueError("Baseline evidence changed")
    game = json.loads((args.game_run / "config.json").read_text())
    _, saved = read_checkpoint(args.game_run / "latest-checkpoint.json")
    validate_source(report, game, saved, asdict(load_config(Path(report["config"]))))
    checkpoint = Path(saved["directory"])
    boundary, total = saved["experiment"]["sample"], report["steps"]
    before = read_rows(args.game_run, interrupted=True)
    # Validate the saved prefix BEFORE spending compute on a continuation.
    stitch(before, [], boundary, boundary)
    output = run_directory("resumed-rule-gameplay")
    result = {
        "scope": __doc__, "status": "running", "seed": report["seed"], "steps": total,
        "source_report": str(args.report), "source_report_sha256": sha256(args.report),
        "source_checkpoint": str(checkpoint),
        "source_brain_sha256": sha256(checkpoint / "brain.npz"),
        "source_trajectory_sha256": sha256(args.game_run / "trajectory.jsonl"),
        "candidate_config": report["config"], "reused_controls_not_new_trials":
        report["reused_controls_not_new_trials"], "rows": [],
    }
    write_json(output / "report.json", result)
    resumed = train(TrainOptions(
        rom=resolve_rom(None, Path.cwd()), device="cuda", seed=report["seed"],
        resume=checkpoint, steps=total - boundary, hz=0, port=args.port, checkpoint_every=500,
    ))
    result["segments"] = [
        {"run": str(args.game_run), "first_sample": 1, "last_sample": boundary},
        {"run": str(resumed), "first_sample": boundary + 1, "last_sample": total},
    ]
    summary = json.loads((resumed / "summary.json").read_text())
    write_json(output / "report.json", result)
    if summary["reason"] != "step_limit" or summary["samples"] != total:
        raise RuntimeError("Continuation interrupted; its checkpoints are preserved")
    rows, verified = stitch(before, read_rows(resumed), boundary, total)
    result["overlap_decisions_exact"] = verified
    result["rows"] = [measure_complete(resumed, summary, rows)]
    result["status"] = "completed"
    write_json(output / "report.json", result)
    print(json.dumps(result["rows"][0]), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
