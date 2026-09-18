"""Finish a registered interrupted practice series, preserving its evidence.

Restore the interrupted attempt's full checkpoint, check its overlapping log,
then finish the ORIGINAL attempt/evaluation schedule. New-game resets remain
explicit interventions. No new episodes, policy, reward or model changes.
"""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from continue_rule_gameplay import measure_complete, read_rows, stitch
from evaluate_saved_gameplay import measure

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def normalized(config):
    return asdict(ExperimentConfig.from_dict(config))


def validate_schedule(report):
    if (report.get("status") != "running" or report.get("recovery")
            or not report.get("initial_actual_game_source")):
        raise ValueError("An interrupted original actual-game practice series is required")
    training, evaluation = report["training_seeds"], report["eval_seeds"]
    if (len(training) < 2 or len(evaluation) < 2
            or len(set(training + evaluation)) != len(training + evaluation)
            or report["initial_actual_game_source"]["source_launch_seed"] in evaluation
            or report["steps_per_attempt"] < 1 or report["evaluation_steps_per_arm"] < 1):
        raise ValueError("Invalid original practice/evaluation schedule")
    rows = report["rows"]
    if (len(rows) >= len(training) or any(r["phase"] != "training" for r in rows)
            or [r["seed"] for r in rows] != training[:len(rows)]
            or any(r["samples"] != report["steps_per_attempt"] for r in rows)):
        raise ValueError("Recovery needs a complete training prefix and unfinished next attempt")


def validate_interrupted(report, game, saved, fixed, expected_weights):
    options, progress = game["options"], saved["experiment"]
    if (options["mode"] != "learn" or progress["mode"] != "learn"
            or not options["intro"] or options.get("resume") or options.get("load_state")
            or options["seed"] != report["training_seeds"][len(report["rows"])]
            or options["steps"] != report["steps_per_attempt"]
            or not 0 < progress["sample"] < report["steps_per_attempt"]
            or normalized(game["config"]) != fixed or normalized(progress["config"]) != fixed
            or game["rom_sha1"] != saved["rom_sha1"]
            or not options.get("weights")
            or Path(options["weights"]).resolve() != expected_weights.resolve()):
        raise ValueError("Interrupted attempt does not match its registered series/weight chain")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--interrupted-run", type=Path, required=True)
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--validate-only", action="store_true",
                   help="Read-only provenance/preflight checks; no emulator or neural simulation")
    args = p.parse_args()
    if not 0 <= args.port <= 65535:
        p.error("Invalid dashboard port")
    source_file = args.source / "report.json"
    source = json.loads(source_file.read_text())
    validate_schedule(source)
    if sha256(Path(source["config"])) != source["config_sha256"]:
        raise ValueError("Source model configuration changed")
    fixed = asdict(load_config(Path(source["config"])))
    initial = source["initial_actual_game_source"]
    initial_checkpoint = Path(initial["checkpoint"])
    if (sha256(initial_checkpoint / "brain.npz") != initial["brain_sha256"]
            or sha256(Path(initial["run"]) / "config.json") != initial["config_sha256"]):
        raise ValueError("Initial actual-game source changed")
    _, initial_state = read_checkpoint(initial_checkpoint)
    if (initial_state["experiment"]["mode"] != "learn"
            or initial_state["experiment"]["sample"] != initial["completed_samples"]
            or normalized(initial_state["experiment"]["config"]) != fixed
            or initial_state["rom_sha1"] != initial["rom_sha1"]):
        raise ValueError("Initial game checkpoint identity changed")
    expected_weights = initial_checkpoint
    copied_provenance = []
    for row in source["rows"]:
        path = Path(row["run"])
        stored = json.loads((path / "config.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        _, completed = read_checkpoint(path / "latest-checkpoint.json")
        if (summary["reason"] != "step_limit" or measure(path) != {
                k: v for k, v in row.items() if k not in {"phase", "seed", "initial_weights"}}
                or stored["options"]["mode"] != "learn" or not stored["options"]["intro"]
                or stored["options"]["seed"] != row["seed"]
                or stored["options"].get("resume") or stored["options"].get("load_state")
                or stored["options"]["steps"] != source["steps_per_attempt"]
                or Path(stored["options"]["weights"]).resolve() != expected_weights.resolve()
                or completed["experiment"]["sample"] != source["steps_per_attempt"]
                or completed["experiment"]["mode"] != "learn"
                or normalized(completed["experiment"]["config"]) != fixed
                or normalized(stored["config"]) != fixed
                or completed["rom_sha1"] != initial["rom_sha1"]):
            raise ValueError("Completed training prefix changed")
        expected_weights = path / "latest-checkpoint.json"
        copied_provenance.append({
            "run": str(path), "checkpoint": completed["directory"],
            "brain_sha256": sha256(Path(completed["directory"]) / "brain.npz"),
            "trajectory_sha256": sha256(path / "trajectory.jsonl"),
        })
    _, saved = read_checkpoint(args.interrupted_run / "latest-checkpoint.json")
    game = json.loads((args.interrupted_run / "config.json").read_text())
    validate_interrupted(source, game, saved, fixed, expected_weights)
    if saved["rom_sha1"] != initial["rom_sha1"]:
        raise ValueError("Interrupted ROM identity changed")
    start, total = saved["experiment"]["sample"], source["steps_per_attempt"]
    before = read_rows(args.interrupted_run, interrupted=True)
    stitch(before, [], start, start)
    checkpoint = Path(saved["directory"])
    if args.validate_only:
        print(json.dumps({"validated": True, "source": str(args.source),
                          "checkpoint": str(checkpoint), "resume_sample": start,
                          "registered_steps": total,
                          "copied_completed_attempts": len(source["rows"])}), flush=True)
        return
    output = run_directory("resumed-game-series")
    config = output / "fixed-config.json"
    write_json(config, fixed)
    report = copy.deepcopy(source)
    report.update(config=str(config), config_sha256=sha256(config))
    report["recovery"] = {
        "scope": __doc__, "source": str(args.source), "source_report_sha256": sha256(source_file),
        "copied_rows_are_not_new_trials": True, "completed_prefix": copied_provenance,
        "checkpoint": str(checkpoint), "brain_sha256": sha256(checkpoint / "brain.npz"),
        "source_trajectory_sha256": sha256(args.interrupted_run / "trajectory.jsonl"),
    }
    write_json(output / "report.json", report)
    print("Recovery:", output, flush=True)
    rom = resolve_rom(None, Path.cwd())
    remaining = source["training_seeds"][len(source["rows"]):]
    previous = None
    for index, seed in enumerate(remaining):
        path = train(TrainOptions(
            rom=rom, device="cuda", seed=seed, hz=0, port=args.port, checkpoint_every=500,
            steps=total - start if index == 0 else total,
            resume=checkpoint if index == 0 else None,
            weights=previous if index else None, intro=index != 0, mode="learn",
        ))
        summary = json.loads((path / "summary.json").read_text())
        if summary["reason"] != "step_limit" or summary["samples"] != total:
            raise RuntimeError("Recovery interrupted; new checkpoints remain preserved")
        if index == 0:
            rows, overlap = stitch(before, read_rows(path), start, total)
            result = measure_complete(path, summary, rows)
            report["recovery"]["overlap_decisions_exact"] = overlap
            report["recovery"]["segments"] = [
                {"run": str(args.interrupted_run), "first_sample": 1, "last_sample": start},
                {"run": str(path), "first_sample": start + 1, "last_sample": total},
            ]
        else:
            result = measure(path)
        row = {"phase": "training", "seed": seed,
               "initial_weights": str(expected_weights if index == 0 else previous), **result}
        report["rows"].append(row)
        _, completed = read_checkpoint(path / "latest-checkpoint.json")
        previous = Path(completed["directory"])
        write_json(output / "report.json", report)
        print(json.dumps(row), flush=True)
    report["evaluation_source"] = str(previous)
    report["evaluation_brain_sha256"] = sha256(previous / "brain.npz")
    write_json(output / "report.json", report)
    for index, seed in enumerate(source["eval_seeds"]):
        arms = ("original", "retained") if index % 2 == 0 else ("retained", "original")
        for arm in arms:
            path = train(TrainOptions(
                rom=rom, device="cuda", seed=seed, steps=source["evaluation_steps_per_arm"],
                hz=0, mode="frozen", intro=True, dashboard=False, checkpoint_every=500,
                config=config if arm == "original" else None,
                weights=previous if arm == "retained" else None,
            ))
            row = {"phase": arm, "seed": seed, **measure(path)}
            assert row["weight_updates_this_evaluation"] == 0
            arrays, _ = read_checkpoint(path / "latest-checkpoint.json")
            expected = (
                arrays["base"] if arm == "original" else read_checkpoint(previous)[0]["weights"]
            )
            np.testing.assert_array_equal(arrays["weights"], expected)
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
            if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
                raise RuntimeError("Frozen evaluation interrupted; partial evidence is preserved")
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
