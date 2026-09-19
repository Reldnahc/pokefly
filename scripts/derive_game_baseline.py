"""Compose a verified matched baseline from a completed one-factor trial.

No simulation or new evidence: join the completed candidate with its original
shared frozen control, retaining the full provenance chain. This permits a
subsequent ONE-factor test against that candidate without relabeling trials.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from continue_rule_gameplay import measure_complete, read_rows, stitch, validate_source
from evaluate_saved_gameplay import measure
from evaluate_visual_learning_rate import validate_one_factor

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, load_config
from pokefly.runner import run_directory, write_json


def read(path):
    return json.loads(Path(path).read_text())


def compose_fresh(path, candidate):
    """A complete unsplit candidate plus its already verified shared control."""
    base = Path(candidate["baseline_report"])
    if sha256(base) != candidate["baseline_report_sha256"]:
        raise ValueError("Original matched controls changed")
    original = read(base)
    if (original.get("status") != "completed" or original.get("continuation_source")
            or len(original["rows"]) != 2
            or {r["mode"] for r in original["rows"]} != {"learn", "frozen"}
            or candidate["reused_controls_not_new_trials"] != original["rows"]
            or candidate["seed"] != original["seed"] or candidate["steps"] != original["steps"]):
        raise ValueError("Seed, budget or shared controls disagree")
    for source in (candidate, original):
        if sha256(Path(source["config"])) != source["config_sha256"]:
            raise ValueError("Source model file changed")
    fixed = asdict(load_config(Path(candidate["config"])))
    factor = candidate["factor"]
    old, new = validate_one_factor(asdict(load_config(Path(original["config"]))), fixed, factor)
    if (candidate[f"old_{factor}"], candidate[f"new_{factor}"]) != (old, new):
        raise ValueError("Reported factor values differ from the actual model settings")
    prefix = candidate.get("prefix_verification", {})
    if prefix.get("samples") != 64 or not prefix.get("all_fields_exact"):
        raise ValueError("The complete 64-decision frozen prefix is required")
    frozen = next(r for r in original["rows"] if r["mode"] == "frozen")
    rows_by_arm, starts, roms, final = {}, set(), set(), None
    for arm, row in (("frozen", frozen), ("learn", candidate["rows"][0])):
        game = Path(row["run"])
        stored, summary = read(game / "config.json"), read(game / "summary.json")
        actual = measure(game)
        if actual != {k: row[k] for k in actual}:
            raise ValueError("Recorded outcome differs from raw gameplay")
        options = stored["options"]
        if (summary["reason"] != "step_limit" or actual["samples"] != candidate["steps"]
                or options["steps"] != candidate["steps"] or options["seed"] != candidate["seed"]
                or options["mode"] != arm or not options["intro"]
                or any(options.get(k) for k in ("resume", "weights", "load_state"))):
            raise ValueError("Fresh matched candidate/control protocol changed")
        arrays, saved = read_checkpoint(game / "latest-checkpoint.json")
        stored_fixed = asdict(ExperimentConfig.from_dict(stored["config"], checkpoint=True))
        if (saved["experiment"]["mode"] != arm
                or saved["experiment"]["sample"] != candidate["steps"]
                or saved["rom_sha1"] != stored["rom_sha1"]
                or asdict(ExperimentConfig.from_dict(
                    saved["experiment"]["config"], checkpoint=True)) != stored_fixed):
            raise ValueError("Final checkpoint does not match its game")
        if arm == "learn":
            if stored_fixed != fixed:
                raise ValueError("Learning run used a different neural model")
            final = Path(saved["directory"])
        else:
            np.testing.assert_array_equal(arrays["weights"], arrays["base"])
            if actual["weight_updates_this_evaluation"]:
                raise ValueError("Shared frozen control changed weights")
            # A reused original may come from an earlier plasticity rule. Its
            # frozen dynamics, not its unused eligibility state, must match.
            current, reference = dict(stored_fixed), dict(fixed)
            current["brain"], reference["brain"] = dict(current["brain"]), dict(reference["brain"])
            current["brain"].pop("plasticity")
            reference["brain"].pop("plasticity")
            if current != reference:
                raise ValueError("Shared control has different physical dynamics or rewards")
        raw = read_rows(game)
        if ([r["sample"] for r in raw] != list(range(1, candidate["steps"] + 1))
                or any(r.get("action_source") != "fly" for r in raw)):
            raise ValueError("Incomplete or externally controlled gameplay")
        rows_by_arm[arm] = raw
        starts.add(sha256(game / "start.state"))
        roms.add(stored["rom_sha1"])
    prefix_game = Path(prefix["run"])
    prefix_rows = read_rows(prefix_game)
    keys = ("action", "buttons", "spikes_total", "groups", "motor_rates_hz", "telemetry",
            "reward", "reward_events", "input_window")
    if (len(prefix_rows) != 64 or len(starts) != 1 or len(roms) != 1
            or sha256(prefix_game / "start.state") not in starts
            or [{k: r[k] for k in keys} for r in prefix_rows]
            != [{k: r[k] for k in keys} for r in rows_by_arm["frozen"][:64]]):
        raise ValueError("Matched initial game or recorded frozen prefix changed")
    return {
        "scope": __doc__, "status": "completed", "new_trials": False,
        "seed": candidate["seed"], "steps": candidate["steps"],
        "config": candidate["config"], "config_sha256": candidate["config_sha256"],
        "rows": [frozen, {"mode": "learn", **candidate["rows"][0]}],
        "provenance": [{"report": str(p), "sha256": sha256(p)} for p in (path, base)],
        "candidate_final_checkpoint": str(final),
        "candidate_final_brain_sha256": sha256(final / "brain.npz"),
        "frozen_control_shared_not_independent": True,
        "candidate_resume_segments_not_independent": False,
        "raw_prefix_rechecked": True,
    }


def compose(path):
    report = read(path)
    if report.get("status") != "completed" or len(report.get("rows", [])) != 1:
        raise ValueError("A completed single-candidate trial is required")
    if "source_report" not in report:
        return compose_fresh(path, report)
    parent = Path(report["source_report"])
    if sha256(parent) != report["source_report_sha256"]:
        raise ValueError("Original candidate protocol changed")
    candidate = read(parent)
    if (candidate.get("factor") != "rule"
            or not candidate.get("prefix_verification", {}).get("all_fields_exact")):
        raise ValueError("An explicitly verified one-rule candidate is required")
    base = Path(candidate["baseline_report"])
    if sha256(base) != candidate["baseline_report_sha256"]:
        raise ValueError("Original matched controls changed")
    original = read(base)
    if (original.get("status") != "completed" or original.get("continuation_source")
            or {r["mode"] for r in original["rows"]} != {"learn", "frozen"}
            or report["reused_controls_not_new_trials"] != original["rows"]
            or candidate["reused_controls_not_new_trials"] != original["rows"]
            or report["seed"] != candidate["seed"] or report["seed"] != original["seed"]
            or report["steps"] != candidate["steps"] or report["steps"] != original["steps"]
            or report["candidate_config"] != candidate["config"]):
        raise ValueError("Seed, budget, source model or shared controls disagree")
    for source in (candidate, original):
        if sha256(Path(source["config"])) != source["config_sha256"]:
            raise ValueError("Source model file changed")
    fixed = asdict(load_config(Path(candidate["config"])))
    validate_one_factor(asdict(load_config(Path(original["config"]))), fixed, "rule")
    segments = report["segments"]
    if len(segments) != 2:
        raise ValueError("Exactly two nonoverlapping recovery segments required")
    before, after = (Path(s["run"]) for s in segments)
    boundary, total = segments[0]["last_sample"], report["steps"]
    if (segments[0]["first_sample"] != 1 or segments[1]["first_sample"] != boundary + 1
            or segments[1]["last_sample"] != total
            or sha256(before / "trajectory.jsonl") != report["source_trajectory_sha256"]):
        raise ValueError("Recovery source prefix or interval changed")
    checkpoint = Path(report["source_checkpoint"])
    if sha256(checkpoint / "brain.npz") != report["source_brain_sha256"]:
        raise ValueError("Recovery checkpoint changed")
    _, before_saved = read_checkpoint(checkpoint)
    validate_source(candidate, read(before / "config.json"), before_saved, fixed)
    options = read(after / "config.json")["options"]
    if (Path(options["resume"]).resolve() != checkpoint.resolve()
            or options["steps"] != total - boundary or options["seed"] != report["seed"]):
        raise ValueError("Resumed attempt did not restore the registered checkpoint/budget")
    rows, overlap = stitch(read_rows(before, interrupted=True), read_rows(after), boundary, total)
    summary = read(after / "summary.json")
    actual = measure_complete(after, summary, rows)
    if (summary["reason"] != "step_limit" or actual != report["rows"][0]
            or overlap != report["overlap_decisions_exact"]):
        raise ValueError("Completed candidate evidence no longer reproduces")
    _, saved = read_checkpoint(after / "latest-checkpoint.json")
    if (saved["experiment"]["mode"] != "learn" or saved["experiment"]["sample"] != total
            or asdict(ExperimentConfig.from_dict(saved["experiment"]["config"])) != fixed
            or saved["rom_sha1"] != before_saved["rom_sha1"]):
        raise ValueError("Final candidate checkpoint does not match its registered trial")
    frozen = next(r for r in original["rows"] if r["mode"] == "frozen")
    return {
        "scope": __doc__, "status": "completed", "new_trials": False,
        "seed": report["seed"], "steps": total,
        "config": candidate["config"], "config_sha256": candidate["config_sha256"],
        "rows": [frozen, {"mode": "learn", **actual}],
        "provenance": [{"report": str(p), "sha256": sha256(p)} for p in (path, parent, base)],
        "candidate_final_checkpoint": saved["directory"],
        "candidate_final_brain_sha256": sha256(Path(saved["directory"]) / "brain.npz"),
        "frozen_control_shared_not_independent": True,
        "candidate_resume_segments_not_independent": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    report = compose(args.report)
    output = run_directory("derived-game-baseline")
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
