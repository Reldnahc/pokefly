"""Compose a verified matched baseline from an exactly recovered rule trial.

No simulation or new evidence: join the completed candidate with its original
shared frozen control, retaining the full provenance chain. This permits a
subsequent ONE-factor test against that candidate without relabeling trials.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from continue_rule_gameplay import measure_complete, read_rows, stitch, validate_source
from evaluate_visual_learning_rate import validate_one_factor

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, load_config
from pokefly.runner import run_directory, write_json


def read(path):
    return json.loads(Path(path).read_text())


def compose(path):
    report = read(path)
    if report.get("status") != "completed" or len(report.get("rows", [])) != 1:
        raise ValueError("A completed recovered single-candidate trial is required")
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
