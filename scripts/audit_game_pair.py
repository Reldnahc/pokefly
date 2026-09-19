"""Audit completed learning/frozen full-game pairs with conservative outcomes.

Early reward delivery and completed encounters are recorded separately. A
reward paid at the last faint is not mislabeled as having cleared every
remaining battle/dialogue screen. This audit does not run or train a fly.
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from evaluate_saved_gameplay import measure

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, load_config
from pokefly.runner import run_directory, write_json

OUTCOMES = ("battle_win", "capture", "gym_win", "rival_win")


def encounter_outcomes(rows, rewards):
    """For a full fresh-game log, separate delivered rewards from encounter ends."""
    events = [event for row in rows for event in row["reward_events"]]
    delivered = Counter(event["category"] for event in events)
    if delivered != Counter(rewards["counts"]):
        raise ValueError("Complete trajectory and final reward ledger disagree")
    pending, active = Counter(), rewards["active"]
    if active and active.get("paid"):
        pending.update(e["category"] for e in events if e.get("encounter") == active["id"])
        if (pending["battle_win"] + pending["capture"] != 1
                or set(pending) - set(OUTCOMES)):
            raise ValueError("Paid open encounter lacks its unique recorded outcome")
    completed = {key: delivered[key] - pending[key] for key in OUTCOMES}
    if min(completed.values()) < 0:
        raise ValueError("Pending outcome exceeds the delivered ledger")
    return {"awarded_outcomes": {key: delivered[key] for key in OUTCOMES},
            "completed_outcomes": completed,
            "paid_but_unfinished_encounter": dict(active) if pending else None}


def progress_measurements(rows, rewards):
    """Observer-only first sampled milestones; never inputs or extra rewards."""
    return {
        "first_battle": next((r["sample"] for r in rows if r["telemetry"]["battle"]), None),
        "first_route1": next((r["sample"] for r in rows if r["telemetry"]["map"] == 12), None),
        "first_victory_reward": next(
            (r["sample"] for r in rows
             if any(e["category"] == "battle_win" for e in r["reward_events"])), None,
        ),
        **encounter_outcomes(rows, rewards),
    }


def audit(paths):
    protocol, seeds, evidence, comparisons = None, set(), [], []
    for path in paths:
        report_file = path / "report.json"
        report = json.loads(report_file.read_text())
        if (report.get("status") != "completed" or report.get("continuation_source")
                or len(report.get("rows", [])) != 2
                or {r["mode"] for r in report["rows"]} != {"learn", "frozen"}):
            raise ValueError("Completed fresh-game learning/frozen pairs required")
        if report["seed"] in seeds:
            raise ValueError("Duplicate seed is not an independent pair")
        seeds.add(report["seed"])
        config = Path(report["config"])
        fixed = asdict(load_config(config))
        if sha256(config) != report["config_sha256"]:
            raise ValueError("Experiment configuration changed")
        current = fixed, report["steps"]
        if protocol is not None and protocol != current:
            raise ValueError("Pairs have different model settings or decision budgets")
        protocol = current
        arms, start_hash, rom_hash = {}, None, None
        for row in report["rows"]:
            game = Path(row["run"])
            actual = measure(game)
            if actual != {key: row[key] for key in actual}:
                raise ValueError("Recorded outcome differs from its raw game evidence")
            stored = json.loads((game / "config.json").read_text())
            summary = json.loads((game / "summary.json").read_text())
            options = stored["options"]
            if (summary["reason"] != "step_limit" or actual["samples"] != report["steps"]
                    or options["seed"] != report["seed"] or options["mode"] != row["mode"]
                    or options["steps"] != report["steps"] or not options["intro"]
                    or any(options.get(k) for k in ("resume", "load_state", "weights"))
                    or asdict(ExperimentConfig.from_dict(stored["config"], checkpoint=True))
                    != fixed):
                raise ValueError("Initial state, mode, seed, model or completion changed")
            arrays, saved = read_checkpoint(game / "latest-checkpoint.json")
            if (saved["experiment"]["sample"] != report["steps"]
                    or saved["experiment"]["mode"] != row["mode"]
                    or asdict(ExperimentConfig.from_dict(
                        saved["experiment"]["config"], checkpoint=True)) != fixed
                    or saved["rom_sha1"] != stored["rom_sha1"]):
                raise ValueError("Final checkpoint does not match this experiment")
            if row["mode"] == "frozen":
                if actual["weight_updates_this_evaluation"]:
                    raise ValueError("Frozen control contains a weight update")
                np.testing.assert_array_equal(arrays["weights"], arrays["base"])
            raw = [json.loads(line)
                   for line in (game / "trajectory.jsonl").read_text().splitlines()]
            if ([r["sample"] for r in raw] != list(range(1, report["steps"] + 1))
                    or any(r["action_source"] != "fly" for r in raw)):
                raise ValueError("Incomplete or externally controlled trajectory")
            game_start = sha256(game / "start.state")
            if ((start_hash is not None and start_hash != game_start)
                    or (rom_hash is not None and rom_hash != stored["rom_sha1"])):
                raise ValueError("Matched arms started from different games")
            start_hash, rom_hash = game_start, stored["rom_sha1"]
            outcome = progress_measurements(raw, saved["rewards"])
            if Counter(summary["reward_counts"]) != Counter(saved["rewards"]["counts"]):
                raise ValueError("Summary and final checkpoint reward counts disagree")
            arms[row["mode"]] = {
                "run": str(game), "checkpoint": saved["directory"],
                "brain_sha256": sha256(Path(saved["directory"]) / "brain.npz"),
                "trajectory_sha256": sha256(game / "trajectory.jsonl"),
                "positions": actual["tiles"], "maps": actual["maps"],
                "first_starter": actual["first_starter"],
                "final_state": actual["final_state"], **outcome,
            }
        comparisons.append({"seed": report["seed"], **arms})
        evidence.append({"path": str(path), "sha256": sha256(report_file)})
    if len(seeds) < 2:
        raise ValueError("At least two complete independent pairs required")
    return {"scope": __doc__, "status": "completed", "evidence": evidence,
            "comparisons": comparisons, "new_trials": False,
            "retained_weight_transfer_tested": False, "statistical_significance_claimed": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, nargs="+", required=True)
    args = p.parse_args()
    report = audit(args.reports)
    output = run_directory("full-game-pair-audit")
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
