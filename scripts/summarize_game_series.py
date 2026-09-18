"""Audit completed practice/retention panels without selecting favorable runs.

Every registered fresh-seed original/retained comparison is included. Repeated
original controls must reproduce their complete trajectories and count once.
Descriptive contrasts only, not statistical significance or game mastery.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from evaluate_saved_gameplay import measure

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, load_config
from pokefly.runner import run_directory, write_json


def read(path):
    return json.loads(Path(path).read_text())


def outcomes(row):
    # Rival-win is a SUBSET of battle-win, not a second encounter.
    return {"tiles": row["tiles"], "starter": row["first_starter"] is not None,
            "first_starter": row["first_starter"],
            "battle_wins": row["reward_counts"].get("battle_win", 0),
            "rival_wins": row["reward_counts"].get("rival_win", 0),
            "route1_visited": 12 in row["maps"],
            "final_total_levels": row["final_state"]["levels"]}


def trajectory_signature(rows):
    ignored = {"run_id", "compute_ms", "pacing"}
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps({k: v for k, v in row.items() if k not in ignored},
                                 sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def audit(paths):
    reports, comparisons, shared, sources = [], [], {}, set()
    protocol = None
    starts = {}
    for path in paths:
        report_file = path / "report.json"
        report = read(report_file)
        if report.get("status") != "completed" or not report.get("recovery"):
            raise ValueError("Complete recovered practice/evaluation panels are required")
        if sha256(Path(report["config"])) != report["config_sha256"]:
            raise ValueError("Evaluation model changed")
        fixed = asdict(load_config(Path(report["config"])))
        current = (fixed, report["eval_seeds"], report["evaluation_steps_per_arm"])
        if protocol is not None and current != protocol:
            raise ValueError("Cannot compare different model/evaluation protocols")
        protocol = current
        source_seed = report["initial_actual_game_source"]["source_launch_seed"]
        if source_seed in sources:
            raise ValueError("Duplicate training source is not an independent replication")
        sources.add(source_seed)
        training, testing = report["training_seeds"], report["eval_seeds"]
        expected_panel = {(s, a) for s in testing for a in ("original", "retained")}
        if (len(testing) < 2 or len(set(training + testing)) != len(training + testing)
                or source_seed in testing):
            raise ValueError("Distinct training and held-out evaluation seeds required")
        training_rows = [r for r in report["rows"] if r["phase"] == "training"]
        if ([r["seed"] for r in training_rows] != training
                or any(r["samples"] != report["steps_per_attempt"] for r in training_rows)):
            raise ValueError("The registered training schedule is incomplete")
        checkpoint = Path(report["evaluation_source"])
        if sha256(checkpoint / "brain.npz") != report["evaluation_brain_sha256"]:
            raise ValueError("Trained weights changed after evaluation")
        trained_arrays, trained = read_checkpoint(checkpoint)
        if (trained["experiment"]["mode"] != "learn"
                or asdict(ExperimentConfig.from_dict(trained["experiment"]["config"])) != fixed):
            raise ValueError("Trained neural model identity changed")
        panel = {}
        for row in report["rows"]:
            if row["phase"] == "training":
                continue
            key = (row["seed"], row["phase"])
            if key in panel or key not in expected_panel:
                raise ValueError("Duplicated, unknown or unregistered evaluation arm")
            game = Path(row["run"])
            if measure(game) != {k: v for k, v in row.items() if k not in {"phase", "seed"}}:
                raise ValueError("Reported evaluation no longer matches its raw game log")
            summary, stored = read(game / "summary.json"), read(game / "config.json")
            options = stored["options"]
            if (summary["reason"] != "step_limit" or options["mode"] != "frozen"
                    or not options["intro"] or options["seed"] != row["seed"]
                    or options.get("resume") or options.get("load_state")
                    or options["steps"] != report["evaluation_steps_per_arm"]
                    or row["weight_updates_this_evaluation"] != 0
                    or asdict(ExperimentConfig.from_dict(stored["config"])) != fixed
                    or stored["rom_sha1"] != trained["rom_sha1"]):
                raise ValueError("Frozen test protocol, model, ROM or completion changed")
            retained = row["phase"] == "retained"
            if ((retained and Path(options["weights"]).resolve() != checkpoint.resolve())
                    or (not retained and options["weights"] is not None)):
                raise ValueError("Evaluation did not use its assigned original/retained weights")
            arrays, state = read_checkpoint(game / "latest-checkpoint.json")
            expected = trained_arrays["weights"] if retained else arrays["base"]
            np.testing.assert_array_equal(arrays["weights"], expected)
            lines = (game / "trajectory.jsonl").read_text().splitlines()
            raw = [json.loads(line) for line in lines]
            if ([r["sample"] for r in raw] != list(range(1, report["evaluation_steps_per_arm"] + 1))
                    or state["experiment"]["sample"] != report["evaluation_steps_per_arm"]
                    or state["experiment"]["mode"] != "frozen"):
                raise ValueError("Incomplete evaluation trajectory/checkpoint")
            start_sha = sha256(game / "start.state")
            if starts.setdefault(row["seed"], start_sha) != start_sha:
                raise ValueError("Matched tests did not start from identical game states")
            signature = trajectory_signature(raw)
            if not retained:
                old = shared.setdefault(row["seed"], {"signature": signature, "runs": []})
                if old["signature"] != signature:
                    raise ValueError("Repeated original controls are not exact replications")
                old["runs"].append(str(game))
            panel[key] = {"run": str(game), "trajectory_sha256": sha256(game / "trajectory.jsonl"),
                          **outcomes(row)}
        if set(panel) != expected_panel:
            raise ValueError("Missing original or retained evaluation arm")
        for seed in testing:
            original, retained = panel[(seed, "original")], panel[(seed, "retained")]
            comparisons.append({"training_source_seed": source_seed, "evaluation_seed": seed,
                                "original": original, "retained": retained,
                                "tile_difference": retained["tiles"] - original["tiles"]})
        reports.append({"path": str(path), "sha256": sha256(report_file)})
    if len(sources) < 2:
        raise ValueError("Both registered independent training sources are required")
    return {"scope": __doc__, "status": "completed", "reports": reports,
            "comparisons": comparisons, "shared_original_controls": shared,
            "independent_training_sources": len(sources),
            "new_trials": False, "statistical_significance_claimed": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, nargs="+", required=True)
    args = p.parse_args()
    report = audit(args.reports)
    output = run_directory("retained-game-series-audit")
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
