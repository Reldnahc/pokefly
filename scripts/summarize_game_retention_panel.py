"""Verify and combine fresh-game panels, counting each shared control once.

Requires both training sources and at least two COMPLETE held-out seeds. All
registered arms are included; no checkpoint selection or significance claim.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from evaluate_game_retention_panel import validate_sources
from evaluate_saved_gameplay import measure
from summarize_game_series import outcomes, read
from train_game_series import completed_game_source

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, load_config
from pokefly.runner import run_directory, write_json


def audit(paths):
    protocol, sources, expected = None, None, None
    panels, starts, evidence = {}, {}, []
    for path in paths:
        report_file = path / "report.json"
        report = read(report_file)
        if report.get("status") != "completed" or not report.get("one_shared_original_per_seed"):
            raise ValueError("Complete shared-control retention panels required")
        if not report.get("intro_intervention"):
            raise ValueError("Full fresh-game tests required, not stage-specific starts")
        fixed = asdict(load_config(Path(report["config"])))
        if sha256(Path(report["config"])) != report["config_sha256"]:
            raise ValueError("Panel configuration changed")
        current_sources = [completed_game_source(Path(s["run"])) for s in report["sources"]]
        if [s[2] for s in current_sources] != report["sources"]:
            raise ValueError("A registered training source changed")
        if validate_sources(current_sources, report["seeds"]) != fixed:
            raise ValueError("Panel and training model differ")
        current = (fixed, report["steps_per_arm"], report["sources"])
        if protocol is not None and current != protocol:
            raise ValueError("Cannot combine different training sources or evaluation protocols")
        protocol, sources = current, current_sources
        expected = {"original": None, **{
            f"retained_{s[2]['source_launch_seed']}": s for s in sources
        }}
        declared = {(seed, arm) for seed in report["seeds"] for arm in expected}
        found = set()
        source_arrays = {arm: read_checkpoint(source[0])[0]["weights"]
                         for arm, source in expected.items() if source is not None}
        for row in report["rows"]:
            key = row["seed"], row["arm"]
            if key not in declared or key in found or key in panels:
                raise ValueError("Duplicate or unregistered evaluation arm")
            found.add(key)
            seed, arm = key
            source = expected[arm]
            game = Path(row["run"])
            actual = measure(game)
            if actual != {k: row[k] for k in actual}:
                raise ValueError("Recorded outcome does not match raw gameplay")
            stored, summary = read(game / "config.json"), read(game / "summary.json")
            options = stored["options"]
            if (summary["reason"] != "step_limit" or actual["samples"] != report["steps_per_arm"]
                    or options["mode"] != "frozen" or options["seed"] != seed
                    or options["steps"] != report["steps_per_arm"] or not options["intro"]
                    or options.get("load_state") or options.get("resume")
                    or actual["weight_updates_this_evaluation"] != 0
                    or asdict(ExperimentConfig.from_dict(stored["config"])) != fixed
                    or stored["rom_sha1"] != sources[0][2]["rom_sha1"]):
                raise ValueError("Fresh frozen test protocol, completion or model changed")
            if ((source is None and options["weights"] is not None)
                    or (source is not None
                        and Path(options["weights"]).resolve() != source[0].resolve())):
                raise ValueError("Evaluation used the wrong assigned weights")
            arrays, saved = read_checkpoint(game / "latest-checkpoint.json")
            np.testing.assert_array_equal(
                arrays["weights"], arrays["base"] if source is None else source_arrays[arm]
            )
            final = Path(saved["directory"])
            if (final.resolve() != Path(row["final_checkpoint"]).resolve()
                    or sha256(final / "brain.npz") != row["final_brain_sha256"]
                    or saved["experiment"]["mode"] != "frozen"
                    or saved["experiment"]["sample"] != report["steps_per_arm"]):
                raise ValueError("Final frozen checkpoint changed")
            lines = (game / "trajectory.jsonl").read_text().splitlines()
            raw = [json.loads(line) for line in lines]
            if [r["sample"] for r in raw] != list(range(1, report["steps_per_arm"] + 1)):
                raise ValueError("Incomplete raw trajectory")
            start = sha256(game / "start.state")
            if start != row["start_state_sha256"] or starts.setdefault(seed, start) != start:
                raise ValueError("Matched arms have different initial game states")
            panels[key] = {"run": str(game), **outcomes(actual),
                           "trajectory_sha256": sha256(game / "trajectory.jsonl")}
        if found != declared:
            raise ValueError("A registered test arm is missing")
        evidence.append({"path": str(path), "sha256": sha256(report_file)})
    seeds = sorted({seed for seed, _ in panels})
    if len(seeds) < 2:
        raise ValueError("At least two complete held-out seeds required; one is a partial panel")
    comparisons = [{"source_launch_seed": source[2]["source_launch_seed"], "seed": seed,
                    "original": panels[(seed, "original")], "retained": panels[(seed, arm)]}
                   for seed in seeds for arm, source in expected.items() if source is not None]
    aggregate = {}
    for arm in expected:
        rows = [panels[(seed, arm)] for seed in seeds]
        aggregate[arm] = {
            "trials": len(rows), "starters": sum(r["starter"] for r in rows),
            "trials_with_battle_win": sum(r["battle_wins"] > 0 for r in rows),
            "battle_wins": sum(r["battle_wins"] for r in rows),
            "trials_reaching_route1": sum(r["route1_visited"] for r in rows),
            "positions": [r["tiles"] for r in rows],
        }
    return {"scope": __doc__, "status": "completed", "evidence": evidence,
            "seeds": seeds, "sources": protocol[2], "comparisons": comparisons,
            "aggregate": aggregate, "new_trials": False,
            "shared_original_controls": len(seeds), "statistical_significance_claimed": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, nargs="+", required=True)
    args = p.parse_args()
    report = audit(args.reports)
    output = run_directory("game-retention-panel-audit")
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
