"""Exploratory actual-game test of anatomical learning scope, not a promoted model.

Two scopes use the same original weights, perturb-v3 rule, fixed serial decoder
and general rewards. Recorded Route-1 starts are explicit reset interventions.
Validate original-weight forward prefixes before reusing existing frozen
controls; those duplicates are verification, not more independent successes.
Synthetic/oracle-trained weights are never loaded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from evaluate_saved_gameplay import measure

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-report", type=Path, required=True)
    p.add_argument("--scope", choices=("motor-inputs-v1", "premotor-one-hop-v2"), required=True)
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    if not 0 <= args.port <= 65535:
        p.error("Invalid dashboard port")
    root = json.loads(args.baseline_report.read_text())
    controls, ancestry = root["rows"], []
    assert len(controls) == len({r["seed"] for r in controls}) >= 2
    for _ in range(8):
        if "state_sha256" in root:
            break
        parent = Path(root["baseline_report"])
        ancestry.append({"path": str(parent), "sha256": sha256(parent)})
        root = json.loads(parent.read_text())
    else:
        raise ValueError("Missing reset-state identity")
    output = run_directory("anatomical-scope-gameplay")
    report = {
        "scope": __doc__,
        "plasticity_scope": args.scope,
        "baseline_report": str(args.baseline_report),
        "baseline_report_sha256": sha256(args.baseline_report),
        "ancestry": ancestry,
        "state_sha256": root["state_sha256"],
        "reused_frozen_controls_not_new_trials": controls,
        "prefix_verification": [],
        "rows": [],
    }
    write_json(output / "report.json", report)
    for control in controls:
        baseline = Path(control["run"])
        stored = json.loads((baseline / "config.json").read_text())
        options = stored["options"]
        assert options["mode"] == "frozen" and options["load_state"]
        assert not any(options[key] for key in ("weights", "resume", "intro"))
        assert options["seed"] == control["seed"] and options["steps"] == control["samples"]
        assert json.loads((baseline / "summary.json").read_text())["reason"] == "step_limit"
        arrays, _ = read_checkpoint(baseline / "latest-checkpoint.json")
        np.testing.assert_array_equal(arrays["weights"], arrays["base"])
        original = ExperimentConfig.from_dict(stored["config"])
        # Replace ONLY the inactive/active learning rule and anatomical scope.
        # This does not change firing dynamics, retina, calibration or decoder.
        plastic = load_config(Path("configs/sensorimotor-perturb-v3.json")).brain.plasticity
        candidate = replace(
            original, brain=replace(original.brain, plasticity=replace(plastic, scope=args.scope))
        )
        game_state = Path(options["load_state"])
        assert sha256(game_state) == root["state_sha256"]
        common = dict(
            rom=resolve_rom(None, Path.cwd()),
            device="cuda",
            seed=control["seed"],
            load_state=game_state,
            hz=0,
            port=args.port,
        )
        prefix = train(
            TrainOptions(**common, steps=64, mode="frozen", dashboard=False, checkpoint_every=0),
            config_override=candidate,
        )
        actual = [
            json.loads(line) for line in (prefix / "trajectory.jsonl").read_text().splitlines()
        ]
        expected = [
            json.loads(line) for line in (baseline / "trajectory.jsonl").read_text().splitlines()
        ][:64]
        keys = (
            "action",
            "buttons",
            "spikes_total",
            "groups",
            "motor_rates_hz",
            "telemetry",
            "reward",
            "reward_events",
        )
        assert len(actual) == len(expected) == 64
        assert [{k: r[k] for k in keys} for r in actual] == [
            {k: r[k] for k in keys} for r in expected
        ]
        assert sha256(prefix / "start.state") == sha256(baseline / "start.state")
        frozen, _ = read_checkpoint(prefix / "latest-checkpoint.json")
        np.testing.assert_array_equal(frozen["weights"], frozen["base"])
        report["prefix_verification"].append(
            {
                "seed": control["seed"],
                "run": str(prefix),
                "samples": 64,
                "all_selected_fields_exact": True,
            }
        )
        write_json(output / "report.json", report)
        path = train(
            TrainOptions(
                **common,
                steps=control["samples"],
                mode="learn",
                dashboard=True,
                checkpoint_every=500,
            ),
            config_override=candidate,
        )
        assert sha256(path / "start.state") == sha256(baseline / "start.state")
        samples = [
            json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()
        ]
        route = [r["telemetry"]["y"] for r in samples if r["telemetry"]["map"] == 12]
        row = {
            "seed": control["seed"],
            **measure(path),
            "route1_minimum_y": min(route) if route else None,
        }
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(json.dumps(row), flush=True)
        if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
            raise RuntimeError("Stopped; partial scope experiment preserved")
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
