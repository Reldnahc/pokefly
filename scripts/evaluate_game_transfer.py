"""Frozen transfer of final actual-game-trained synapses to a matched game start.

Reuse validated original-weight controls explicitly, not as new trials. Only
internal learned weights may affect the policy difference. Outcome reward
timing may differ but has no forward effect in frozen evaluations. No synthetic
assay weights or intermediate best-checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from evaluate_saved_gameplay import measure

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, TrainOptions, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-runs", type=Path, nargs="+", required=True)
    p.add_argument("--baseline-report", type=Path, required=True)
    p.add_argument("--port", type=int, default=8778)
    args = p.parse_args()
    if not 0 <= args.port <= 65535:
        p.error("Invalid dashboard port")
    root = json.loads(args.baseline_report.read_text())
    controls = root["rows"]
    assert len(controls) == len({r["seed"] for r in controls}) >= 2
    ancestry = []
    for _ in range(8):
        if "state_sha256" in root:
            break
        parent = Path(root["baseline_report"])
        ancestry.append({"path": str(parent), "sha256": sha256(parent)})
        root = json.loads(parent.read_text())
    else:
        raise ValueError("Missing original reset-state identity")
    sources = []
    for path in args.source_runs:
        stored = json.loads((path / "config.json").read_text())
        options = stored["options"]
        summary = json.loads((path / "summary.json").read_text())
        assert summary["reason"] == "step_limit" and options["mode"] == "learn"
        assert options["intro"] and not any(
            options[key] for key in ("resume", "weights", "load_state")
        ), "Source must be actual fresh-intro training from original weights"
        arrays, saved = read_checkpoint(path / "latest-checkpoint.json")
        assert saved["experiment"]["sample"] == summary["samples"] == options["steps"]
        assert not np.array_equal(arrays["weights"], arrays["base"])
        checkpoint = Path(saved["directory"])
        sources.append(
            {
                "run": str(path),
                "checkpoint": str(checkpoint),
                "brain_sha256": sha256(checkpoint / "brain.npz"),
                "source_config": stored["config"],
                "training_seed": options["seed"],
            }
        )
    output = run_directory("final-game-weight-transfer")
    report = {
        "scope": __doc__,
        "sources": sources,
        "baseline_report": str(args.baseline_report),
        "baseline_report_sha256": sha256(args.baseline_report),
        "state_sha256": root["state_sha256"],
        "ancestry": ancestry,
        "reused_original_controls_not_new_trials": controls,
        "rows": [],
    }
    write_json(output / "report.json", report)
    for index, control in enumerate(controls):
        baseline = Path(control["run"])
        stored = json.loads((baseline / "config.json").read_text())
        options = stored["options"]
        assert options["mode"] == "frozen" and options["load_state"]
        assert not any(options[key] for key in ("resume", "weights", "intro"))
        assert options["seed"] == control["seed"] and options["steps"] == control["samples"]
        assert json.loads((baseline / "summary.json").read_text())["reason"] == "step_limit"
        original, _ = read_checkpoint(baseline / "latest-checkpoint.json")
        np.testing.assert_array_equal(original["weights"], original["base"])
        game_state = Path(options["load_state"])
        # Fresh state loads prime one video frame, identically in both arms.
        # Validate the PRE-prime source against its original manifest here;
        # compare actual POST-prime start.state files after the evaluation.
        assert sha256(game_state) == root["state_sha256"]
        for source in sources if index % 2 == 0 else sources[::-1]:
            assert source["training_seed"] != control["seed"]
            old = asdict(ExperimentConfig.from_dict(stored["config"]))
            new = asdict(ExperimentConfig.from_dict(source["source_config"]))
            old["rewards"].pop("timing")
            new["rewards"].pop("timing")
            assert old == new, "Forward configuration differs beyond inactive reward timing"
            checkpoint = Path(source["checkpoint"])
            learned, _ = read_checkpoint(checkpoint)
            path = train(
                TrainOptions(
                    rom=resolve_rom(None, Path.cwd()),
                    device="cuda",
                    seed=control["seed"],
                    steps=control["samples"],
                    load_state=game_state,
                    mode="frozen",
                    weights=checkpoint,
                    hz=0,
                    dashboard=True,
                    port=args.port,
                    checkpoint_every=500,
                )
            )
            assert sha256(path / "start.state") == sha256(baseline / "start.state")
            final, saved = read_checkpoint(path / "latest-checkpoint.json")
            np.testing.assert_array_equal(final["weights"], learned["weights"])
            samples = [
                json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()
            ]
            route = [r["telemetry"]["y"] for r in samples if r["telemetry"]["map"] == 12]
            active = saved["rewards"]["active"]
            row = {
                "source_run": source["run"],
                "seed": control["seed"],
                **measure(path),
                "route1_minimum_y": min(route) if route else None,
                "paid_but_unfinished_encounter": active if active and active.get("paid") else None,
            }
            assert row["weight_updates_this_evaluation"] == 0
            assert source["brain_sha256"] == sha256(checkpoint / "brain.npz")
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
            if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
                raise RuntimeError("Stopped; incomplete evaluation evidence preserved")
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
