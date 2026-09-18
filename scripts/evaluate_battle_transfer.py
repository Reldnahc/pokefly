"""Fresh-game transfer of actual battle-trained internal weights, frozen evaluation.

Practice resets are explicit interventions, not an uninterrupted playthrough.
No synthetic assay weights, forced route or learned adapter. Validated original
full-game controls are REUSED and not counted as additional trials. Different
outcome timing is recorded but cannot change a frozen policy's forward dynamics.
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
    p.add_argument("--training-report", type=Path, required=True)
    p.add_argument("--baseline-report", type=Path, required=True)
    p.add_argument("--port", type=int, default=8779)
    args = p.parse_args()
    if not 0 <= args.port <= 65535:
        p.error("Invalid dashboard port")
    training = json.loads(args.training_report.read_text())
    learned = [r for r in training["rows"] if r["phase"] == "training"]
    assert [r["seed"] for r in learned] == training["training_seeds"]
    assert all(r["finished"] for r in learned)
    checkpoint = Path(learned[-1]["checkpoint"])
    arrays, saved = read_checkpoint(checkpoint)
    config = ExperimentConfig.from_dict(saved["experiment"]["config"], checkpoint=True)
    source = json.loads(args.baseline_report.read_text())
    controls = [r for r in source["rows"] if r["phase"] == "original"]
    assert len(controls) == len({r["seed"] for r in controls}) >= 2
    assert {r["seed"] for r in controls} == set(source["eval_seeds"])
    assert not set(training["training_seeds"]) & set(source["eval_seeds"])
    output = run_directory("battle-to-game-transfer")
    report = {
        "scope": __doc__,
        "training_report": str(args.training_report),
        # The producer may still be completing separate frozen evaluation rows.
        "training_rows_snapshot": learned,
        "source_checkpoint": str(checkpoint),
        "source_brain_sha256": sha256(checkpoint / "brain.npz"),
        "baseline_report": str(args.baseline_report),
        "baseline_report_sha256": sha256(args.baseline_report),
        "reused_original_controls_not_new_trials": controls,
        "rows": [],
    }
    write_json(output / "report.json", report)
    for control in controls:
        baseline = Path(control["run"])
        record = json.loads((baseline / "config.json").read_text())
        options = record["options"]
        assert options["mode"] == "frozen" and options["intro"]
        assert not options["weights"] and not options["resume"] and not options["load_state"]
        assert options["seed"] == control["seed"] and options["steps"] == control["samples"]
        assert json.loads((baseline / "summary.json").read_text())["reason"] == "step_limit"
        original_arrays, _ = read_checkpoint(baseline / "latest-checkpoint.json")
        np.testing.assert_array_equal(original_arrays["weights"], original_arrays["base"])
        old, new = asdict(ExperimentConfig.from_dict(record["config"])), asdict(config)
        old["rewards"].pop("timing")
        new["rewards"].pop("timing")
        assert old == new, "Frozen original differs beyond reward delivery timing"
        path = train(
            TrainOptions(
                rom=resolve_rom(None, Path.cwd()),
                device="cuda",
                seed=control["seed"],
                steps=control["samples"],
                intro=True,
                mode="frozen",
                weights=checkpoint,
                hz=0,
                dashboard=True,
                port=args.port,
                checkpoint_every=500,
            )
        )
        assert sha256(path / "start.state") == sha256(baseline / "start.state")
        final_arrays, final = read_checkpoint(path / "latest-checkpoint.json")
        np.testing.assert_array_equal(final_arrays["weights"], arrays["weights"])
        row = {"seed": control["seed"], **measure(path)}
        active = final["rewards"]["active"]
        row["paid_but_unfinished_encounter"] = active if active and active.get("paid") else None
        assert row["weight_updates_this_evaluation"] == 0
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(json.dumps(row), flush=True)
        if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
            raise RuntimeError("Stopped; partial transfer evidence preserved")
    assert report["source_brain_sha256"] == sha256(checkpoint / "brain.npz")
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
