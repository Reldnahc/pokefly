"""One-factor serial-delivery test against REUSED, validated frozen controls.

Recorded Route-1 reset is a diagnostic intervention, not a training curriculum.
Only delivery timing changes; original weights, noise, circuit, decoder, frame
budget and rewards are identical. No game state enters the controller.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from evaluate_saved_gameplay import measure

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-report", type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.baseline_report.read_text())
    controls = source["rows"]
    assert len(controls) >= 2 and len({r["seed"] for r in controls}) == len(controls)
    output = run_directory("button-timing-comparison")
    report = {
        "scope": __doc__,
        "baseline_report": str(args.baseline_report),
        "baseline_report_sha256": sha256(args.baseline_report),
        "reused_controls_not_new_trials": controls,
        "rows": [],
    }
    write_json(output / "report.json", report)
    for control in controls:
        baseline_run = Path(control["run"])
        stored = json.loads((baseline_run / "config.json").read_text())
        assert stored["options"]["mode"] == "frozen"
        assert not stored["options"]["weights"] and not stored["options"]["resume"]
        assert stored["options"]["seed"] == control["seed"]
        assert control["samples"] == stored["options"]["steps"]
        assert json.loads((baseline_run / "summary.json").read_text())["reason"] == "step_limit"
        arrays, _ = read_checkpoint(baseline_run / "latest-checkpoint.json")
        assert (arrays["weights"] == arrays["base"]).all()
        state = Path(stored["options"]["load_state"])
        # Previous comparison validates this file against the original bout
        # protocol; repeat the transitive identity check before reusing its arm.
        original = json.loads(Path(source["baseline_report"]).read_text())
        assert sha256(state) == original["state_sha256"]
        saved_config = output / f"baseline-config-{control['seed']}.json"
        write_json(saved_config, stored["config"])
        config = load_config(saved_config)
        assert config.button_timing == "simultaneous-v1"
        candidate = replace(config, button_timing="serial-v2")
        candidate_path = output / f"candidate-config-{control['seed']}.json"
        write_json(candidate_path, asdict(candidate))
        path = train(
            TrainOptions(
                rom=resolve_rom(None, Path.cwd()),
                device="cuda",
                seed=control["seed"],
                load_state=state,
                steps=control["samples"],
                config=candidate_path,
                mode="frozen",
                hz=0,
                dashboard=False,
                checkpoint_every=500,
            )
        )
        row = {"seed": control["seed"], **measure(path)}
        assert row["weight_updates_this_evaluation"] == 0
        route = []
        with (path / "trajectory.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                if r["telemetry"]["map"] == 12:
                    route.append(r["telemetry"]["y"])
        row["route1_minimum_y"] = min(route) if route else None
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(json.dumps(row), flush=True)
        if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
            raise RuntimeError("Stopped; partial report/checkpoint preserved")
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
