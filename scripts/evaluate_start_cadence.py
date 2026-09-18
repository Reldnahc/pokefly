"""One-factor Start-cooldown test against explicitly REUSED frozen bout controls.

All actions still originate in the fixed neural decoder. No game/menu state
controls button availability. Supplied recorded Route-1 reset is a diagnostic
intervention; this is control capacity, not learned navigation.
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
    p.add_argument("--cooldown", type=float, default=5.0)
    args = p.parse_args()
    if not 1 < args.cooldown <= 60:
        p.error("Choose a finite cooldown above the 1 s baseline and <=60 s")
    source = json.loads(args.baseline_report.read_text())
    state = Path(source["state"])
    assert sha256(state) == source["state_sha256"]
    controls = [r for r in source["rows"] if r["arbitration"] == "sustained-v3"]
    assert len(controls) == len(source["seeds"]) >= 2
    assert {r["seed"] for r in controls} == set(source["seeds"])
    output = run_directory("start-cadence-comparison")
    report = {
        "scope": __doc__,
        "baseline_report": str(args.baseline_report),
        "baseline_report_sha256": sha256(args.baseline_report),
        "cooldown_seconds": args.cooldown,
        "reused_controls_not_new_trials": controls,
        "rows": [],
    }
    write_json(output / "report.json", report)
    for control in controls:
        baseline_run = Path(control["run"])
        stored = json.loads((baseline_run / "config.json").read_text())
        assert stored["options"]["mode"] == "frozen"
        assert not stored["options"]["weights"] and not stored["options"]["resume"]
        assert sha256(Path(stored["options"]["load_state"])) == source["state_sha256"]
        assert stored["options"]["seed"] == control["seed"]
        assert control["samples"] == source["steps"]
        arrays, _ = read_checkpoint(baseline_run / "latest-checkpoint.json")
        assert (arrays["weights"] == arrays["base"]).all()
        saved_config = output / f"baseline-config-{control['seed']}.json"
        write_json(saved_config, stored["config"])
        config = load_config(saved_config)
        assert config.brain.motor.start_cooldown_seconds == 1.0
        motor = replace(config.brain.motor, start_cooldown_seconds=args.cooldown)
        candidate = replace(config, brain=replace(config.brain, motor=motor))
        candidate_path = output / f"candidate-config-{control['seed']}.json"
        write_json(candidate_path, asdict(candidate))
        path = train(
            TrainOptions(
                rom=resolve_rom(None, Path.cwd()),
                device="cuda",
                seed=control["seed"],
                load_state=state,
                steps=source["steps"],
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
