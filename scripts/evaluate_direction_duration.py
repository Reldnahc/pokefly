"""Frozen fixed-controller mechanism tests, not learning or a route policy.

Compare either a fixed three-second rate trace or the motor-adaptation/neutral-
calibration candidate with validated, explicitly reused one-second controls.
No forced action, collision detector, RAM policy input or reward change.
The adaptation arm changes two coupled physiological settings, not the decoder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from evaluate_saved_gameplay import measure

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, TrainOptions, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-report", type=Path, required=True)
    p.add_argument(
        "--candidate", choices=("direction-3s", "adaptive-motor"), default="direction-3s"
    )
    args = p.parse_args()
    source = json.loads(args.baseline_report.read_text())
    controls = source["rows"]
    assert len(controls) == len({r["seed"] for r in controls}) >= 2
    root, ancestry = source, []
    for _ in range(8):
        if "state_sha256" in root:
            break
        path = Path(root["baseline_report"])
        ancestry.append({"path": str(path), "sha256": sha256(path)})
        root = json.loads(path.read_text())
    else:
        raise ValueError("Missing original starting-state identity")
    output = run_directory(
        "direction-duration-comparison"
        if args.candidate == "direction-3s"
        else "motor-adaptation-comparison"
    )
    report = {
        "scope": __doc__,
        "baseline_report": str(args.baseline_report),
        "baseline_report_sha256": sha256(args.baseline_report),
        "ancestry": ancestry,
        "state_sha256": root["state_sha256"],
        "candidate": args.candidate,
        "direction_trace_seconds": 3.0 if args.candidate == "direction-3s" else 1.0,
        "reused_controls_not_new_trials": controls,
        "rows": [],
    }
    write_json(output / "report.json", report)
    for control in controls:
        baseline = Path(control["run"])
        stored = json.loads((baseline / "config.json").read_text())
        options = stored["options"]
        assert options["mode"] == "frozen" and not options["weights"] and not options["resume"]
        assert options["seed"] == control["seed"] and options["steps"] == control["samples"]
        assert json.loads((baseline / "summary.json").read_text())["reason"] == "step_limit"
        state = Path(options["load_state"])
        assert sha256(state) == root["state_sha256"]
        arrays, _ = read_checkpoint(baseline / "latest-checkpoint.json")
        assert (arrays["weights"] == arrays["base"]).all()
        config = ExperimentConfig.from_dict(stored["config"])
        assert config.brain.motor.arbitration == "sustained-v3"
        assert config.brain.motor.direction_trace_seconds == 1
        candidate = replace(
            config,
            brain=replace(
                config.brain, motor=replace(config.brain.motor, direction_trace_seconds=3.0)
            ),
        )
        if args.candidate == "adaptive-motor":
            assert config.brain.dynamics.motor_adaptation_increment == 0
            calibration = Path("fly-data/intrinsic-neutral-adaptive-v1.npz")
            assert calibration.is_file()
            candidate = replace(
                config,
                brain=replace(
                    config.brain,
                    intrinsic_calibration=str(calibration).replace("\\", "/"),
                    dynamics=replace(
                        config.brain.dynamics,
                        motor_adaptation_increment=0.02,
                        motor_adaptation_seconds=3.0,
                    ),
                ),
            )
            report["neutral_calibration_sha256"] = sha256(calibration)
        path = train(
            TrainOptions(
                rom=resolve_rom(None, Path.cwd()),
                device="cuda",
                seed=control["seed"],
                load_state=state,
                steps=control["samples"],
                mode="frozen",
                hz=0,
                dashboard=False,
                checkpoint_every=500,
            ),
            config_override=candidate,
        )
        rows = [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]
        route = [r["telemetry"]["y"] for r in rows if r["telemetry"]["map"] == 12]
        directions = [
            next((b for b in r["buttons"] if b in ("up", "down", "left", "right")), None)
            for r in rows
        ]
        bouts, previous, length = [], None, 0
        for direction in directions + [None]:
            if direction != previous or direction is None:
                if length:
                    bouts.append(length)
                previous, length = direction, 0
            if direction is not None:
                length += 1
        row = {
            "seed": control["seed"],
            **measure(path),
            "route1_minimum_y": min(route) if route else None,
            "mean_direction_bout": sum(bouts) / len(bouts) if bouts else 0,
            "longest_direction_bout": max(bouts, default=0),
        }
        assert row["weight_updates_this_evaluation"] == 0
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(json.dumps(row), flush=True)
        if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
            raise RuntimeError("Stopped; partial result preserved")
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
