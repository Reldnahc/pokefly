"""Frozen fresh-frame test AFTER neutral calibration and serial button fixes.

The earlier temporal screen used the weak-Up model. Here all arms retain the
same corrected circuit and delivery schedule. Explicitly reuse the completed
serial/snapshot controls; endpoint isolates pipeline timing from fresh video.
Route-1 resets are measurement interventions, not a waypoint training policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from evaluate_saved_gameplay import measure
from evaluate_temporal import measurements

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, TrainOptions, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-report", type=Path, required=True)
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
        raise ValueError("Missing original frozen starting-state identity")
    output = run_directory("corrected-temporal-comparison")
    report = {
        "scope": __doc__,
        "baseline_report": str(args.baseline_report),
        "baseline_report_sha256": sha256(args.baseline_report),
        "control_ancestry": ancestry,
        "state_sha256": root["state_sha256"],
        "reused_snapshot_controls_not_new_trials": controls,
        "rows": [],
    }
    write_json(output / "report.json", report)
    for index, control in enumerate(controls):
        baseline = Path(control["run"])
        saved = json.loads((baseline / "config.json").read_text())
        options = saved["options"]
        state = Path(options["load_state"])
        assert sha256(state) == root["state_sha256"]
        assert options["mode"] == "frozen" and not options["weights"] and not options["resume"]
        assert options["seed"] == control["seed"] and options["steps"] == control["samples"]
        assert json.loads((baseline / "summary.json").read_text())["reason"] == "step_limit"
        arrays, _ = read_checkpoint(baseline / "latest-checkpoint.json")
        assert (arrays["weights"] == arrays["base"]).all()
        config = ExperimentConfig.from_dict(saved["config"])
        assert config.button_timing == "serial-v2" and config.visual_timing == "snapshot-v1"
        expected = [
            json.loads(line) for line in (baseline / "trajectory.jsonl").read_text().splitlines()
        ]
        modes = ("endpoint-v1", "stream-v1") if index == 0 else ("stream-v1", "endpoint-v1")
        for mode in modes:
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
                config_override=replace(config, visual_timing=mode),
            )
            actual = [
                json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()
            ]
            same = all(
                a["action"] == b["action"] and a["telemetry"] == b["telemetry"]
                for a, b in zip(expected, actual, strict=True)
            )
            if mode == "endpoint-v1":
                assert same, "Endpoint control must preserve the frozen snapshot trajectory"
            route = [r["telemetry"]["y"] for r in actual if r["telemetry"]["map"] == 12]
            row = {
                "seed": control["seed"],
                "timing": mode,
                **measure(path),
                "same_actions_and_states_as_snapshot": same,
                "route1_minimum_y": min(route) if route else None,
                "moving_input_windows": measurements(path)["moving_input_windows"],
            }
            assert row["weight_updates_this_evaluation"] == 0
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
            if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
                raise RuntimeError("Stopped; partial report preserved")
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
