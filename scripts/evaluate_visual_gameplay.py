"""Matched actual-game visual-model learning/frozen trials, not proof of learning.

Every fresh pair starts with original synapses and an explicitly scripted intro.
Explicit continuations restore BOTH final game/neural states without a reset;
their --steps budget is additional, and both original source runs are retained.
Calibration is frozen, input is raw pixels, button mapping and general rewards
are unchanged. Synthetic/oracle assay weights are NEVER loaded into the game.
No route, game-state action filter, waypoint reward or action quota is used.
Retained-weight evaluation on new seeds is required after any apparent benefit.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evaluate_saved_gameplay import measure

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--learn-first", action="store_true")
    parser.add_argument(
        "--continue-from", type=Path,
        help="Completed matched report directory; --steps is ADDITIONAL per arm, no reset",
    )
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.port <= 65535:
        parser.error("Positive steps and valid dashboard port required")
    output = run_directory("visual-model-gameplay")
    config = output / "fixed-config.json"
    previous = None
    if args.continue_from:
        previous = json.loads((args.continue_from / "report.json").read_text())
        if (
            previous.get("status") != "completed" or previous["seed"] != args.seed
            or {row["mode"] for row in previous["rows"]} != {"learn", "frozen"}
            or len(previous["rows"]) != 2
        ):
            parser.error("Completed two-arm report with the same seed required")
        source_config = Path(previous["config"])
        if sha256(source_config) != previous["config_sha256"]:
            raise ValueError("Source configuration changed")
        write_json(config, json.loads(source_config.read_text()))
    else:
        write_json(config, asdict(load_config(Path("configs/visual-rate-v1.json"))))
    rom = resolve_rom(None, Path.cwd())
    report = {
        "scope": __doc__, "seed": args.seed, "steps": args.steps,
        "config": str(config), "config_sha256": sha256(config),
        "rows": [], "status": "running",
        "continuation_source": str(args.continue_from) if args.continue_from else None,
        "steps_are_additional_per_arm": bool(previous),
    }
    if previous:
        report["source_report_sha256"] = sha256(args.continue_from / "report.json")
    write_json(output / "report.json", report)
    for mode in (("learn", "frozen") if args.learn_first else ("frozen", "learn")):
        checkpoint, starting_sample = None, 0
        source_row = None
        if previous:
            source_row = next(row for row in previous["rows"] if row["mode"] == mode)
            source_path = Path(source_row["run"])
            summary = json.loads((source_path / "summary.json").read_text())
            if summary["reason"] != "step_limit":
                raise ValueError("Source arm did not complete")
            _, saved = read_checkpoint(source_path / "latest-checkpoint.json")
            if saved["experiment"]["mode"] != mode:
                raise ValueError("Continuation mode mismatch")
            starting_sample = int(saved["experiment"]["sample"])
            if starting_sample != summary["samples"]:
                raise ValueError("Source checkpoint is not its final completed sample")
            checkpoint = Path(saved["directory"])
        path = train(TrainOptions(
            rom=rom, device="cuda", seed=args.seed, steps=args.steps, mode=mode,
            config=None if checkpoint else config, intro=not bool(checkpoint),
            resume=checkpoint, hz=0, dashboard=mode == "learn",
            port=args.port, checkpoint_every=500,
        ))
        samples = [
            json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()
        ]
        route = [r["telemetry"]["y"] for r in samples if r["telemetry"]["map"] == 12]
        row = {"mode": mode, **measure(path), "route1_minimum_y": min(route) if route else None}
        row["starting_sample"] = starting_sample
        row["total_samples"] = starting_sample + row["samples"]
        if source_row:
            row["source_run"] = source_row["run"]
            row["source_checkpoint"] = str(checkpoint)
            # Counts/maps/novelty ledger in the application summary are cumulative;
            # first-event measurements below include the explicitly linked prefix.
            for key in ("first_party_count", "first_starter"):
                if source_row.get(key) is not None:
                    row[key] = source_row[key]
            ys = [y for y in (row["route1_minimum_y"], source_row.get("route1_minimum_y"))
                  if y is not None]
            row["route1_minimum_y"] = min(ys) if ys else None
            row["suffix_only_metrics"] = ["town_samples", "town_lower_edge_fraction",
                                          "weight_updates_this_evaluation", "samples"]
        if mode == "frozen":
            assert row["weight_updates_this_evaluation"] == 0
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(json.dumps(row), flush=True)
        if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
            raise RuntimeError("Stopped; partial actual-game results preserved")
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
