"""Matched actual-game visual-model learning/frozen trials, not proof of learning.

Every run starts with original synapses and an explicitly scripted intro.
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

from pokefly.checkpoint import sha256
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--learn-first", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.port <= 65535:
        parser.error("Positive steps and valid dashboard port required")
    output = run_directory("visual-model-gameplay")
    config = output / "fixed-config.json"
    write_json(config, asdict(load_config(Path("configs/visual-rate-v1.json"))))
    rom = resolve_rom(None, Path.cwd())
    report = {
        "scope": __doc__, "seed": args.seed, "steps": args.steps,
        "config": str(config), "config_sha256": sha256(config),
        "rows": [], "status": "running",
    }
    write_json(output / "report.json", report)
    for mode in (("learn", "frozen") if args.learn_first else ("frozen", "learn")):
        path = train(TrainOptions(
            rom=rom, device="cuda", seed=args.seed, steps=args.steps, mode=mode,
            config=config, intro=True, hz=0, dashboard=mode == "learn",
            port=args.port, checkpoint_every=500,
        ))
        samples = [
            json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()
        ]
        route = [r["telemetry"]["y"] for r in samples if r["telemetry"]["map"] == 12]
        row = {"mode": mode, **measure(path), "route1_minimum_y": min(route) if route else None}
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
