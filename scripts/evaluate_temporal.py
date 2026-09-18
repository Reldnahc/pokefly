"""Matched temporal-input trials; no parameter fitting or game reward changes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from pokefly.checkpoint import read_checkpoint
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import write_json


def measurements(path):
    rows = [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]
    n = len(rows)
    states = [r["telemetry"] for r in rows]
    positions = Counter((s["map"], s["x"], s["y"]) for s in states)
    summary = json.loads((path / "summary.json").read_text())
    return {
        "run": str(path),
        "samples": n,
        "button_counts": summary["button_counts"],
        "reward": summary["reward"],
        "reward_counts": summary["reward_counts"],
        "distinct_positions": len(positions),
        "most_common_position_fraction": positions.most_common(1)[0][1] / n,
        "bedroom_bottom_row_fraction": sum(s["map"] == 38 and s["y"] == 7 for s in states) / n,
        "town_bottom_two_rows_fraction": sum(s["map"] == 0 and s["y"] >= 16 for s in states) / n,
        "up_active_fraction": sum(r["groups"]["forward_Up"] > 0 for r in rows) / n,
        "population_mean_hz": summary["population_mean_hz"],
        "plasticity": summary["plasticity"],
        "house_exit": summary["house_exit"],
        "maps": summary["maps"],
        "moving_input_windows": sum(
            r.get("input_window", {}).get("unique_frames", 1) > 1 for r in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    args = parser.parse_args()
    if args.steps < 1 or len(args.seeds) < 2 or not args.output.is_dir():
        parser.error("Require positive steps, >=2 seeds, and an existing research directory")
    _, saved = read_checkpoint(args.source_run / "latest-checkpoint.json")
    states = {
        "bedroom": args.source_run / "start.state",
        "pallet": Path(saved["directory"]) / "game.state",
    }
    config = load_config(Path("configs/sensory-isolated-v1.json"))
    report = {
        "protocol": "Frozen original weights; same game starts, seeds, button mappings and "
        "rewards. Snapshot legacy versus pipelined endpoint versus interleaved raw frames. "
        "Endpoint and stream have identical neural budgets including one initial window.",
        "seeds": args.seeds,
        "steps": args.steps,
        "rows": [],
        "learned_gameplay_demonstrated": False,
    }
    for name, state in states.items():
        for seed in args.seeds:
            for timing in ("snapshot-v1", "endpoint-v1", "stream-v1"):
                path = train(
                    TrainOptions(
                        rom=resolve_rom(None, Path.cwd()),
                        load_state=state,
                        device="cuda",
                        seed=seed,
                        mode="frozen",
                        steps=args.steps,
                        hz=0,
                        dashboard=False,
                        checkpoint_every=0,
                    ),
                    config_override=replace(config, visual_timing=timing),
                )
                row = {"start": name, "seed": seed, "timing": timing, **measurements(path)}
                report["rows"].append(row)
                write_json(args.output / "temporal-gameplay.json", report)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "start",
                                "seed",
                                "timing",
                                "button_counts",
                                "distinct_positions",
                                "up_active_fraction",
                                "moving_input_windows",
                            )
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
