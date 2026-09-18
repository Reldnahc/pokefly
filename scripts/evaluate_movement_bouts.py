"""Matched frozen control: short button pulses versus decaying neural walking bouts.

A supplied recorded game state is an explicit diagnostic reset. No route,
forced buttons, RAM policy input or learned adapter. This tests control capacity,
NOT reward learning; same original neural weights/noise in each paired arm.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from evaluate_saved_gameplay import measure

from pokefly.checkpoint import sha256
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1201, 1202])
    p.add_argument("--steps", type=int, default=6000)
    args = p.parse_args()
    if len(args.seeds) < 2 or args.steps < 1 or not args.state.is_file():
        p.error("Recorded state, at least two seeds and positive budget required")
    output = run_directory("movement-bout-comparison")
    cfg = load_config(Path("configs/sensorimotor-dual-v1.json"))
    report = {
        "protocol": __doc__, "state": str(args.state), "state_sha256": sha256(args.state),
        "seeds": args.seeds, "steps": args.steps, "rows": [],
    }
    write_json(output / "report.json", report)
    for i, seed in enumerate(args.seeds):
        modes = ("parallel-v2", "sustained-v3")
        for mode in modes if i % 2 == 0 else modes[::-1]:
            config = replace(cfg, brain=replace(cfg.brain, motor=replace(
                cfg.brain.motor, arbitration=mode, direction_trace_seconds=1.0,
            )))
            path = train(
                TrainOptions(
                    rom=resolve_rom(None, Path.cwd()), device="cuda", seed=seed,
                    load_state=args.state, steps=args.steps,
                    mode="frozen", hz=0, dashboard=False, checkpoint_every=500,
                ),
                config_override=config,
            )
            rows = [
                json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()
            ]
            route = [r for r in rows if r["telemetry"]["map"] == 12]
            directions = [next((b for b in r["buttons"] if b in ("up", "down", "left", "right")),
                               None) for r in rows]
            bouts, current, length = [], None, 0
            for direction in directions + [None]:
                if direction != current or direction is None:
                    if length:
                        bouts.append(length)
                    current, length = direction, 0
                if direction is not None:
                    length += 1
            row = {
                "seed": seed, "arbitration": mode, **measure(path),
                "route1_minimum_y": min((r["telemetry"]["y"] for r in route), default=None),
                "mean_direction_bout": sum(bouts) / len(bouts) if bouts else 0,
                "longest_direction_bout": max(bouts, default=0),
            }
            assert row["weight_updates_this_evaluation"] == 0
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
