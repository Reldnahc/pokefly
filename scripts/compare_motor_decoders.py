"""Matched frozen gameplay: only fixed button arbitration differs.

Also checks actual emulator movement with isolated human-specified pulses.
Those setup/diagnostic pulses are separate from every autonomous trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from pokefly.emulator import RedEmulator
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-state", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid-v1.json"))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=64)
    args = parser.parse_args()
    if args.steps < 1 or args.trials < 2 or not args.load_state.is_file():
        parser.error("Require a real starting state, positive steps, and at least two trials")
    rom = resolve_rom(None, Path.cwd())
    original_hash = hashlib.sha1(rom.read_bytes()).hexdigest()
    output = run_directory("motor-comparison")
    config = load_config(args.config)
    report = {
        "scope": "Frozen weights; identical starting state and matched neural seeds. "
        "Only exclusive-v1 versus parallel-v2 arbitration differs. No forced gameplay action.",
        "starting_state": str(args.load_state.resolve()),
        "state_sha256": hashlib.sha256(args.load_state.read_bytes()).hexdigest(),
        "steps_per_trial": args.steps,
        "trials": [],
    }
    for seed in range(args.seed, args.seed + args.trials):
        for arbitration in ("exclusive-v1", "parallel-v2"):
            trial_config = replace(
                config,
                brain=replace(
                    config.brain, motor=replace(config.brain.motor, arbitration=arbitration)
                ),
            )
            path = train(
                TrainOptions(
                    rom=rom,
                    load_state=args.load_state,
                    device=args.device,
                    seed=seed,
                    steps=args.steps,
                    mode="frozen",
                    dashboard=False,
                    hz=0,
                    checkpoint_every=0,
                ),
                config_override=trial_config,
            )
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            rows = [
                json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()
            ]
            positions = [
                (r["telemetry"]["map"], r["telemetry"]["x"], r["telemetry"]["y"]) for r in rows
            ]
            row = {
                "run": str(path),
                **summary,
                "forward_spikes": sum(r["groups"]["forward_Up"] for r in rows),
                "forward_active_windows": sum(r["groups"]["forward_Up"] > 0 for r in rows),
                "position_transitions": sum(
                    a != b for a, b in zip(positions, positions[1:], strict=False)
                ),
            }
            report["trials"].append(row)
            write_json(output / "report.json", report)
            if summary["reason"] != "step_limit":
                raise RuntimeError("Comparison interrupted; partial results preserved")
    # This setup is specific to the documented bedroom fixture, not a route policy.
    pulses = []
    with RedEmulator(rom) as game:
        game.load(args.load_state)
        if game.state().position == (38, 3, 6):
            for action in ("up", "up+a", "up+b", "up+start"):
                game.load(args.load_state)
                game.act("right", 24)
                game.act("wait", 60)
                before = game.state().position
                game.act(action, 24)
                game.act("wait", 60)
                after = game.state().position
                pulses.append({"command": action, "before": before, "after": after})
        else:
            report["isolated_pulse_skip"] = "Starting state is not the documented bedroom fixture"
    report["isolated_emulator_pulses"] = pulses
    report["rom_unchanged"] = hashlib.sha1(rom.read_bytes()).hexdigest() == original_hash
    assert report["rom_unchanged"]
    report["interpretation"] = (
        "Delivered Up and movement are different measurements: walls, dialogs and menus can "
        "consume inputs. More buttons/tiles do not establish intention or learned gameplay. "
        "No neural weights, thresholds, current, rewards or movement mappings were retuned."
    )
    write_json(output / "report.json", report)
    print(f"Motor comparison: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
