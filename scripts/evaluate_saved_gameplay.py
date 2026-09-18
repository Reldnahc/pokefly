"""Held-out, frozen gameplay: saved internal weights versus original weights.

Both arms use the saved circuit configuration and identical initial game states
and noise seeds. No synthetic button rewards, forced route, or further learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import TrainOptions, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def measure(path):
    summary = json.loads((path / "summary.json").read_text())
    rows = [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]
    town = [r for r in rows if r["telemetry"]["map"] == 0 and not r["telemetry"]["battle"]]
    return {
        "run": str(path),
        "samples": summary["new_samples"],
        "tiles": summary["tiles"],
        "maps": summary["maps"],
        "reward": summary["reward"],
        "reward_counts": summary["reward_counts"],
        "buttons": summary["button_counts"],
        "first_party_count": next((r["sample"] for r in rows if r["telemetry"]["party"]), None),
        # The game increments party count BEFORE completing the nickname screen
        # and initializing the Pokemon. A zero-level entry is not a starter yet.
        "first_starter": next(
            (r["sample"] for r in rows if r["telemetry"]["party"] and r["telemetry"]["levels"]),
            None,
        ),
        "final_state": summary["final_state"],
        "town_samples": len(town),
        "town_lower_edge_fraction": sum(r["telemetry"]["y"] >= 16 for r in town) / len(town)
        if town
        else None,
        "first_house_exit": summary["first_house_exit"],
        "weight_updates_this_evaluation": sum(r["learning"]["changed_this_reward"] for r in rows),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument("--load-state", type=Path)
    start.add_argument(
        "--intro", action="store_true", help="Explicit fresh-game intro intervention"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[701, 702, 703])
    p.add_argument("--steps", type=int, default=6000)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2 or args.steps < 1:
        p.error("At least two DISTINCT seeds and a positive evaluation budget are required")
    pointer = args.source_run / "latest-checkpoint.json"
    _, state = read_checkpoint(pointer)
    source = json.loads((args.source_run / "config.json").read_text())
    summary = json.loads((args.source_run / "summary.json").read_text())
    if (
        source["options"]["mode"] != "learn" or state["experiment"]["mode"] != "learn"
        or summary["reason"] != "step_limit"
        or state["experiment"]["sample"] != summary["samples"]
    ):
        raise ValueError("Retention requires the final checkpoint of a completed learning run")
    if source["options"]["seed"] in args.seeds:
        raise ValueError("Held-out seeds must not include the source training launch seed")
    checkpoint = Path(state["directory"])  # Pin it; never follow a moving latest pointer.
    output = run_directory("retained-gameplay")
    # Materialize the exact saved model settings, not whatever a named profile
    # might contain after future edits. This is a new experimental artifact.
    config = output / "saved-config.json"
    write_json(config, state["experiment"]["config"])
    report = {
        "source_run": str(args.source_run),
        "source_checkpoint": state["directory"],
        "source_brain_sha256": sha256(checkpoint / "brain.npz"),
        "source_completed_sample": summary["samples"],
        "start_state": str(args.load_state) if args.load_state else None,
        "start_state_sha256": sha256(args.load_state) if args.load_state else None,
        "intro_intervention": args.intro,
        "seeds": args.seeds,
        "steps_per_trial": args.steps,
        "protocol": "Both arms frozen; matched game state/noise, different internal weights only",
        "caveat": "Town-edge occupancy includes stationary menu/dialogue time; not collision time",
        "rows": [],
        "status": "running",
    }
    write_json(output / "report.json", report)
    for index, seed in enumerate(args.seeds):
        arms = ("original", "retained") if index % 2 == 0 else ("retained", "original")
        for arm in arms:
            path = train(
                TrainOptions(
                    rom=resolve_rom(None, Path.cwd()),
                    device="cuda",
                    seed=seed,
                    load_state=args.load_state,
                    intro=args.intro,
                    steps=args.steps,
                    mode="frozen",
                    hz=0,
                    dashboard=False,
                    checkpoint_every=0,
                    config=config if arm == "original" else None,
                    weights=checkpoint if arm == "retained" else None,
                )
            )
            row = {"arm": arm, "seed": seed, **measure(path)}
            assert row["weight_updates_this_evaluation"] == 0
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
            if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
                raise RuntimeError("Stopped; incomplete retention panel preserved")
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
