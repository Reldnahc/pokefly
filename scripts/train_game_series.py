"""Retained internal learning across full new-game attempts, then frozen evaluation.

Only the explicitly labeled intro setup is scripted. Each bounded attempt
starts a new game/novelty ledger and carries the previous brain's synapses.
No stage-specific resets, synthetic training rewards, route or policy features.
These episode resets are training interventions, not one uninterrupted playthrough.
"""

from __future__ import annotations

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/sensorimotor-dual-v1.json"))
    p.add_argument("--training-seeds", nargs="+", type=int, default=[1401, 1402, 1403, 1404])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[1501, 1502])
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--port", type=int, default=8779)
    args = p.parse_args()
    if (
        args.steps < 1
        or len(args.training_seeds) < 2
        or len(args.eval_seeds) < 2
        or set(args.training_seeds) & set(args.eval_seeds)
        or len(set(args.training_seeds + args.eval_seeds))
        != len(args.training_seeds + args.eval_seeds)
        or not 0 <= args.port <= 65535
    ):
        p.error("Positive budget, valid port, and distinct training/evaluation seeds required")
    output = run_directory("retained-game-series")
    config = output / "fixed-config.json"
    write_json(config, asdict(load_config(args.config)))
    rom = resolve_rom(None, Path.cwd())
    report = {
        "scope": __doc__,
        "config": str(config),
        "config_sha256": sha256(config),
        "training_seeds": args.training_seeds,
        "eval_seeds": args.eval_seeds,
        "steps_per_attempt": args.steps,
        "rows": [],
        "evaluation": "Original and retained weights frozen, matched intro/game/noise seeds",
    }
    write_json(output / "report.json", report)
    previous = None
    for seed in args.training_seeds:
        source = previous / "latest-checkpoint.json" if previous else None
        path = train(
            TrainOptions(
                rom=rom,
                device="cuda",
                seed=seed,
                steps=args.steps,
                hz=0,
                mode="learn",
                intro=True,
                dashboard=True,
                port=args.port,
                config=config if source is None else None,
                weights=source,
                checkpoint_every=500,
            )
        )
        row = {
            "phase": "training",
            "seed": seed,
            "initial_weights": str(source) if source else None,
            **measure(path),
        }
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(json.dumps(row), flush=True)
        if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
            raise RuntimeError(f"Series stopped; partial results and checkpoint saved: {output}")
        previous = path
    source = previous / "latest-checkpoint.json"
    report["evaluation_source"] = str(source)
    for i, seed in enumerate(args.eval_seeds):
        arms = ("original", "retained") if i % 2 == 0 else ("retained", "original")
        for arm in arms:
            path = train(
                TrainOptions(
                    rom=rom,
                    device="cuda",
                    seed=seed,
                    steps=args.steps,
                    hz=0,
                    mode="frozen",
                    intro=True,
                    dashboard=False,
                    checkpoint_every=500,
                    config=config if arm == "original" else None,
                    weights=source if arm == "retained" else None,
                )
            )
            row = {"phase": arm, "seed": seed, **measure(path)}
            assert row["weight_updates_this_evaluation"] == 0
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
            if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
                raise RuntimeError(f"Evaluation stopped; partial results saved: {output}")
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
