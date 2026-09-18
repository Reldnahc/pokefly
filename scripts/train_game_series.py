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

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def completed_game_source(path):
    """Pin an actual game checkpoint; ROM-free assay .npz files are not accepted."""
    stored = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    if stored["options"]["mode"] != "learn" or summary["reason"] != "step_limit":
        raise ValueError("Initial source must be a completed actual-game learning run")
    _, saved = read_checkpoint(path / "latest-checkpoint.json")
    if (
        saved["experiment"]["mode"] != "learn"
        or saved["experiment"]["sample"] != summary["samples"]
    ):
        raise ValueError("Source must use its final completed learning checkpoint")
    checkpoint = Path(saved["directory"])
    return (
        checkpoint,
        saved["experiment"]["config"],
        {
            "run": str(path),
            "checkpoint": str(checkpoint),
            "brain_sha256": sha256(checkpoint / "brain.npz"),
            "config_sha256": sha256(path / "config.json"),
            "rom_sha1": saved["rom_sha1"],
            "completed_samples": summary["samples"],
            "source_launch_seed": stored["options"]["seed"],
        },
    )


def defer_evaluation(report, output, source):
    """Pin the final trained brain; defer tests without calling the study complete."""
    if not report.get("defer_evaluation", False):
        return False
    _, saved = read_checkpoint(source)
    if (saved["experiment"]["mode"] != "learn"
            or saved["experiment"]["sample"] != report["steps_per_attempt"]):
        raise ValueError("Deferred evaluation requires the final completed training brain")
    checkpoint = Path(saved["directory"])
    report.update(status="training_completed", evaluation_pending=True,
                  evaluation_source=str(checkpoint),
                  evaluation_brain_sha256=sha256(checkpoint / "brain.npz"))
    write_json(output / "report.json", report)
    print("Training complete; shared frozen evaluation remains pending:", output, flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        help="Defaults to sensorimotor-dual-v1; incompatible with --initial-game-run",
    )
    p.add_argument(
        "--initial-game-run",
        type=Path,
        help="Continue actual-game-trained synapses across explicit new-game attempts",
    )
    p.add_argument("--training-seeds", nargs="+", type=int, default=[1401, 1402, 1403, 1404])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[1501, 1502])
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument(
        "--evaluation-steps", type=int, help="Frozen budget per evaluation arm; defaults to --steps"
    )
    p.add_argument("--port", type=int, default=8779)
    p.add_argument("--defer-evaluation", action="store_true",
                   help="Train the full sequence, then use a separate shared-control panel")
    args = p.parse_args()
    if (
        args.steps < 1
        or (args.evaluation_steps is not None and args.evaluation_steps < 1)
        or len(args.training_seeds) < (1 if args.initial_game_run else 2)
        or len(args.eval_seeds) < 2
        or set(args.training_seeds) & set(args.eval_seeds)
        or len(set(args.training_seeds + args.eval_seeds))
        != len(args.training_seeds + args.eval_seeds)
        or not 0 <= args.port <= 65535
    ):
        p.error("Positive budget, valid port, and distinct training/evaluation seeds required")
    if args.config and args.initial_game_run:
        p.error("An initial game run restores its saved configuration; omit --config")
    evaluation_steps = args.evaluation_steps or args.steps
    output = run_directory("retained-game-series")
    config = output / "fixed-config.json"
    initial, provenance = None, None
    if args.initial_game_run:
        initial, source_config, provenance = completed_game_source(args.initial_game_run)
        if provenance["source_launch_seed"] in args.eval_seeds:
            p.error("Evaluation seed overlaps the source training launch")
        write_json(config, source_config)
    else:
        profile = args.config or Path("configs/sensorimotor-dual-v1.json")
        write_json(config, asdict(load_config(profile)))
    rom = resolve_rom(None, Path.cwd())
    report = {
        "scope": __doc__,
        "config": str(config),
        "config_sha256": sha256(config),
        "training_seeds": args.training_seeds,
        "eval_seeds": args.eval_seeds,
        "steps_per_attempt": args.steps,
        "evaluation_steps_per_arm": evaluation_steps,
        "initial_actual_game_source": provenance,
        "rows": [],
        "status": "running",
        "defer_evaluation": args.defer_evaluation,
        "evaluation": "Original and retained weights frozen, matched intro/game/noise seeds",
    }
    write_json(output / "report.json", report)
    print("Series:", output, flush=True)
    previous = None
    for seed in args.training_seeds:
        source = previous / "latest-checkpoint.json" if previous else initial
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
    if defer_evaluation(report, output, source):
        return
    for i, seed in enumerate(args.eval_seeds):
        arms = ("original", "retained") if i % 2 == 0 else ("retained", "original")
        for arm in arms:
            path = train(
                TrainOptions(
                    rom=rom,
                    device="cuda",
                    seed=seed,
                    steps=evaluation_steps,
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
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
