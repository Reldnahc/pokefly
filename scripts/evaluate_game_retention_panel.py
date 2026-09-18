"""Frozen fresh-game tests of multiple actual-game-trained brains.

One original-weight control per held-out seed, shared by ALL trained sources.
No new training, selected checkpoints, synthetic weights or stage-specific
starts. A single-seed report is explicitly a partial panel, not confirmation.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from evaluate_saved_gameplay import measure
from train_game_series import completed_game_source

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import ExperimentConfig, TrainOptions, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def validate_sources(sources, seeds):
    if len(sources) < 2 or len({str(s[0].resolve()) for s in sources}) != len(sources):
        raise ValueError("At least two distinct completed actual-game sources required")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Distinct held-out seeds required")
    launch = [s[2]["source_launch_seed"] for s in sources]
    if len(set(launch)) != len(launch) or set(launch) & set(seeds):
        raise ValueError("Independent training sources and disjoint test seeds required")
    models = [asdict(ExperimentConfig.from_dict(s[1], checkpoint=True)) for s in sources]
    if any(model != models[0] for model in models):
        raise ValueError("Sources must share one exact saved model configuration")
    if len({s[2]["rom_sha1"] for s in sources}) != 1:
        raise ValueError("Training sources used different ROMs")
    return models[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-runs", type=Path, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=6000)
    args = p.parse_args()
    if args.steps < 1:
        p.error("Positive preselected evaluation budget required")
    sources = [completed_game_source(path) for path in args.source_runs]
    fixed = validate_sources(sources, args.seeds)
    output = run_directory("game-retention-panel")
    config = output / "fixed-config.json"
    write_json(config, fixed)
    report = {
        "scope": __doc__, "status": "running", "seeds": args.seeds, "steps_per_arm": args.steps,
        "single_seed_partial_panel": len(args.seeds) == 1,
        "sources": [source[2] for source in sources],
        "config": str(config), "config_sha256": sha256(config), "rows": [],
        "one_shared_original_per_seed": True, "intro_intervention": True,
        "protocol": "Original and ALL final actual-game weights frozen; matched fresh intro/noise",
    }
    write_json(output / "report.json", report)
    print("Panel:", output, flush=True)
    arrays = {str(source[0]): read_checkpoint(source[0])[0]["weights"] for source in sources}
    choices = [("original", None)] + [
        (f"retained_{source[2]['source_launch_seed']}", source[0]) for source in sources
    ]
    for seed in args.seeds:
        # Order is determined by the declared seed, not by observed outcomes.
        ordered = choices if seed % 2 else list(reversed(choices))
        start_sha = None
        for arm, checkpoint in ordered:
            path = train(TrainOptions(
                rom=resolve_rom(None, Path.cwd()), device="cuda", seed=seed, steps=args.steps,
                mode="frozen", hz=0, intro=True, dashboard=False, checkpoint_every=500,
                config=config if checkpoint is None else None, weights=checkpoint,
            ))
            result = measure(path)
            row = {"arm": arm, "seed": seed, **result}
            if result["weight_updates_this_evaluation"] != 0:
                raise ValueError("Evaluation unexpectedly changed neural weights")
            final, saved = read_checkpoint(path / "latest-checkpoint.json")
            if saved["experiment"]["mode"] != "frozen":
                raise ValueError("Evaluation checkpoint is not frozen")
            expected = final["base"] if checkpoint is None else arrays[str(checkpoint)]
            np.testing.assert_array_equal(final["weights"], expected)
            current_start = sha256(path / "start.state")
            if start_sha is not None and current_start != start_sha:
                raise ValueError("Matched arms did not start from identical game states")
            start_sha = current_start
            row["start_state_sha256"] = current_start
            row["final_checkpoint"] = saved["directory"]
            row["final_brain_sha256"] = sha256(Path(saved["directory"]) / "brain.npz")
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
            summary = json.loads((path / "summary.json").read_text())
            if summary["reason"] != "step_limit" or result["samples"] != args.steps:
                raise RuntimeError("Panel interrupted; partial evidence and checkpoints preserved")
    for checkpoint, _, provenance in sources:
        if sha256(checkpoint / "brain.npz") != provenance["brain_sha256"]:
            raise ValueError("Source weights changed during the retention test")
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
