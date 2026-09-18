"""Frozen learned-versus-original gameplay and legacy-checkpoint regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_temporal import measurements

from pokefly.checkpoint import read_checkpoint
from pokefly.experiment import TrainOptions, train
from pokefly.rom import resolve_rom
from pokefly.runner import write_json


def rows(path):
    return [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-only", action="store_true")
    args = parser.parse_args()
    rom = resolve_rom(None, Path.cwd())
    pointer = args.source_run / "latest-checkpoint.json"
    _, saved = read_checkpoint(pointer)
    # A completed earlier generation has an independently recorded continuation.
    checkpoints = sorted((args.source_run / "checkpoints").glob("step-*"))
    checkpoint = checkpoints[-2]
    _, legacy = read_checkpoint(checkpoint)
    start = int(legacy["experiment"]["sample"])
    length = min(20, int(saved["experiment"]["sample"]) - start)
    assert length > 0
    path = train(
        TrainOptions(
            rom=rom,
            device="cuda",
            resume=checkpoint,
            steps=length,
            hz=0,
            dashboard=False,
            checkpoint_every=0,
        )
    )
    expected = [r for r in rows(args.source_run) if start < r["sample"] <= start + length]
    actual = rows(path)
    ignored = {"run_id", "compute_ms", "pacing"}
    assert len(expected) == len(actual) == length
    for a, b in zip(expected, actual, strict=True):
        assert {k: v for k, v in a.items() if k not in ignored} == {
            k: v for k, v in b.items() if k not in ignored
        }, a["sample"]
    report = {
        "source_checkpoint": saved["directory"],
        "legacy_resume": {"source": str(checkpoint), "run": str(path), "exact_decisions": length},
        "rows": [],
        "learned_gameplay_demonstrated": False,
    }
    write_json(args.output / "retained-gameplay.json", report)
    if args.legacy_only:
        print("Exact legacy continuation:", report["legacy_resume"], flush=True)
        return
    baseline = json.loads((args.output / "temporal-gameplay.json").read_text())["rows"]
    for seed in (101, 102, 103):
        path = train(
            TrainOptions(
                rom=rom,
                device="cuda",
                weights=pointer,
                load_state=Path(saved["directory"]) / "game.state",
                seed=seed,
                mode="frozen",
                steps=1000,
                hz=0,
                dashboard=False,
                checkpoint_every=0,
            )
        )
        reference = next(
            r
            for r in baseline
            if r["seed"] == seed and r["start"] == "pallet" and r["timing"] == "snapshot-v1"
        )
        before, after = rows(Path(reference["run"])), rows(path)
        row = {
            "seed": seed,
            **measurements(path),
            "baseline_run": reference["run"],
            "baseline_positions": reference["distinct_positions"],
            "different_actions": sum(
                a["action"] != b["action"] for a, b in zip(before, after, strict=True)
            ),
        }
        report["rows"].append(row)
        write_json(args.output / "retained-gameplay.json", report)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "seed",
                        "distinct_positions",
                        "baseline_positions",
                        "different_actions",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
