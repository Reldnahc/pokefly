"""Exact real-application checkpoint branch test; not new gameplay evidence.

Continue a fixed saved checkpoint, save an intermediate boundary, then repeat
the remaining suffix from that boundary at a different wall-clock speed. Check
every sample plus all final neural arrays and reward state. Sources are read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import TrainOptions, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def rows(path):
    return [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]


def comparable(row):
    return {k: v for k, v in row.items() if k not in ("run_id", "compute_ms", "pacing")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--split", type=int, default=50)
    p.add_argument("--require-paid-active", action="store_true")
    args = p.parse_args()
    if not 0 < args.split < args.steps:
        p.error("The split must be strictly inside the continuation")
    _, saved = read_checkpoint(args.checkpoint)
    start = saved["experiment"]["sample"]
    if start % args.split:
        p.error("Checkpoint sample must align with the requested saving interval")
    source = Path(saved["directory"])
    protected = {name: sha256(source / name) for name in ("brain.npz", "state.json", "game.state")}
    output = run_directory("checkpoint-window-verification")
    options = dict(
        rom=resolve_rom(None, Path.cwd()),
        device="cuda",
        mode=saved["experiment"]["mode"],
        dashboard=False,
    )
    first = train(
        TrainOptions(
            **options,
            resume=source,
            steps=args.steps,
            hz=0,
            checkpoint_every=args.split,
        )
    )
    intermediate = list((first / "checkpoints").glob(f"step-{start + args.split:08d}-*"))
    assert len(intermediate) == 1
    _, boundary = read_checkpoint(intermediate[0])
    active = boundary["rewards"]["active"]
    if args.require_paid_active:
        assert active and active.get("paid"), (
            "Expected an already rewarded but unfinished encounter"
        )
    second = train(
        TrainOptions(
            **options,
            resume=intermediate[0],
            steps=args.steps - args.split,
            hz=40,
            checkpoint_every=0,
        )
    )
    expected, actual = rows(first)[args.split :], rows(second)
    assert [comparable(r) for r in expected] == [comparable(r) for r in actual]
    aa, sa = read_checkpoint(first / "latest-checkpoint.json")
    ab, sb = read_checkpoint(second / "latest-checkpoint.json")
    assert aa.keys() == ab.keys()
    for key in aa:
        np.testing.assert_array_equal(aa[key], ab[key], err_msg=key)
    assert sa["neural"] == sb["neural"] and sa["rewards"] == sb["rewards"]
    assert protected == {name: sha256(source / name) for name in protected}
    report = {
        "scope": __doc__,
        "source": str(source),
        "source_sha256": protected,
        "uninterrupted_branch": str(first),
        "resumed_suffix": str(second),
        "boundary_sample": start + args.split,
        "boundary_active_encounter": active,
        "matched_suffix_samples": len(actual),
        "all_neural_arrays_and_rewards_identical": True,
        "source_unchanged": True,
    }
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
