"""Verify and illustrate an existing fly run, NOT a new autonomous experiment.

Replays the logged actions from the recorded start state. Every sampled RAM
measurement and reward event must match. Screenshots are unmodified emulator
frames; filenames/report explicitly label this as recorded playback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.emulator import RedEmulator
from pokefly.rewards import GeneralRewards, RewardConfig, RewardHooks
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", type=int, default=[])
    parser.add_argument("--save-states", action="store_true")
    parser.add_argument("--first-battle-start", action="store_true")
    parser.add_argument("--through", type=int, help="Verify a completed prefix of a running log")
    args = parser.parse_args()
    if not args.samples and not args.first_battle_start:
        parser.error("Select samples or the first confirmed battle start")
    selected = set(args.samples)
    found_start = False
    config = json.loads((args.run / "config.json").read_text())
    if args.through is None and not (args.run / "summary.json").is_file():
        parser.error("A running log requires an explicit --through sample")
    rows = []
    with (args.run / "trajectory.jsonl").open(encoding="utf-8") as log:
        for line in log:
            row = json.loads(line)
            if args.through is not None and row["sample"] > args.through:
                break
            rows.append(row)
    if not rows or (args.through is not None and rows[-1]["sample"] != args.through):
        parser.error("Requested completed prefix is unavailable")
    if not set(args.samples).issubset({r["sample"] for r in rows}):
        parser.error("Requested samples must exist in the recorded run")
    output = run_directory("recorded-replay-proof")
    rewards = GeneralRewards(RewardConfig(**config["config"]["rewards"]))
    resume = config["options"]["resume"]
    if resume:
        _, checkpoint = read_checkpoint(Path(resume))
        if int(checkpoint["experiment"]["sample"]) + 1 != rows[0]["sample"]:
            parser.error("Source checkpoint pointer moved; exact starting reward history required")
        rewards.restore(checkpoint["rewards"])
    with RedEmulator(resolve_rom(None, Path.cwd())) as game:
        game.load(args.run / "start.state", advance=False)
        if not resume:
            rewards.baseline(game.state())
            if game.state().battle:
                rewards.start(game.pyboy.memory)
        with RewardHooks(game, rewards) as hooks:
            for row in rows:
                assert row["action_source"] == "fly"
                game.act(row["action"], config["config"]["frames"])
                assert game.state().telemetry() == row["telemetry"], row["sample"]
                reward, events = rewards.drain()
                assert reward == row["reward"] and events == row["reward_events"], row["sample"]
                if args.first_battle_start and not found_start and rewards.active is not None:
                    selected.add(row["sample"])
                    found_start = True
                if row["sample"] in selected:
                    game.screenshot(output / f"recorded-sample-{row['sample']:06d}.png")
                    if args.save_states:
                        hooks.detach()
                        try:
                            game.save(output / f"recorded-sample-{row['sample']:06d}.state")
                        finally:
                            hooks.attach()
                        write_json(
                            output / f"recorded-sample-{row['sample']:06d}.rewards.json",
                            rewards.state(),
                        )
    write_json(
        output / "report.json",
        {
            "kind": "recorded replay, NOT a new autonomous success",
            "source_run": str(args.run),
            "trajectory_sha256": sha256(args.run / "trajectory.jsonl")
            if args.through is None
            else None,
            "verified_rows_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "verified_through": rows[-1]["sample"],
            "partial_run": args.through is not None,
            "continuation_reward_source": str(resume) if resume else None,
            "verified_samples": len(rows),
            "all_sampled_states_and_reward_events_match": True,
            "screenshots": sorted(selected),
        },
    )
    print("Verified recorded playback:", output, flush=True)


if __name__ == "__main__":
    main()
