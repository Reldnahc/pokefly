"""Recorded-playback menu occupancy audit, never a controller or new success.

Exact-ROM execution hooks time DisplayStartMenu through return to OverworldLoop.
They include nested party/item/settings pages. CPU-frame boundaries make timing
approximate to a frame; an initial menu before the first hook remains unknown.
Every sampled state and reward must reproduce the source trajectory exactly.
RAM/execution measurements are not available to any neural action selector.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.emulator import RedEmulator
from pokefly.rewards import SYMBOLS_REVISION, GeneralRewards, RewardConfig, RewardHooks
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json

# Same pinned pokered symbols/validated ROM as the existing outcome hooks.
# Use 0x0402 rather than the reward observer's 0x03ff to avoid duplicate hooks.
MENU_HOOKS = (
    ("menu", 0x2ACD, bytes.fromhex("3e04e0b8")),
    ("overworld", 0x0402, bytes.fromhex("cdaf20")),
)


class MenuClock:
    def __init__(self):
        self.frame = 0
        self.context = "unknown"
        self.frames = Counter()
        self.entries = 0

    def advance(self, frame):
        if frame < self.frame:
            raise ValueError("Replay frame clock moved backwards")
        self.frames[self.context] += frame - self.frame
        self.frame = frame

    def event(self, name, frame):
        self.advance(frame)
        if name == "menu":
            self.entries += 1
            self.context = "menu"
        elif name == "overworld":
            self.context = "not_start_menu"
        else:
            raise ValueError("Unknown measured context")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--through", type=int)
    args = p.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    if args.through is None and not (args.run / "summary.json").is_file():
        p.error("Active logs require an explicit completed --through sample")
    rows = []
    with (args.run / "trajectory.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if args.through is not None and row["sample"] > args.through:
                break
            rows.append(row)
    if not rows or (args.through is not None and rows[-1]["sample"] != args.through):
        p.error("Requested completed prefix is unavailable")
    rewards = GeneralRewards(RewardConfig(**config["config"]["rewards"]))
    if config["options"]["resume"]:
        _, saved = read_checkpoint(Path(config["options"]["resume"]))
        if saved["experiment"]["sample"] + 1 != rows[0]["sample"]:
            p.error("Starting checkpoint pointer moved")
        rewards.restore(saved["rewards"])
    clock = MenuClock()
    output = run_directory("recorded-menu-audit")
    bins = {}
    with RedEmulator(resolve_rom(None, Path.cwd())) as game:
        game.load(args.run / "start.state", advance=False)
        clock.frame = game.pyboy.frame_count
        if not config["options"]["resume"]:
            rewards.baseline(game.state())
            if game.state().battle:
                rewards.start(game.pyboy.memory)
        for name, address, signature in MENU_HOOKS:
            if bytes(game.pyboy.memory[address : address + len(signature)]) != signature:
                raise ValueError(f"Menu audit ROM signature mismatch: {name}")
        registered = []
        try:
            for name, address, _ in MENU_HOOKS:
                game.pyboy.hook_register(
                    0, address, lambda name: clock.event(name, game.pyboy.frame_count), name
                )
                registered.append(address)
            with (
                RewardHooks(game, rewards),
                (output / "measurements.jsonl").open("w", encoding="utf-8") as stream,
            ):
                for row in rows:
                    before = clock.frames.copy()
                    game.act(row["action"], config["config"]["frames"])
                    clock.advance(game.pyboy.frame_count)
                    assert game.state().telemetry() == row["telemetry"], row["sample"]
                    reward, events = rewards.drain()
                    assert (reward, events) == (row["reward"], row["reward_events"]), row["sample"]
                    measured = dict(clock.frames - before)
                    bins.setdefault((row["sample"] - 1) // 5000, Counter()).update(measured)
                    stream.write(json.dumps({"sample": row["sample"], **measured}) + "\n")
        finally:
            for address in registered:
                game.pyboy.hook_deregister(0, address)
    total = sum(clock.frames.values())
    report = {
        "scope": __doc__,
        "source_run": str(args.run),
        "symbols_revision": SYMBOLS_REVISION,
        "source_trajectory_sha256": sha256(args.run / "trajectory.jsonl")
        if args.through is None
        else None,
        "through": rows[-1]["sample"],
        "verified_samples": len(rows),
        "all_states_and_rewards_match": True,
        "menu_entries": clock.entries,
        "frames": dict(clock.frames),
        "start_menu_fraction": clock.frames["menu"] / total,
        "bins_5000_decisions": {str(k): dict(v) for k, v in bins.items()},
        "timing_caveat": "Frame-boundary approximation; not_start_menu includes battles/dialogue",
    }
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
