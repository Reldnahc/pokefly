"""Replay-only measurement of faint/victory/capture to delivered reward delays.

No reward or policy changes. Uses the existing validated execution hooks and
asserts every recorded state and reward. Results describe causal-credit timing,
not new autonomous battles or invented intermediate progress rewards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.emulator import RedEmulator
from pokefly.rewards import GeneralRewards, RewardConfig, RewardHooks
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


class TimedRewards(GeneralRewards):
    def __init__(self, config, frame_clock):
        super().__init__(config)
        self.frame_clock = frame_clock
        self.sample = 0
        self.history = {}
        self.outcomes = []

    def event(self, name, memory):
        encounter = self.active
        pending = len(self.pending)
        super().event(name, memory)
        if name == "start":
            encounter = self.active
        if encounter is None or name == "overworld":
            return
        history = self.history.setdefault(encounter["id"], [])
        history.append({"hook": name, "frame": self.frame_clock(), "sample": self.sample})
        events = self.pending[pending:]
        if name == "end" and events:
            self.outcomes.append(
                {"encounter": encounter["id"], "events": events, "history": list(history)}
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    if not (args.run / "summary.json").exists():
        p.error("A completed source run is required")
    config = json.loads((args.run / "config.json").read_text())
    output = run_directory("recorded-outcome-latency")
    with RedEmulator(
        resolve_rom(None, Path.cwd()),
        button_timing=config["config"].get("button_timing", "simultaneous-v1"),
    ) as game:
        game.load(args.run / "start.state", advance=False)
        rewards = TimedRewards(
            RewardConfig(**config["config"]["rewards"]), lambda: game.pyboy.frame_count
        )
        rows = [
            json.loads(line) for line in (args.run / "trajectory.jsonl").read_text().splitlines()
        ]
        if config["options"]["resume"]:
            _, saved = read_checkpoint(Path(config["options"]["resume"]))
            if saved["experiment"]["sample"] + 1 != rows[0]["sample"]:
                p.error("Starting checkpoint pointer moved")
            rewards.restore(saved["rewards"])
        else:
            rewards.baseline(game.state())
            if game.state().battle:
                rewards.start(game.pyboy.memory)
        with RewardHooks(game, rewards):
            for row in rows:
                rewards.sample = row["sample"]
                game.act(row["action"], config["config"]["frames"])
                assert game.state().telemetry() == row["telemetry"], row["sample"]
                assert rewards.drain() == (row["reward"], row["reward_events"]), row["sample"]
    results = []
    for outcome in rewards.outcomes:
        end = outcome["history"][-1]
        markers = [
            h for h in outcome["history"] if h["hook"] in ("faint", "trainer_win", "ball_done")
        ]
        results.append(
            {
                **outcome,
                "delays": [
                    {
                        "from_hook": h["hook"],
                        "game_frames": end["frame"] - h["frame"],
                        "decisions": end["sample"] - h["sample"],
                        "neural_seconds": (end["sample"] - h["sample"])
                        * config["config"]["brain"]["brain_steps"]
                        * 0.02,
                    }
                    for h in markers
                ],
            }
        )
    report = {
        "scope": __doc__,
        "source_run": str(args.run),
        "trajectory_sha256": sha256(args.run / "trajectory.jsonl"),
        "verified_samples": len(rows),
        "all_sampled_states_and_rewards_match": True,
        "neural_dt_seconds": 0.02,
        "outcomes": results,
    }
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
