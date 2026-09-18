"""Counterfactual reward-timing audit on exact recorded gameplay, not new play.

Two read-only observers see identical validated ROM hooks. Legacy rewards must
match every recorded sample; completed encounters must have identical outcomes
and amounts in the earlier observer. An unfinished encounter is reported, never
silently counted as a newly completed victory. Neither ledger supplies actions.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.emulator import RedEmulator
from pokefly.rewards import GeneralRewards, RewardConfig, RewardHooks
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


class PairedObservers:
    def __init__(self, legacy, candidate, game):
        self.ledgers = {"legacy": legacy, "candidate": candidate}
        self.game, self.sample = game, 0
        self.records = {name: [] for name in self.ledgers}
        self.completed = []

    def event(self, hook, memory):
        active = self.ledgers["legacy"].active
        encounter = active["id"] if active else None
        for name, ledger in self.ledgers.items():
            before = len(ledger.pending)
            ledger.event(hook, memory)
            self.records[name].extend(
                {
                    "sample": self.sample,
                    "frame": self.game.pyboy.frame_count,
                    "hook": hook,
                    "event": dict(event),
                }
                for event in ledger.pending[before:]
            )
        if hook == "end" and encounter is not None:
            self.completed.append(encounter)
            outcomes = {
                name: [r["event"] for r in rows if r["event"].get("encounter") == encounter]
                for name, rows in self.records.items()
            }
            assert outcomes["legacy"] == outcomes["candidate"], (encounter, outcomes)


def audit(source, rom, timing):
    config = json.loads((source / "config.json").read_text())
    if not (source / "summary.json").exists():
        raise ValueError("A completed source run is required")
    reward_config = RewardConfig(**config["config"]["rewards"])
    if reward_config.timing != "encounter-end-v1":
        raise ValueError("Source must use legacy reward timing")
    legacy = GeneralRewards(reward_config)
    candidate = GeneralRewards(replace(reward_config, timing=timing))
    rows = [json.loads(line) for line in (source / "trajectory.jsonl").read_text().splitlines()]
    with RedEmulator(
        rom, button_timing=config["config"].get("button_timing", "simultaneous-v1")
    ) as game:
        game.load(source / "start.state", advance=False)
        if config["options"]["resume"]:
            _, saved = read_checkpoint(Path(config["options"]["resume"]))
            if saved["experiment"]["sample"] + 1 != rows[0]["sample"]:
                raise ValueError("Source resume pointer moved")
            for ledger in (legacy, candidate):
                ledger.restore(copy.deepcopy(saved["rewards"]))
        else:
            for ledger in (legacy, candidate):
                ledger.baseline(game.state())
                if game.state().battle:
                    ledger.start(game.pyboy.memory)
        observers = PairedObservers(legacy, candidate, game)
        with RewardHooks(game, observers):
            for row in rows:
                observers.sample = row["sample"]
                game.act(row["action"], config["config"]["frames"])
                assert game.state().telemetry() == row["telemetry"], row["sample"]
                assert legacy.drain() == (row["reward"], row["reward_events"]), row["sample"]
                candidate.drain()
    outcomes = []
    for encounter in observers.completed:
        pair = {
            name: [r for r in records if r["event"].get("encounter") == encounter]
            for name, records in observers.records.items()
        }
        if pair["legacy"]:
            earlier = pair["legacy"][0]["sample"] - pair["candidate"][0]["sample"]
            outcomes.append({"encounter": encounter, **pair, "decisions_earlier": earlier})
            assert earlier >= 0
    # Novelty is entirely unchanged, including delivery timing.
    novelty = {
        name: [r for r in records if "encounter" not in r["event"]]
        for name, records in observers.records.items()
    }
    assert novelty["legacy"] == novelty["candidate"]
    return {
        "source_run": str(source),
        "trajectory_sha256": sha256(source / "trajectory.jsonl"),
        "samples": len(rows),
        "sampled_states_and_legacy_rewards_match": True,
        "completed_encounters": len(observers.completed),
        "completed_outcomes_identical": True,
        "novelty_delivery_identical": True,
        "outcomes": outcomes,
        "unfinished_encounter": copy.deepcopy(candidate.active),
        "total": {name: ledger.total for name, ledger in observers.ledgers.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument(
        "--timing", choices=("confirmed-outcome-v2", "last-faint-v3"), default="last-faint-v3"
    )
    args = p.parse_args()
    output = run_directory("reward-timing-audit")
    report = {
        "scope": __doc__,
        "timing": args.timing,
        "reward_implementation_sha256": sha256(Path("src/pokefly/rewards.py")),
        "rows": [],
    }
    rom = resolve_rom(None, Path.cwd())
    for source in args.runs:
        result = audit(source, rom, args.timing)
        report["rows"].append(result)
        write_json(output / "report.json", report)
        print(json.dumps(result), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
