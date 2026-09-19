"""Replay-only measurement of faint/victory/capture to delivered reward delays.

No reward or policy changes. Uses the existing validated execution hooks and
asserts every recorded state and reward. Results describe reward-delivery timing,
not new autonomous battles or invented intermediate progress rewards.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.emulator import RedEmulator
from pokefly.rewards import (
    SOURCE_REVISION,
    SYMBOLS_REVISION,
    GeneralRewards,
    RewardConfig,
    RewardHooks,
)
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
        marker = {"hook": name, "frame": self.frame_clock(), "sample": self.sample}
        if name == "end":
            # Observer-only raw state, including encounters that paid nothing.
            # Do not infer a win/loss merely from overworld return or no reward.
            marker.update(
                battle_result_raw=int(memory[0xCF0B]),
                party_alive=self._party_alive(memory),
                player_hp=256 * int(memory[0xD015]) + int(memory[0xD016]),
                enemy_hp=256 * int(memory[0xCFE6]) + int(memory[0xCFE7]),
                enemy_faint_observed=encounter["fainted"],
                trainer_victory_observed=encounter["trainer_won"],
                captured_species=encounter["captured"],
            )
        history.append(marker)
        events = self.pending[pending:]
        if events:
            self.outcomes.append(
                {"encounter": encounter["id"], "events": events, "delivered_at": marker}
            )


class MoveConfirmationHooks:
    """Observer only: successful menu selection, not a causal damage attribution.

    Pinned pokered engine/battle/core.asm, SelectMenuItem.transformedMoveSelected:
    the hook is AFTER the accepted move is written, not cursor highlighting.
    Automatic/repeated/Struggle moves need not pass this selection path.
    https://github.com/pret/pokered/blob/a1a22aaf84d1675bcdbaeb194592379d586d838e/engine/battle/core.asm
    """

    bank, block, address = 15, 0x538D, 0x539B
    signature = bytes.fromhex("fa26cc211cd04f0600097eeadcccafc9")

    def __init__(self, game, rewards):
        self.game, self.rewards = game, rewards

    def record(self, _context=None):
        rewards = self.rewards
        if rewards.active is None:
            return
        memory = self.game.pyboy.memory
        rewards.history.setdefault(rewards.active["id"], []).append({
            "hook": "move_confirmed", "frame": rewards.frame_clock(), "sample": rewards.sample,
            "move_id": int(memory[0xCCDC]), "slot": int(memory[0xCC2E]),
        })

    def __enter__(self):
        actual = bytes(self.game.pyboy.memory[
            self.bank, self.block:self.block + len(self.signature)
        ])
        if actual != self.signature:
            raise ValueError("Move-confirmation observer ROM signature mismatch")
        self.game.pyboy.hook_register(self.bank, self.address, self.record, None)
        return self

    def __exit__(self, *_args):
        self.game.pyboy.hook_deregister(self.bank, self.address)


def summarize_outcomes(rewards, brain_steps, eligibility_seconds=None):
    results = []
    for outcome in rewards.outcomes:
        delivered = outcome["delivered_at"]
        history = rewards.history[outcome["encounter"]]
        markers = [h for h in history
                   if h["hook"] in ("faint", "trainer_win", "ball_done")
                   and h["frame"] <= delivered["frame"]]
        result = {
            **outcome, "history": list(history),
            "encounter_finished": any(h["hook"] == "end" for h in history),
            "delays": [{
                "from_hook": h["hook"], "game_frames": delivered["frame"] - h["frame"],
                "decisions": delivered["sample"] - h["sample"],
                "neural_seconds": (delivered["sample"] - h["sample"]) * brain_steps * .02,
            } for h in markers],
        }
        choices = [h for h in history if h["hook"] == "move_confirmed"
                   and h["frame"] <= delivered["frame"]]
        if choices:
            choice = choices[-1]
            decisions = delivered["sample"] - choice["sample"]
            neural_seconds = decisions * brain_steps * .02
            result["last_confirmed_move"] = {
                **choice, "game_frames_to_reward": delivered["frame"] - choice["frame"],
                "decisions_to_reward": decisions, "neural_seconds_to_reward": neural_seconds,
                "confirmed_choices_before_outcome": len(choices),
                "isolated_trace_decay_fraction": math.exp(-neural_seconds / eligibility_seconds)
                if eligibility_seconds else None,
                "caveat": "Last observed choice, not proof it caused the outcome; decay excludes "
                          "new inputs and uses decision-bucket timing, not measured eligibility",
            }
        else:
            result["last_confirmed_move"] = None
        results.append(result)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--whole-game-start", action="store_true",
                   help="Reject resumed/stage-start sources; replay the full recorded game opening")
    p.add_argument("--move-confirmations", action="store_true",
                   help="Measure accepted move-to-reward delays; requires --whole-game-start")
    args = p.parse_args()
    if args.move_confirmations and not args.whole_game_start:
        p.error("Move confirmation audit requires a whole-game replay, not a battle reset")
    if not (args.run / "summary.json").exists():
        p.error("A completed source run is required")
    config = json.loads((args.run / "config.json").read_text())
    if args.whole_game_start and (not config["options"]["intro"]
                                 or config["options"].get("resume")
                                 or config["options"].get("load_state")):
        p.error("Whole-game playback requires an intro-start source, not a saved stage")
    output = run_directory("recorded-outcome-latency")
    with RedEmulator(
        resolve_rom(None, Path.cwd()),
        button_timing=config["config"].get("button_timing", "simultaneous-v1"),
    ) as game:
        game.load(args.run / "start.state", advance=False)
        if args.whole_game_start and game.state().battle:
            raise ValueError("Whole-game playback cannot start in a battle")
        rewards = TimedRewards(
            RewardConfig(**config["config"]["rewards"]), lambda: game.pyboy.frame_count
        )
        rows = [
            json.loads(line) for line in (args.run / "trajectory.jsonl").read_text().splitlines()
        ]
        if args.whole_game_start and rows[0]["sample"] != 1:
            raise ValueError("Whole-game playback must include its initial decisions")
        if config["options"]["resume"]:
            _, saved = read_checkpoint(Path(config["options"]["resume"]))
            if saved["experiment"]["sample"] + 1 != rows[0]["sample"]:
                p.error("Starting checkpoint pointer moved")
            rewards.restore(saved["rewards"])
        else:
            rewards.baseline(game.state())
            if game.state().battle:
                rewards.start(game.pyboy.memory)
        with (RewardHooks(game, rewards),
              MoveConfirmationHooks(game, rewards) if args.move_confirmations else nullcontext()):
            for row in rows:
                rewards.sample = row["sample"]
                game.act(row["action"], config["config"]["frames"])
                assert game.state().telemetry() == row["telemetry"], row["sample"]
                assert rewards.drain() == (row["reward"], row["reward_events"]), row["sample"]
    brain_config = config["config"]["brain"]
    results = summarize_outcomes(rewards, brain_config["brain_steps"],
                                brain_config["plasticity"]["eligibility_seconds"])
    report = {
        "scope": __doc__,
        "source_run": str(args.run),
        "trajectory_sha256": sha256(args.run / "trajectory.jsonl"),
        "verified_samples": len(rows),
        "all_sampled_states_and_rewards_match": True,
        "new_autonomous_trials": False,
        "neural_training": False,
        "whole_game_start_required": args.whole_game_start,
        "move_confirmations_measured": args.move_confirmations,
        "symbols_revision": SYMBOLS_REVISION,
        "source_revision": SOURCE_REVISION,
        "measurement_limit": "Hook-to-reward timing, not a measured neural credit contribution",
        "neural_dt_seconds": 0.02,
        "outcomes": results,
        "encounter_histories": [
            {"encounter": encounter, "history": history}
            for encounter, history in rewards.history.items()
        ],
    }
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
