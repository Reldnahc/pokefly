"""Controlled repeated-battle learning, not autonomous whole-game progression.

Every episode explicitly resets the same recorded game state/reward history.
Pixels alone drive the fly; existing game outcome rewards alone change internal
synapses. RAM measures the episode boundary, never a choice of buttons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pokefly.checkpoint import sha256
from pokefly.emulator import RedEmulator
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.rewards import GeneralRewards, RewardHooks
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def episode(c, rom, state_path, ledger_path, seed, *, enabled, maximum, log_path, reward_config):
    mean = c.plasticity.reward_mean.copy() if enabled else None
    c.reset_dynamics(seed)
    if mean is not None:
        c.plasticity.reward_mean[...] = mean
    rewards = GeneralRewards(reward_config)
    rewards.restore(json.loads(ledger_path.read_text()))
    if rewards.active is None:
        raise ValueError("Recorded reward ledger must contain the active battle")
    events, finished = [], False
    with RedEmulator(rom) as game:
        game.load(state_path, advance=False)
        if not game.state().battle:
            raise ValueError("Recorded game state is not in a battle")
        frame = game.screen()
        with RewardHooks(game, rewards), log_path.open("x", encoding="utf-8") as log:
            for index in range(maximum):
                observation = c.observe(frame)
                action, _ = c.choose(observation)
                game.act(action, 24)
                reward, new_events = rewards.drain()
                c.reinforce(reward, enabled=enabled)
                frame = game.screen()
                events.extend(new_events)
                log.write(
                    json.dumps(
                        {
                            "sample": index + 1,
                            "action": action,
                            "action_source": "fly",
                            "reward": reward,
                            "reward_events": new_events,
                            "telemetry": game.state().telemetry(),
                        }
                    )
                    + "\n"
                )
                if rewards.active is None:
                    finished = True
                    break
        result = {
            "seed": seed,
            "decisions": index + 1,
            "finished": finished,
            "win": any(e["category"] == "battle_win" for e in events),
            "events": events,
            "final": game.state().telemetry(),
            "learning": c.plasticity.metrics(),
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/sensorimotor-bounded-v1.json"))
    p.add_argument("--training-episodes", type=int, default=8)
    p.add_argument("--maximum", type=int, default=3000)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=list(range(1001, 1009)))
    args = p.parse_args()
    if min(args.training_episodes, args.maximum) < 1 or len(args.eval_seeds) < 2:
        p.error("Positive training/budget and at least two evaluation seeds required")
    output = run_directory("controlled-battle-learning")
    config = load_config(args.config)
    c = InternalBrain(device="cuda", config=config.brain)
    initial, initial_state = c.snapshot()
    rom = resolve_rom(None, Path.cwd())
    report = {
        "protocol": "Explicit repeated recorded battle reset, NOT whole-game autonomy. "
        "Existing rewards only; no chosen-button feedback or forced actions. "
        "Fresh and retained evaluations frozen with matched starts/noise seeds.",
        "config": str(args.config),
        "state": str(args.state),
        "ledger": str(args.ledger),
        "state_sha256": sha256(args.state),
        "ledger_sha256": sha256(args.ledger),
        "training_episodes": args.training_episodes,
        "maximum_decisions": args.maximum,
        "training_seeds": list(range(901, 901 + args.training_episodes)),
        "eval_seeds": args.eval_seeds,
        "rows": [],
    }
    write_json(output / "report.json", report)
    for phase, seeds in (
        ("original", args.eval_seeds),
        ("training", report["training_seeds"]),
        ("retained", args.eval_seeds),
    ):
        if phase == "training":
            c.restore(initial, initial_state, weights_only=True)
        for seed in seeds:
            row = {
                "phase": phase,
                **episode(
                    c,
                    rom,
                    args.state,
                    args.ledger,
                    seed,
                    enabled=phase == "training",
                    maximum=args.maximum,
                    log_path=output / f"{phase}-{seed}.jsonl",
                    reward_config=config.rewards,
                ),
            }
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(phase, seed, "win", row["win"], "decisions", row["decisions"], flush=True)
            if phase == "training":
                arrays, state = c.snapshot()
                np.savez_compressed(output / f"learned-{seed}.npz", **arrays)
                write_json(output / f"learned-{seed}.json", state)
        if phase == "original":
            np.testing.assert_array_equal(c.plasticity.weights, c.plasticity.base)
        if phase == "training":
            trained_weights = c.plasticity.weights.copy()
        if phase == "retained":
            np.testing.assert_array_equal(c.plasticity.weights, trained_weights)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
