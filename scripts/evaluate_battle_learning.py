"""HISTORICAL battle-reset study; not the current authorized training protocol.

The user declined battle-start practice on 2026-09-18. Preserve this harness
for the existing evidence, but do not run further battle-reset training.
Use train_game_series.py for whole-game attempts carrying learned synapses.

Every episode explicitly resets the same recorded game state/reward history.
Pixels alone drive the fly; existing game outcome rewards alone change internal
synapses. RAM measures the episode boundary, never a choice of buttons.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pokefly.checkpoint import save_checkpoint, sha256
from pokefly.emulator import RedEmulator
from pokefly.experiment import ExperimentConfig, load_config
from pokefly.internal_brain import InternalBrain
from pokefly.rewards import GeneralRewards, RewardHooks
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def episode(
    c,
    rom,
    state_path,
    ledger_path,
    seed,
    *,
    enabled,
    maximum,
    log_path,
    reward_config,
    checkpoint_output=None,
    experiment_config=None,
):
    mean = c.plasticity.reward_mean.copy() if enabled else None
    c.reset_dynamics(seed)
    if mean is not None:
        c.plasticity.reward_mean[...] = mean
    rewards = GeneralRewards(reward_config)
    rewards.restore(json.loads(ledger_path.read_text()))
    if rewards.active is None:
        raise ValueError("Recorded reward ledger must contain the active battle")
    events, finished = [], False
    actions, spike_totals = Counter(), Counter()
    with RedEmulator(
        rom,
        button_timing=experiment_config.button_timing if experiment_config else "simultaneous-v1",
    ) as game:
        game.load(state_path, advance=False)
        if not game.state().battle:
            raise ValueError("Recorded game state is not in a battle")
        frame = game.screen()
        with RewardHooks(game, rewards) as hooks, log_path.open("x", encoding="utf-8") as log:
            for index in range(maximum):
                observation = c.observe(frame)
                action, _ = c.choose(observation)
                actions[action] += 1
                spike_totals.update(observation.groups)
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
            checkpoint = None
            if checkpoint_output is not None:
                checkpoint = save_checkpoint(
                    checkpoint_output,
                    c,
                    game,
                    hooks,
                    rewards,
                    frame=frame,
                    experiment={
                        "sample": index + 1,
                        "mode": "learn",
                        "config": asdict(experiment_config),
                        "actions": dict(actions),
                        "spike_totals": dict(spike_totals),
                        "visited_this_trial": [],
                        "first_house_exit": None,
                        "starts_in_house": False,
                        "intervention": "Explicit reset to recorded battle for diagnostic training",
                    },
                )
        result = {
            "seed": seed,
            "decisions": index + 1,
            "finished": finished,
            "win": any(e["category"] == "battle_win" for e in events),
            "events": events,
            "final": game.state().telemetry(),
            "learning": c.plasticity.metrics(),
            "checkpoint": str(checkpoint) if checkpoint else None,
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
    p.add_argument(
        "--initial-neural",
        type=Path,
        help="Saved neural NPZ from actual battle training; paired JSON required",
    )
    p.add_argument(
        "--reference-control",
        type=Path,
        help="Reuse an identical frozen baseline, never counted as new trials",
    )
    args = p.parse_args()
    if (
        args.training_episodes < 0
        or args.maximum < 1
        or len(args.eval_seeds) < 2
        or (args.training_episodes == 0 and args.initial_neural is None)
    ):
        p.error(
            "Positive budget, two evaluation seeds, and training or saved neural weights required"
        )
    output = run_directory("controlled-battle-learning")
    config = load_config(args.config)
    if config.frames != 24 or config.visual_timing != "snapshot-v1":
        raise ValueError("Battle assay currently implements only 24-frame snapshot timing")
    c = InternalBrain(device="cuda", config=config.brain)
    initial, initial_state = c.snapshot()
    rom = resolve_rom(None, Path.cwd())
    report = {
        "protocol": "Explicit repeated recorded battle reset, NOT whole-game autonomy. "
        "Existing rewards only; no chosen-button feedback or forced actions. "
        "Fresh and retained evaluations frozen with matched starts/noise seeds.",
        "config": str(args.config),
        "config_snapshot": asdict(config),
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
    if args.initial_neural:
        report["initial_neural"] = str(args.initial_neural)
        report["initial_neural_sha256"] = sha256(args.initial_neural)
        report["initial_neural_metadata_sha256"] = sha256(args.initial_neural.with_suffix(".json"))
    if args.reference_control:
        reference = json.loads(args.reference_control.read_text())
        old_config = asdict(
            ExperimentConfig.from_dict(reference["config_snapshot"])
            if reference.get("config_snapshot")
            else load_config(Path(reference["config"]))
        )
        current_config = asdict(config)
        # Passive eligibility traces cannot affect frozen forward dynamics. Fail closed
        # if ANY other configuration parameter or trial input differs.
        for record in (old_config, current_config):
            record["brain"]["plasticity"].pop("eligibility_seconds")
            record["brain"]["plasticity"].pop("slow_eligibility_seconds")
            record["brain"]["plasticity"].pop("trace_mixing")
        if old_config != current_config or any(
            reference[key] != report[key]
            for key in ("state_sha256", "ledger_sha256", "eval_seeds", "maximum_decisions")
        ):
            raise ValueError("Reference control differs beyond passive eligibility traces")
        originals = [r for r in reference["rows"] if r["phase"] == "original"]
        if [r["seed"] for r in originals] != args.eval_seeds:
            raise ValueError("Incomplete reference control")
        report["rows"].extend(dict(r, reused_from=str(args.reference_control)) for r in originals)
        report["reference_control_sha256"] = sha256(args.reference_control)
    write_json(output / "report.json", report)
    for phase, seeds in (
        ("original", args.eval_seeds),
        ("training", report["training_seeds"]),
        ("retained", args.eval_seeds),
    ):
        if phase == "original" and args.reference_control:
            continue
        if phase == "training":
            if args.initial_neural:
                with np.load(args.initial_neural, allow_pickle=False) as archive:
                    learned_arrays = {k: archive[k].copy() for k in archive.files}
                learned_state = json.loads(args.initial_neural.with_suffix(".json").read_text())
                c.restore(learned_arrays, learned_state, weights_only=True)
            else:
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
                    checkpoint_output=output / f"training-{seed}" if phase == "training" else None,
                    experiment_config=config,
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
