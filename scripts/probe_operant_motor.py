"""ROM-free internal motor-learning diagnostic, never a Pokemon reward scheme.

One constant raw image; actual fixed-decoder target button earns synthetic
feedback. Opposite targets and reversal test motor credit assignment before a
harder two-cue task. The learner only gets spikes and a scalar reward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pokefly.actions import pressed_buttons
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.runner import run_directory, write_json


def episode(
    c, frame, seed, count, *, target=None, schedule=None, enabled=False, preserve_feedback=False
):
    reward_mean = c.plasticity.reward_mean.copy() if preserve_feedback else None
    c.reset_dynamics(seed)
    if reward_mean is not None:
        c.plasticity.reward_mean[...] = reward_mean
    for _ in range(8):
        c.observe(frame)
    actions, rewards = [], []
    for index in range(count):
        action = c.choose(c.observe(frame))[0]
        reward = (
            float(target in pressed_buttons(action))
            if schedule is None and target
            else float(schedule[index])
            if schedule is not None
            else 0.0
        )
        if target or schedule is not None:
            c.reinforce(reward, enabled=enabled)
        actions.append(action)
        rewards.append(reward)
    return {"actions": actions, "rewards": rewards}


def rates(result):
    return {
        button: sum(button in pressed_buttons(a) for a in result["actions"])
        / len(result["actions"])
        for button in ("up", "down", "left", "right", "a", "b", "start")
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/sensorimotor-v1.json"))
    p.add_argument("--seed", type=int, default=501)
    p.add_argument("--training", type=int, default=512)
    p.add_argument("--initial-target", choices=("up", "down"), default="up")
    p.add_argument("--preserve-feedback", action="store_true")
    args = p.parse_args()
    output = run_directory("operant-motor-probe")
    c = InternalBrain(device="cuda", config=load_config(args.config).brain)
    baseline, state = c.snapshot()
    gray = np.full((144, 160, 3), 128, np.uint8)
    pre = episode(c, gray, args.seed + 100000, 256)
    report = {
        "protocol": "ROM-free fixed-image operant diagnostic; no action identity enters learner.",
        "config": str(args.config),
        "seed": args.seed,
        "training_per_stage": args.training,
        "initial_target": args.initial_target,
        "preserve_feedback_between_stages": args.preserve_feedback,
        "pre_rates": rates(pre),
        "pre": pre,
        "rows": [],
    }
    schedules = {}
    opposite = "down" if args.initial_target == "up" else "up"
    for arm in ("paired", "shuffled", "frozen"):
        c.restore(baseline, state, weights_only=True)
        for stage, target in enumerate((args.initial_target, opposite)):
            schedule = None
            if arm == "shuffled":
                schedule = np.random.default_rng(args.seed + stage).permutation(schedules[stage])
            elif arm == "frozen":
                schedule = schedules[stage]
            train = episode(
                c,
                gray,
                args.seed + stage * 1000,
                args.training,
                target=target if arm == "paired" else None,
                schedule=schedule,
                enabled=arm != "frozen",
                preserve_feedback=args.preserve_feedback,
            )
            if arm == "paired":
                schedules[stage] = train["rewards"]
            metrics = c.plasticity.metrics()
            learned, learned_state = c.snapshot()
            np.savez_compressed(output / f"{arm}-{stage}.npz", **learned)
            write_json(output / f"{arm}-{stage}.json", learned_state)
            test = episode(c, gray, args.seed + 100000, 256)
            row = {
                "arm": arm,
                "target": target,
                "training": train,
                "retention": test,
                "rates": rates(test),
                "plasticity": metrics,
            }
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print(arm, target, row["rates"], "weight", metrics["max_relative_change"], flush=True)
            if args.preserve_feedback:
                # The reward-free test is a separate branch, not an erasure of
                # the learner's slow reward expectation before reversal.
                c.restore(learned, learned_state)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
