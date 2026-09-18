"""Separate operant-capacity assay, NOT Pokemon shaping or a button policy.

Raw half-white images alternate in fixed blocks. Synthetic reward follows an
actual fixed-decoder Left/Right choice, with opposite assignments counterbalanced.
No action is forced; only existing internal synapses can learn. Reward-free,
fresh-noise tests and reversal measure choices, not just changed weights.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from pokefly.actions import pressed_buttons
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import write_json


def warmup(c, seed):
    c.reset_dynamics(seed)
    for _ in range(8):
        c.observe(test_patterns()["black"])


def test_choices(c, *, seed, decisions=48, first_only=False, cues=("left", "right")):
    patterns = test_patterns()
    result = []
    original = c.plasticity.weights.copy()
    for noise_seed in (seed + 100000, seed + 200000):
        for cue in cues:
            warmup(c, noise_seed)
            counts = Counter()
            for _ in range(decisions):
                action = c.choose(c.observe(patterns[cue]))[0]
                counts[action] += 1
            result.append(
                {"cue": cue, "noise_seed": noise_seed, "actions": dict(counts), "n": decisions}
            )
            if first_only:
                np.testing.assert_array_equal(original, c.plasticity.weights)
                return result
    np.testing.assert_array_equal(original, c.plasticity.weights)
    return result


def score(rows, reverse, *, buttons=("left", "right"), cues=("left", "right")):
    hits, choices, total = 0, 0, 0
    per_cue = {cue: {button: 0 for button in buttons} for cue in cues}
    for row in rows:
        target = buttons[cues.index(row["cue"]) ^ int(reverse)]
        for action, n in row["actions"].items():
            pressed = pressed_buttons(action)
            hits += n * (target in pressed)
            choices += n * bool(set(pressed) & set(buttons))
            total += n
            for button in buttons:
                per_cue[row["cue"]][button] += n * (button in pressed)
    preferences = [
        counts[buttons[0]] / sum(counts.values()) if sum(counts.values()) else None
        for counts in per_cue.values()
    ]
    contrast = (
        (preferences[0] - preferences[1]) * (-1 if reverse else 1)
        if all(value is not None for value in preferences)
        else None
    )
    return {
        "target_rate": hits / total,
        "conditional_accuracy": hits / choices if choices else None,
        "competing_choices": choices,
        **({"left_right_choices": choices} if buttons == ("left", "right") else {}),
        "decisions": total,
        # Descriptive additions only: the existing criterion-2 screen below
        # remains unchanged. A global button bias has zero cue contrast even
        # when cue-specific response rates make pooled accuracy misleading.
        "per_cue_button_counts": per_cue,
        "aligned_cue_preference_contrast": contrast,
        "balanced_conditional_accuracy": 0.5 + 0.5 * contrast if contrast is not None else None,
    }


def train_phase(
    c,
    *,
    seed,
    reverse,
    arm,
    schedule=None,
    decisions=256,
    preserve_feedback=False,
    buttons=("left", "right"),
    cues=("left", "right"),
):
    feedback_name = next(
        (name for name in ("reward_mean", "reward_expectation") if hasattr(c.plasticity, name)),
        None,
    )
    mean = (
        getattr(c.plasticity, feedback_name).copy() if preserve_feedback and feedback_name else None
    )
    warmup(c, seed)
    if mean is not None:
        getattr(c.plasticity, feedback_name)[...] = mean
    patterns = test_patterns()
    pulses, actions, presented = [], [], []
    for index in range(decisions):
        cue_index = (index // 16) % 2
        cue = cues[cue_index]
        target = buttons[cue_index ^ int(reverse)]
        action = c.choose(c.observe(patterns[cue]))[0]
        reward = (
            float(target in pressed_buttons(action)) if schedule is None else float(schedule[index])
        )
        c.reinforce(reward, enabled=arm != "frozen")
        pulses.append(reward)
        actions.append(action)
        presented.append(cue)
    return {"rewards": pulses, "actions": actions, "cues": presented, "total_reward": sum(pulses)}


def unpair(training, seed):
    # Permute within cue: preserve each cue's reward amount, remove the alignment
    # to the selected motor command. This is not a cue-value-imbalance control.
    rng = np.random.default_rng(seed)
    rewards = np.asarray(training["rewards"], float)
    cues = np.asarray(training["cues"])
    result = rewards.copy()
    for cue in np.unique(cues):
        mask = cues == cue
        result[mask] = rng.permutation(rewards[mask])
        assert result[mask].sum() == rewards[mask].sum()
    return result


def compare_arms(paired, unpaired, reverse, *, buttons=("left", "right"), cues=("left", "right")):
    """Activity increases alone cannot pass a cue-discrimination screen."""
    acquired, reversed_score = (s["score"] for s in paired["stages"])
    before_reversal = score(
        paired["stages"][0]["retention"], not reverse, buttons=buttons, cues=cues
    )
    deltas = {}
    for metric in ("target_rate", "conditional_accuracy"):
        for label, after, before in (
            ("acquisition_minus_pre", acquired, paired["pre_score"]),
            ("acquisition_minus_unpaired", acquired, unpaired["stages"][0]["score"]),
            ("reversal_minus_before_reversal", reversed_score, before_reversal),
            ("reversal_minus_unpaired", reversed_score, unpaired["stages"][1]["score"]),
        ):
            a, b = after[metric], before[metric]
            deltas[f"{metric}_{label}"] = None if a is None or b is None else a - b
    return {
        **deltas,
        "screen_pass": all(v is not None and v >= 0.05 for v in deltas.values())
        and all(
            r["conditional_accuracy"] is not None and r["conditional_accuracy"] >= 0.55
            for r in (acquired, reversed_score)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rules", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--training", type=int, default=256)
    parser.add_argument("--preserve-feedback", action="store_true")
    parser.add_argument(
        "--buttons",
        nargs=2,
        default=["left", "right"],
        choices=("up", "down", "left", "right", "a", "b", "start"),
    )
    parser.add_argument(
        "--cues", nargs=2, default=["left", "right"], choices=tuple(test_patterns())
    )
    args = parser.parse_args()
    buttons, cues = tuple(args.buttons), tuple(args.cues)
    if (
        len(set(buttons)) != 2
        or len(set(cues)) != 2
        or not any(
            set(buttons) <= channel
            for channel in ({"up", "down", "left", "right"}, {"a", "b", "start"})
        )
    ):
        parser.error("Two distinct images and two competing buttons from the same channel required")
    if len(args.seeds) < 2 or not args.output.is_dir():
        parser.error("At least two seeds and an existing output directory required")
    if args.training < 32:
        parser.error("At least 32 training decisions per phase required")
    cfg = load_config(args.config or Path("configs/sensory-isolated-v1.json")).brain
    rules = args.rules or (
        [cfg.plasticity.rule] if args.config else ["dan-targeted-v1", "compartment-ema-v1"]
    )
    report = {
        "scope": "ROM-free diagnostic. Synthetic action-contingent feedback only in this assay; "
        "never installed as a Pokemon reward. No buttons, readout weights, "
        "or sensory features trained.",
        "protocol": {
            "training_decisions_per_phase": args.training,
            "config": str(args.config) if args.config else "sensory-isolated-v1",
            "preserve_feedback_between_stages": args.preserve_feedback,
            "cue_block_decisions": 16,
            "feedback": 1.0,
            "probe_decisions_per_cue_per_noise": 48,
            "probe_noise_streams": 2,
            "arms": ["paired", "unpaired_within_cue", "frozen"],
            "warmup_decisions": 8,
            "counterbalanced_mappings": 2,
            "cue_images": cues,
            "rewarded_button_pair": buttons,
            "retention": "fresh dynamics and zero reward; weights persist",
            "reload_check": (
                "First 48-decision held-out cue/noise action histogram matches after disk reload"
            ),
            "reversal": "opposite cue-to-direction contingencies, same training budget",
            "criterion_version": 2,
            "criterion": "Exploratory screen, not a significance test: BOTH paired target rate "
            "and conditional competing-button accuracy must "
            "improve >=0.05 over both pretest and unpaired at retention; after reversal the "
            "new target metrics must improve >=0.05 over pre-reversal and unpaired. "
            "Both retained conditional accuracies must also be >=0.55. Require "
            "this for every seed/mapping before testing independent confirmation seeds.",
        },
        "rows": [],
        "learned_gameplay_demonstrated": False,
    }
    for rule in rules:
        config = replace(cfg, plasticity=replace(cfg.plasticity, rule=rule))
        c = InternalBrain(device="cuda", config=config)
        baseline_arrays, baseline_state = c.snapshot()
        for seed in args.seeds:
            for reverse in (False, True):
                c.restore(baseline_arrays, baseline_state, weights_only=True)
                pretest = test_choices(c, seed=seed, cues=cues)
                schedules = {}
                for arm in ("paired", "unpaired_within_cue", "frozen"):
                    c.restore(baseline_arrays, baseline_state, weights_only=True)
                    np.testing.assert_array_equal(c.plasticity.weights, c.plasticity.base)
                    stage_rows = []
                    for stage in (0, 1):
                        mapping = reverse if stage == 0 else not reverse
                        training_seed = seed + 1000 + stage * 10000
                        schedule = None
                        if arm == "unpaired_within_cue":
                            schedule = unpair(schedules[stage], training_seed + 123456)
                        elif arm == "frozen":
                            schedule = schedules[stage]["rewards"]
                        training = train_phase(
                            c,
                            seed=training_seed,
                            reverse=mapping,
                            arm=arm,
                            schedule=schedule,
                            decisions=args.training,
                            preserve_feedback=args.preserve_feedback,
                            buttons=buttons,
                            cues=cues,
                        )
                        if arm == "paired":
                            schedules[stage] = training
                        metrics = c.plasticity.metrics()
                        arrays, state = c.snapshot()
                        label = (
                            f"{rule}-{seed}-{'reversed' if reverse else 'aligned'}-{arm}-{stage}"
                        )
                        archive = args.output / (label + ".npz")
                        np.savez_compressed(archive, **arrays)
                        write_json(args.output / (label + ".json"), state)
                        retention = test_choices(c, seed=seed, cues=cues)
                        # Reload the on-disk weights and repeat one held-out stream.
                        with np.load(archive, allow_pickle=False) as data:
                            restored = {k: data[k].copy() for k in data.files}
                        c.restore(restored, state, weights_only=True)
                        repeat = test_choices(c, seed=seed, first_only=True, cues=cues)
                        assert repeat == retention[:1], (
                            "Saved learned weights changed the retention result"
                        )
                        if args.preserve_feedback:
                            c.restore(restored, state)
                        stage_rows.append(
                            {
                                "stage": "acquisition" if stage == 0 else "reversal",
                                "reverse_mapping": mapping,
                                "training": training,
                                "learning": metrics,
                                "retention": retention,
                                "score": score(retention, mapping, buttons=buttons, cues=cues),
                                "reload_exact": True,
                                "checkpoint": str(archive),
                            }
                        )
                    row = {
                        "rule": rule,
                        "seed": seed,
                        "initial_reverse_mapping": reverse,
                        "arm": arm,
                        "pretest": pretest,
                        "pre_score": score(pretest, reverse, buttons=buttons, cues=cues),
                        "stages": stage_rows,
                    }
                    report["rows"].append(row)
                    write_json(args.output / "learning-choices.json", report)
                    print(
                        json.dumps(
                            {
                                "rule": rule,
                                "seed": seed,
                                "reverse": reverse,
                                "arm": arm,
                                "pre": row["pre_score"],
                                "post": [s["score"] for s in stage_rows],
                                "max_weight_change": stage_rows[-1]["learning"][
                                    "max_relative_change"
                                ],
                            }
                        ),
                        flush=True,
                    )
        del c
    results = []
    for rule in rules:
        for seed in args.seeds:
            for reverse in (False, True):
                arms = {
                    r["arm"]: r
                    for r in report["rows"]
                    if r["rule"] == rule
                    and r["seed"] == seed
                    and r["initial_reverse_mapping"] == reverse
                }
                paired, unpaired = arms["paired"], arms["unpaired_within_cue"]
                results.append(
                    {
                        "rule": rule,
                        "seed": seed,
                        "initial_reverse_mapping": reverse,
                        **compare_arms(paired, unpaired, reverse, buttons=buttons, cues=cues),
                    }
                )
    report["comparisons"] = results
    report["rules_passing_screen"] = [
        r for r in rules if all(x["screen_pass"] for x in results if x["rule"] == r)
    ]
    write_json(args.output / "learning-choices.json", report)


if __name__ == "__main__":
    main()
