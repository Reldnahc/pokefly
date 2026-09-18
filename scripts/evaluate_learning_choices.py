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


def test_choices(c, *, seed, decisions=48, first_only=False):
    patterns = test_patterns()
    result = []
    original = c.plasticity.weights.copy()
    for noise_seed in (seed + 100000, seed + 200000):
        for cue in ("left", "right"):
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


def score(rows, reverse):
    hits, choices, total = 0, 0, 0
    for row in rows:
        target = ("right" if row["cue"] == "left" else "left") if reverse else row["cue"]
        for action, n in row["actions"].items():
            buttons = pressed_buttons(action)
            hits += n * (target in buttons)
            choices += n * bool(set(buttons) & {"left", "right"})
            total += n
    return {
        "target_rate": hits / total,
        "conditional_accuracy": hits / choices if choices else None,
        "left_right_choices": choices,
        "decisions": total,
    }


def train_phase(c, *, seed, reverse, arm, schedule=None, decisions=256, preserve_feedback=False):
    mean = c.plasticity.reward_mean.copy() if preserve_feedback else None
    warmup(c, seed)
    if mean is not None:
        c.plasticity.reward_mean[...] = mean
    patterns = test_patterns()
    pulses, actions, cues = [], [], []
    for index in range(decisions):
        cue = ("left", "right")[(index // 16) % 2]
        target = ("right" if cue == "left" else "left") if reverse else cue
        action = c.choose(c.observe(patterns[cue]))[0]
        reward = (
            float(target in pressed_buttons(action)) if schedule is None else float(schedule[index])
        )
        c.reinforce(reward, enabled=arm != "frozen")
        pulses.append(reward)
        actions.append(action)
        cues.append(cue)
    return {"rewards": pulses, "actions": actions, "cues": cues, "total_reward": sum(pulses)}


def unpair(training, seed):
    # Permute within cue: preserve each cue's reward amount, remove the alignment
    # to the selected motor command. This is not a cue-value-imbalance control.
    rng = np.random.default_rng(seed)
    rewards = np.asarray(training["rewards"], float)
    cues = np.asarray(training["cues"])
    result = rewards.copy()
    for cue in ("left", "right"):
        mask = cues == cue
        result[mask] = rng.permutation(rewards[mask])
        assert result[mask].sum() == rewards[mask].sum()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rules", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--training", type=int, default=256)
    parser.add_argument("--preserve-feedback", action="store_true")
    args = parser.parse_args()
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
            "retention": "fresh dynamics and zero reward; weights persist",
            "reload_check": (
                "First 48-decision held-out cue/noise action histogram matches after disk reload"
            ),
            "reversal": "opposite cue-to-direction contingencies, same training budget",
            "criterion": "Exploratory screen, not a significance test: paired target rate must "
            "improve >=0.05 over both pretest and unpaired at retention; after reversal the "
            "new target rate must improve >=0.05 over pre-reversal and unpaired. Require "
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
                pretest = test_choices(c, seed=seed)
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
                        retention = test_choices(c, seed=seed)
                        # Reload the on-disk weights and repeat one held-out stream.
                        with np.load(archive, allow_pickle=False) as data:
                            restored = {k: data[k].copy() for k in data.files}
                        c.restore(restored, state, weights_only=True)
                        repeat = test_choices(c, seed=seed, first_only=True)
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
                                "score": score(retention, mapping),
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
                        "pre_score": score(pretest, reverse),
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
                acquisition = paired["stages"][0]["score"]["target_rate"]
                reversal = paired["stages"][1]["score"]["target_rate"]
                before_reversal = score(paired["stages"][0]["retention"], not reverse)[
                    "target_rate"
                ]
                deltas = {
                    "acquisition_minus_pre": acquisition - paired["pre_score"]["target_rate"],
                    "acquisition_minus_unpaired": acquisition
                    - unpaired["stages"][0]["score"]["target_rate"],
                    "reversal_minus_before_reversal": reversal - before_reversal,
                    "reversal_minus_unpaired": reversal
                    - unpaired["stages"][1]["score"]["target_rate"],
                }
                results.append(
                    {
                        "rule": rule,
                        "seed": seed,
                        "initial_reverse_mapping": reverse,
                        **deltas,
                        "screen_pass": all(v >= 0.05 for v in deltas.values()),
                    }
                )
    report["comparisons"] = results
    report["rules_passing_screen"] = [
        r for r in rules if all(x["screen_pass"] for x in results if x["rule"] == r)
    ]
    write_json(args.output / "learning-choices.json", report)


if __name__ == "__main__":
    main()
