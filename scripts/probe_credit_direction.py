"""ROM-free local-credit direction audit, never exported as a gameplay brain.

Freeze a circuit while accumulating its reward-centered eligibility update.
Compare two independent noise estimates, then test bounded temporary changes
along the averaged direction AND its negative on independent noise. This is a
batch diagnostic of the update signal, not online learning or an external
policy. Only existing internal synapses change; the decoder remains fixed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from evaluate_learning_choices import score, warmup

from pokefly.actions import pressed_buttons
from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


def bounded_direction(gradient, post, n, maximum=0.25):
    """Positive diagonal scaling; symmetric signs, no action/cue information."""
    maximum_per_cell = np.zeros(n, np.float64)
    np.maximum.at(maximum_per_cell, post, np.abs(gradient))
    denominator = maximum_per_cell[post]
    return maximum * np.divide(
        gradient, denominator, out=np.zeros_like(gradient), where=denominator > 0
    )


def cosine(a, b):
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / scale) if scale else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/sensorimotor-score-v2.json"))
    p.add_argument("--decisions", type=int, default=4096)
    args = p.parse_args()
    if args.decisions < 32 or args.decisions % 32:
        p.error("Use complete balanced 32-decision cue cycles")
    c = InternalBrain(device="cuda", config=load_config(args.config).brain)
    assert c.config.plasticity.rule == "sensorimotor-score-v2"
    assert c.config.plasticity.scope == "motor-inputs-v1"
    original, state = c.snapshot()
    original_graph = c.brain.weights.copy()
    data = configure_runtime()
    hashes = {name: sha256(data / name) for name in ("brain.npz", "weights.npz")}
    output = run_directory("local-credit-direction-audit")
    report = {
        "scope": __doc__,
        "config": str(args.config),
        "collection_seeds": [501, 601],
        "collection_decisions_each": args.decisions,
        "evaluation_seeds": [2301, 2302, 2303, 2304],
        "decisions_per_cue": 128,
        "maximum_factor_change": 0.25,
        "scaling": "positive diagonal: common max-absolute gradient within each target cell",
        "neural_weights_exported": False,
        "game_used": False,
        "source_hashes": hashes,
        "collection": [],
        "rows": [],
    }
    patterns, gradients = test_patterns(), []
    seconds = c.brain_steps * c.brain.dt
    mean_decay = np.exp(-seconds / c.config.plasticity.reward_expectation_seconds)
    for seed in report["collection_seeds"]:
        c.restore(original, state, weights_only=True)
        warmup(c, seed + 1000)
        total = np.zeros_like(c.plasticity.base, dtype=np.float64)
        half = np.zeros_like(total)
        mean, rewards = 0.0, 0
        actions = {cue: Counter() for cue in ("left", "right")}
        for index in range(args.decisions):
            cue = ("left", "right")[(index // 16) % 2]
            action = c.choose(c.observe(patterns[cue]))[0]
            reward = float(cue in pressed_buttons(action))
            signal = float(np.tanh(reward))
            total += (signal - mean) * c.plasticity.factor_eligibility()
            mean = mean_decay * mean + (1 - mean_decay) * signal
            rewards += int(reward)
            actions[cue][action] += 1
            if index + 1 == args.decisions // 2:
                half[:] = total
            if (index + 1) % 1024 == 0:
                print("frozen credit collection", seed, index + 1, flush=True)
        np.testing.assert_array_equal(c.brain.weights, original_graph)
        gradients.append(total / args.decisions)
        row = {
            "seed": seed,
            "rewards": rewards,
            "half_estimate_cosine": cosine(half, total - half),
            "update_direction_l2": float(np.linalg.norm(total / args.decisions)),
            "score": score([{"cue": k, "actions": dict(v)} for k, v in actions.items()], False),
        }
        report["collection"].append(row)
        write_json(output / "report.json", report)
        print("collected", row, flush=True)
    report["independent_estimate_cosine"] = cosine(*gradients)
    gradient = np.mean(gradients, axis=0)
    direction = bounded_direction(gradient, c.plasticity.post, c.brain.n)
    report["scaled_direction_l2"] = float(np.linalg.norm(direction))
    report["directional_update_inner_product"] = float(gradient @ direction)
    for arm, sign in (("original", 0), ("positive_direction", 1), ("negative_direction", -1)):
        c.restore(original, state, weights_only=True)
        factor = 1 + sign * direction
        assert factor.min() >= 0.75 - 1e-7 and factor.max() <= 1.25 + 1e-7
        c.plasticity.weights[:] = c.plasticity.base * factor
        c.sync_weights()
        frozen = c.brain.weights.copy()
        records = []
        for seed in report["evaluation_seeds"]:
            for cue in ("left", "right"):
                warmup(c, seed)
                choices = [c.choose(c.observe(patterns[cue]))[0] for _ in range(128)]
                records.append(
                    {
                        "cue": cue,
                        "noise_seed": seed,
                        "actions": dict(Counter(choices)),
                        "n": len(choices),
                    }
                )
        np.testing.assert_array_equal(c.brain.weights, frozen)
        row = {"arm": arm, "score": score(records, False), "records": records}
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(arm, row["score"], flush=True)
    c.restore(original, state, weights_only=True)
    np.testing.assert_array_equal(c.brain.weights, original_graph)
    assert hashes == {name: sha256(data / name) for name in hashes}
    report["original_graph_restored_and_source_hashes_unchanged"] = True
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
