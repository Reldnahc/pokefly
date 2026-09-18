"""Necessary circuit-capacity checks, NOT evidence of trained gameplay."""

from __future__ import annotations

import numpy as np

from pokefly.internal_brain import InternalBrain
from pokefly.motors import MOTOR_GROUPS
from pokefly.pixel_brain import paired_changes, test_patterns
from pokefly.runner import run_directory, write_json


def probe_internal(*, device="auto", seed=64, steps=384, trials=2):
    if steps < 12 or trials < 1:
        raise ValueError("Probe needs at least 12 neural steps and one trial")
    controller = InternalBrain(device=device, seed=seed)
    brain = controller.brain
    output = run_directory("internal-probe")
    patterns = test_patterns()
    names = [
        "black",
        "black_repeat",
        "white",
        "left",
        "right",
        "top",
        "bottom",
        "black_weights_x4_control",
    ]
    pairs, conditions = [], []
    for trial in range(trials):
        records, actions = {}, {}
        for name in names:
            controller.reset_dynamics(seed + trial)
            controller.plasticity.weights[:] = controller.plasticity.base * (
                4 if name == "black_weights_x4_control" else 1
            )
            controller.sync_weights()
            drive = controller.retina.encode(patterns.get(name, patterns["black"]))
            traces = {
                key: np.zeros((steps, len(idx)), bool) for key, idx in controller.groups.items()
            }
            counts = np.zeros(brain.n, np.int32)
            decisions = []
            for step in range(steps):
                fired = brain.step(eye_drive=drive)
                counts[fired] += 1
                mask = np.zeros(brain.n, bool)
                mask[fired] = True
                for key, idx in controller.groups.items():
                    traces[key][step] = mask[idx]
                if (step + 1) % controller.brain_steps == 0:
                    decisions.append(
                        controller.decoder.choose(counts, controller.brain_steps * brain.dt)[0]
                    )
                    counts.fill(0)
            records[name], actions[name] = traces, decisions
            kc = traces["Kenyon_cells"]
            conditions.append(
                {
                    "seed": seed + trial,
                    "stimulus": name,
                    "mean_KC_hz": float(kc.mean() / brain.dt),
                    "KC_fraction_above_45hz": float((kc.mean(axis=0) / brain.dt > 45).mean()),
                    "motor_spikes": {k: int(traces[g].sum()) for k, g in MOTOR_GROUPS.items()},
                    "actions": decisions,
                }
            )
            print(f"internal probe seed={seed + trial} stimulus={name}", flush=True)
        for first, second in (
            ("black", "black_repeat"),
            ("black", "white"),
            ("left", "right"),
            ("top", "bottom"),
            ("black", "black_weights_x4_control"),
        ):
            pairs.append(
                {
                    "seed": seed + trial,
                    "first": first,
                    "second": second,
                    "groups": paired_changes(records[first], records[second]),
                    "different_actions": sum(
                        a != b for a, b in zip(actions[first], actions[second], strict=True)
                    ),
                }
            )
    spatial = [p for p in pairs if p["first"] in ("left", "top")]
    repeatable = all(
        all(g["different_neuron_steps"] == 0 for g in p["groups"].values())
        for p in pairs
        if p["second"] == "black_repeat"
    )
    report = {
        "identity": controller.identity(),
        "neural_steps_per_stimulus": steps,
        "trials": trials,
        "matched_noise_repeatable": repeatable,
        "spatial_pairs_with_different_actions": sum(p["different_actions"] > 0 for p in spatial),
        "spatial_pairs_tested": len(spatial),
        "motor_min_spatially_affected_neurons": {
            button: min(p["groups"][group]["affected_neurons"] for p in spatial)
            for button, group in MOTOR_GROUPS.items()
        },
        "internal_weight_perturbation_changes_actions": any(
            p["different_actions"] > 0 for p in pairs if p["second"] == "black_weights_x4_control"
        ),
        "conditions": conditions,
        "comparisons": pairs,
        "learned_gameplay_demonstrated": False,
        "biological_vision_validated": False,
        "caveat": "Fourfold weights are an artificial capacity control, not learned weights. "
        "Any pixel sensitivity is necessary but not proof of usable sight. "
        "Near-50Hz KC saturation is a major limitation of the upstream dynamics.",
    }
    write_json(output / "report.json", report)
    print(f"Internal circuit report: {output / 'report.json'}", flush=True)
    return output
