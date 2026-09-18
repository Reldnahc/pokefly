"""ROM-free, controlled component evaluation. Never trains a button decoder.

The image-conditioning task is deliberately separate from Pokemon rewards.
Reports numerical sensitivity, association and limitations, not animal fidelity.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from pokefly.actions import pressed_buttons
from pokefly.dynamics import DynamicsConfig
from pokefly.internal_brain import InternalBrain
from pokefly.motors import MOTOR_GROUPS
from pokefly.pixel_brain import paired_changes, test_patterns
from pokefly.plasticity import PlasticityConfig
from pokefly.runner import run_directory, write_json


def _step(c, drive, *, learning_trace=False, inject=()):
    fired = c.brain.step(eye_drive=drive, inject=inject)
    if learning_trace:
        c.plasticity.observe(fired, c.brain.dt)
    return fired


def _patterns(c, *, seed, steps):
    records, rows = {}, []
    patterns = test_patterns()
    for name in (*patterns, "black_repeat"):
        c.reset_dynamics(seed)
        drive = c.retina.encode(patterns.get(name, patterns["black"]))
        traces = {k: np.zeros((steps, len(idx)), bool) for k, idx in c.groups.items()}
        actions, counts = [], np.zeros(c.brain.n, np.int32)
        for step in range(steps):
            fired = _step(c, drive)
            mask = np.zeros(c.brain.n, bool)
            mask[fired] = True
            counts[fired] += 1
            for k, idx in c.groups.items():
                traces[k][step] = mask[idx]
            if (step + 1) % c.brain_steps == 0:
                actions.append(c.decoder.choose(counts, c.brain_steps * c.brain.dt)[0])
                counts.fill(0)
        burn = steps // 3
        kc = traces["Kenyon_cells"][burn:]
        rows.append(
            {
                "stimulus": name,
                "seed": seed,
                "discarded_startup_steps": burn,
                "steady_mean_hz": {
                    k: float(v[burn:].mean() / c.brain.dt) for k, v in traces.items()
                },
                "KC_fraction_above_45hz": float((kc.mean(axis=0) / c.brain.dt > 45).mean()),
                "KC_mean_active_fraction_per_step": float(kc.mean()),
                "KC_ever_active_fraction": float(kc.any(axis=0).mean()),
                "actions": actions,
            }
        )
        records[name] = traces, actions
    comparisons = []
    for a, b in (
        ("black", "black_repeat"),
        ("black", "white"),
        ("left", "right"),
        ("top", "bottom"),
    ):
        comparisons.append(
            {
                "first": a,
                "second": b,
                "seed": seed,
                "groups": paired_changes(records[a][0], records[b][0]),
                "steady_groups": paired_changes(
                    {k: v[steps // 3 :] for k, v in records[a][0].items()},
                    {k: v[steps // 3 :] for k, v in records[b][0].items()},
                ),
                "different_actions": sum(
                    x != y for x, y in zip(records[a][1], records[b][1], strict=True)
                ),
            }
        )
    return {"conditions": rows, "comparisons": comparisons}


def _motor_controls(c, seed):
    """Direct neural stimulation tests decoder access, NOT visual motor access."""
    result = {}
    drive = c.retina.encode(test_patterns()["black"])
    for button, cells in c.motors.items():
        c.reset_dynamics(seed)
        counts = np.zeros(c.brain.n, np.int32)
        for _ in range(c.brain_steps):
            counts[_step(c, drive, inject=[(cells, 1.1)])] += 1
        action, rates = c.decoder.choose(counts, c.brain_steps * c.brain.dt)
        result[button] = {
            "body_ids": c.body_ids[cells].tolist(),
            "injected_voltage_per_step": 1.1,
            "target_spikes": int(counts[cells].sum()),
            "chosen": action,
            "correct_readout": button in pressed_buttons(action),
            "motor_rates_hz": rates,
        }
    return result


def evaluate_circuits(*, config, device="auto", seed=64, steps=384, trials=2):
    if steps < 48 or trials < 1:
        raise ValueError("At least 48 steps and one trial required")
    if config.dynamics.profile != "hybrid-v1":
        raise ValueError("Circuit ablation requires the explicit hybrid profile")
    output = run_directory("circuit-evaluation")
    variants = {
        "baseline": replace(config, dynamics=DynamicsConfig(), plasticity=PlasticityConfig()),
        "adaptation_only": replace(config, dynamics=replace(config.dynamics, graded_vision=False)),
        "hybrid": config,
    }
    report = {"steps": steps, "seeds": list(range(seed, seed + trials)), "variants": {}}
    for name, variant in variants.items():
        c = InternalBrain(device=device, seed=seed, config=variant)
        rows = []
        for s in range(seed, seed + trials):
            rows.append(_patterns(c, seed=s, steps=steps))
            print(f"circuit profile={name} seed={s} complete", flush=True)
        report["variants"][name] = {
            "identity": c.identity(),
            "trials": rows,
            "motor_activation_controls": _motor_controls(c, seed),
            "population_sizes": {k: len(v) for k, v in c.groups.items()},
            "plasticity": c.plasticity.metrics(),
        }
        write_json(output / "report.json", report)
        del c
    report["scope"] = (
        "Direct motor injection is ONLY a diagnostic. No gameplay, decoder fitting, "
        "semantic input or rewards used. Graded cells produce no spikes. "
        "Numerical sensitivity is not validated vision."
    )
    report["parameters_selected_using_gameplay"] = False
    write_json(output / "report.json", report)
    print(f"Circuit evaluation: {output / 'report.json'}", flush=True)
    return output


def _presentation(c, frame, *, seed, steps, warmup=96):
    # Reset removes transient memory between presentations; only synaptic memory
    # can carry over. Every control gets identical image/noise sequences.
    c.reset_dynamics(seed)
    blank = c.retina.encode(test_patterns()["black"])
    for _ in range(warmup):
        _step(c, blank)
    drive, counts = c.retina.encode(frame), np.zeros(c.brain.n, np.int32)
    for _ in range(steps):
        counts[_step(c, drive, learning_trace=True)] += 1
    return counts


def _responses(c, patterns, *, seed, steps):
    return {
        name: _presentation(c, frame, seed=seed, steps=steps) for name, frame in patterns.items()
    }


def _readout(c, naive, before, after):
    p = c.plasticity
    result = {}
    for name in naive:
        input_counts = naive[name][p.pre].astype(np.float64)
        original = np.sum(np.abs(p.base) * input_counts)
        remaining = np.sum(np.abs(p.weights) * input_counts)
        result[name] = {
            "fixed_naive_input_synaptic_suppression": float(1 - remaining / original)
            if original
            else None,
            "naive_MBON_spikes": int(before[name][c.groups["MBONs"]].sum()),
            "retained_MBON_spikes": int(after[name][c.groups["MBONs"]].sum()),
            "MBON_count_L1_change": int(
                np.abs(after[name][c.groups["MBONs"]] - before[name][c.groups["MBONs"]]).sum()
            ),
            "motor_spikes": {b: int(after[name][c.motors[b]].sum()) for b in MOTOR_GROUPS},
        }
    return result


def evaluate_association(*, config, device="auto", seed=64, trials=2, pairings=12, steps=96):
    if trials < 2 or pairings < 2 or steps < 12:
        raise ValueError("Association test requires >=2 seeds, >=2 pairings and >=12 steps")
    if config.plasticity.rule != "dan-targeted-v1":
        raise ValueError("Association diagnostic expects the explicit DAN-targeted rule")
    output = run_directory("visual-association")
    images = test_patterns()
    patterns = {"A": images["left"], "B": images["right"]}
    report = {
        "config": asdict(config),
        "pairings": pairings,
        "steps_per_presentation": steps,
        "reward": 1.0,
        "conditions": [],
        "protocol": "Raw left/right half-white images. Reset dynamics then 96 black warmup "
        "steps before each presentation; only weights carry over. A rewarded, B unpaired. "
        "Shuffled control alternates rewarded A/B; frozen has identical A rewards but no "
        "updates. Pre/post tests have no reward, identical noise, and fresh dynamics. "
        "Reversal rewards B. This is NOT a Pokemon reward or trained decoder.",
    }
    for s in range(seed, seed + trials):
        for condition in ("paired", "shuffled", "frozen"):
            c = InternalBrain(device=device, seed=s, config=config)
            naive = _responses(c, patterns, seed=s + 100000, steps=steps * 2)
            for pairing in range(pairings):
                for name, frame in patterns.items():
                    _presentation(c, frame, seed=s + 1000 + pairing, steps=steps)
                    rewarded = name == ("B" if condition == "shuffled" and pairing % 2 else "A")
                    c.reinforce(float(rewarded), enabled=condition != "frozen")
            retained = _responses(c, patterns, seed=s + 100000, steps=steps * 2)
            row = {
                "seed": s,
                "condition": condition,
                "identity": c.identity(),
                "retention": _readout(c, naive, naive, retained),
                "plasticity": c.plasticity.metrics(),
            }
            if condition == "paired":
                retained_weights = c.plasticity.weights.copy()
                repeat = _responses(c, patterns, seed=s + 100000, steps=steps * 2)
                row["reward_free_repeat_exact"] = all(
                    np.array_equal(retained[k], repeat[k]) for k in retained
                )
                row["reward_free_weights_unchanged"] = bool(
                    np.array_equal(retained_weights, c.plasticity.weights)
                )
                for pairing in range(pairings):
                    for name, frame in patterns.items():
                        _presentation(c, frame, seed=s + 1000 + pairing, steps=steps)
                        c.reinforce(float(name == "B"))
                reversed_responses = _responses(c, patterns, seed=s + 100000, steps=steps * 2)
                row["reversal"] = _readout(c, naive, naive, reversed_responses)
            report["conditions"].append(row)
            write_json(output / "report.json", report)
            print(f"association seed={s} condition={condition} complete", flush=True)
            del c
    report["interpretation"] = (
        "Fixed-input synaptic suppression isolates stored weight effects, not behavior. "
        "MBON spike changes alone can reflect nonlinear dynamics. Rewarded-image specificity "
        "must exceed shuffled/frozen controls; reversal should change specificity. Neither "
        "this assay nor changed weights demonstrates learned Pokemon behavior."
    )
    report["learned_gameplay_demonstrated"] = False
    write_json(output / "report.json", report)
    print(f"Visual association report: {output / 'report.json'}", flush=True)
    return output
