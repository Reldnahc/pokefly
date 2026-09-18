"""Read-only forward-circuit audit with explicitly isolated counterfactuals.

No emulator, rewards, training, saved weights, or production configuration is
changed. Current removal and injected voltages below are DIAGNOSTICS, not fixes.
Run with: python -m pokefly.forward_probe --frame path/to/start.png --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse

from pokefly.actions import count_buttons
from pokefly.dynamics import DynamicsConfig
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime

TRACKED_TYPES = (
    "DNg100",
    "DNge054",
    "DNp09",
    "DNg97",
    "DNg75",
    "DNge050",
    "DNge053",
    "AL-AST1",
    "ALON3",
)


def signed_contributions(rows, mean_release, gain):
    """Mean increments per neural timestep; rows=postsynaptic, columns=presynaptic."""
    contributions = rows.multiply(mean_release * gain).tocsr()
    positive = np.asarray(contributions.maximum(0).sum(axis=1)).ravel()
    negative = np.asarray(contributions.minimum(0).sum(axis=1)).ravel()
    return contributions, positive, negative


def host(value):
    return value if isinstance(value, np.ndarray) else value.get()


def audit_condition(c, matrix, types, classes, *, frame, seed, steps, burn, intervention):
    b, xp = c.brain, c.brain.xp
    c.reset_dynamics(seed)
    original_noise = b.noise_hz
    tracked = b.cells(list(TRACKED_TYPES))
    up = c.motors["up"]
    brakes = b.cells(["DNge054"])
    counts, steady_counts = np.zeros(b.n, np.int32), np.zeros(b.n, np.int32)
    window = np.zeros(b.n, np.int32)
    release_sum = xp.zeros(b.n, xp.float64)
    voltage_sum, voltage_max = np.zeros(len(tracked)), np.full(len(tracked), -np.inf)
    actions = Counter()
    drive = c.retina.encode(frame)
    inject = []
    removed = sparse.csr_matrix((len(up), b.n), dtype=np.float32)
    original_current = c.hybrid.current if c.hybrid else None
    isolated_input_targets = np.array([], dtype=np.int64)
    if intervention == "no_noise":
        b.noise_hz = 0
    elif intervention == "no_DNge054_tonic":
        inject = [(brakes, -float(b.tonic))]
    elif intervention == "stimulate_DNp09":
        inject = [(b.cells(["DNp09"]), 1.1)]
    elif intervention == "stimulate_Up":
        inject = [(up, 0.1)]
    elif intervention in ("no_BM_recurrent_input", "no_nonvisual_sensory_recurrent_input"):
        if not c.hybrid:
            raise ValueError("Sensory current diagnostic requires hybrid dynamics")
        selected = (
            np.char.startswith(types, "BM_")
            if intervention == "no_BM_recurrent_input"
            else np.char.find(classes, "sensory") >= 0
        )
        selected[b.visual] = False
        isolated_input_targets = np.flatnonzero(selected)
        isolated_gpu = xp.asarray(isolated_input_targets)

        def current(release):
            result = original_current(release)
            result[isolated_gpu, 0] = 0
            return result

        c.hybrid.current = current
    elif intervention in ("remove_DNge054_to_Up", "remove_all_inhibition_to_Up"):
        if not c.hybrid:
            raise ValueError("Current-removal diagnostic requires hybrid dynamics")
        removed = matrix[up].minimum(0).tocsr()
        if intervention == "remove_DNge054_to_Up":
            removed = removed.multiply(types == "DNge054").tocsr()
        removed.eliminate_zeros()
        pre = np.unique(removed.indices)
        pre_gpu, up_gpu = xp.asarray(pre), xp.asarray(up)
        coefficients = xp.asarray(removed[:, pre].toarray())

        def current(release):
            result = original_current(release)
            result[up_gpu, 0] -= coefficients @ release[pre_gpu]
            return result

        c.hybrid.current = current
    elif intervention != "unaltered":
        raise ValueError(intervention)
    try:
        for step in range(steps):
            # Synaptic input uses PREVIOUS-step release, not the new spikes below.
            if step >= burn:
                if c.hybrid:
                    release_sum += c.hybrid.release
                else:
                    release_sum[b.fired] += 1
            fired = b.step(eye_drive=drive, inject=inject)
            counts[fired] += 1
            window[fired] += 1
            if step >= burn:
                steady_counts[fired] += 1
                voltage = host(b.v[tracked, 0])
                voltage_sum += voltage
                voltage_max = np.maximum(voltage_max, voltage)
            if (step + 1) % c.brain_steps == 0:
                action, _ = c.decoder.choose(window, c.brain_steps * b.dt)
                actions[action] += 1
                window.fill(0)
    finally:
        b.noise_hz = original_noise
        if original_current is not None:
            c.hybrid.current = original_current
    mean_release = host(release_sum) / (steps - burn)
    incoming, exc, inh = signed_contributions(matrix[tracked], mean_release, b.gain)
    removed_current = np.asarray(removed @ mean_release).ravel() * b.gain
    neurons = []
    for k, cell in enumerate(tracked):
        row = incoming.getrow(k)
        by_type, by_class = Counter(), Counter()
        for pre, value in zip(row.indices, row.data, strict=True):
            by_type[str(types[pre])] += float(value)
            by_class[str(classes[pre])] += float(value)
        up_index = np.flatnonzero(up == cell)
        removed_increment = float(removed_current[up_index[0]]) if len(up_index) else 0.0
        neurons.append(
            {
                "body_id": int(c.body_ids[cell]),
                "type": str(types[cell]),
                "spikes": int(counts[cell]),
                "post_startup_hz": float(steady_counts[cell] / ((steps - burn) * b.dt)),
                "mean_voltage_after_reset": float(voltage_sum[k] / (steps - burn)),
                "max_voltage_after_reset": float(voltage_max[k]),
                "mean_excitatory_increment": float(exc[k]),
                "mean_inhibitory_increment_before_removal": float(inh[k]),
                "removed_negative_increment": removed_increment,
                "effective_mean_synaptic_increment": float(exc[k] + inh[k] - removed_increment),
                "top_input_types": dict(sorted(by_type.items(), key=lambda kv: -abs(kv[1]))[:20]),
                "net_input_by_superclass": dict(by_class),
            }
        )
    assert np.array_equal(c.plasticity.weights, c.plasticity.base)
    return {
        "seed": seed,
        "intervention": intervention,
        "neurons": neurons,
        "forward_spikes": int(counts[up].sum()),
        "actions": dict(actions),
        "button_counts": count_buttons(actions),
        "weights_unchanged": True,
        "incoming_current_zeroed_only_in_diagnostic": int(len(isolated_input_targets)),
        "population_activity": {
            name: {
                "cells": int(mask.sum()),
                "mean_post_startup_hz": float(steady_counts[mask].mean() / ((steps - burn) * b.dt)),
                "fraction_above_45hz": float(
                    (steady_counts[mask] / ((steps - burn) * b.dt) > 45).mean()
                ),
            }
            for name, mask in {
                "BM_InOm": types == "BM_InOm",
                "BM_prefix": np.char.startswith(types, "BM_"),
                "ORN_prefix": np.char.startswith(types, "ORN_"),
                "cb_sensory": classes == "cb_sensory",
            }.items()
            if mask.any()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid-v1.json"))
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument(
        "--sensory-followup",
        action="store_true",
        help="Isolate incoming currents to nonvisual sensory cells; diagnostic only",
    )
    args = parser.parse_args()
    config = load_config(args.config).brain
    if config.dynamics.profile != "hybrid-v1":
        parser.error("The audit requires the explicit hybrid-v1 configuration")
    if args.steps < 96 or args.steps % config.brain_steps or args.trials < 2:
        parser.error("Require >=96 steps, whole decision windows, and >=2 trials")
    data = configure_runtime()
    output = run_directory("forward-sensory-audit" if args.sensory_followup else "forward-audit")
    hashes = {
        name: hashlib.sha256((data / name).read_bytes()).hexdigest()
        for name in ("brain.npz", "weights.npz")
    }
    matrix = sparse.load_npz(data / "weights.npz").tocsr()
    with np.load(data / "brain.npz", allow_pickle=False) as meta:
        types, classes = meta["cell_type"].astype(str), meta["superclass"].astype(str)
    with Image.open(args.frame) as image:
        bedroom = np.asarray(image.convert("RGB"))
    patterns = test_patterns()
    stimuli = {"bedroom": bedroom, "black": patterns["black"], "white": patterns["white"]}
    # Static stimuli isolate persistent inhibition; dynamic vision needs its own assay.
    report = {
        "scope": "Frozen, ROM-free neural diagnostic. No production configuration or "
        "connectome changed. Current removal / stimulation are causal controls, NOT fixes.",
        "steps": args.steps,
        "discarded_startup_steps": args.steps // 6,
        "frame_sha256": hashlib.sha256(args.frame.read_bytes()).hexdigest(),
        "data_sha256": hashes,
        "profiles": {},
        "units": "Synaptic values are mean voltage increments per 20ms step, gain included. "
        "Voltage sampled AFTER spiking reset. Spike totals include startup; means exclude it. "
        "Top inputs use signed sums; net near zero does not imply no connections.",
    }
    for profile in ("hybrid-v1",) if args.sensory_followup else ("baseline", "hybrid-v1"):
        variant = replace(config, dynamics=DynamicsConfig()) if profile == "baseline" else config
        c = InternalBrain(device=args.device, seed=args.seed, config=variant)
        b = c.brain
        detail = {
            "identity": c.identity(),
            "conditions": [],
            "constants": {
                "dt": b.dt,
                "decay": float(b.decay),
                "tonic": float(b.tonic),
                "noise_hz": b.noise_hz,
                "noise_amplitude": b.noise_amp,
                "threshold": 1,
            },
        }
        report["profiles"][profile] = detail
        interventions = ["unaltered", "no_noise"]
        if profile == "hybrid-v1":
            interventions += [
                "no_DNge054_tonic",
                "remove_DNge054_to_Up",
                "remove_all_inhibition_to_Up",
                "stimulate_DNp09",
                "stimulate_Up",
            ]
        if args.sensory_followup:
            interventions = [
                "unaltered",
                "no_BM_recurrent_input",
                "no_nonvisual_sensory_recurrent_input",
            ]
        for seed in range(args.seed, args.seed + args.trials):
            for stimulus, frame in stimuli.items():
                for intervention in (
                    interventions
                    if stimulus == "bedroom" or args.sensory_followup
                    else ["unaltered"]
                ):
                    row = audit_condition(
                        c,
                        matrix,
                        types,
                        classes,
                        frame=frame,
                        seed=seed,
                        steps=args.steps,
                        burn=args.steps // 6,
                        intervention=intervention,
                    )
                    row["stimulus"] = stimulus
                    detail["conditions"].append(row)
                    write_json(output / "report.json", report)
                    print(
                        f"{profile} seed={seed} {stimulus} {intervention}: "
                        f"Up spikes={row['forward_spikes']} commands={row['button_counts']['up']}",
                        flush=True,
                    )
        del c
    report["original_data_unchanged"] = all(
        hashlib.sha256((data / name).read_bytes()).hexdigest() == digest
        for name, digest in hashes.items()
    )
    assert report["original_data_unchanged"]
    write_json(output / "report.json", report)
    print(f"Forward audit: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
