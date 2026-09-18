"""Frozen signed-current and retained-weight audit. No gameplay/reward intervention.

Quiet nonvisual cells are a candidate inactive-peripheral-input boundary. The
all-inhibition-to-Up removal is a capacity control only, never a game profile.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse

from pokefly.actions import count_buttons
from pokefly.checkpoint import read_checkpoint
from pokefly.experiment import load_config
from pokefly.forward_probe import host, signed_contributions
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import write_json


def stimulus(name, frames, step):
    if name == "moving_grating":
        x = np.arange(160)[None, :]
        gray = np.broadcast_to(((x + step * 2) % 32 < 16) * 255, (144, 160))
        return np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)
    return frames[name]


def audit(c, matrix, types, frames, *, name, seed, steps=1536, remove_inhibition=False):
    c.reset_dynamics(seed)
    b, xp = c.brain, c.brain.xp
    tracked = np.unique(np.concatenate([*c.motors.values(), b.cells(["DNge054"])]))
    original_current = c.hybrid.current
    up = c.motors["up"]
    removed = matrix[up].minimum(0).tocsr()
    pre = np.unique(removed.indices)
    coefficients = xp.asarray(removed[:, pre].toarray())
    gpu_up, gpu_pre = xp.asarray(up), xp.asarray(pre)
    if remove_inhibition:

        def current(release):
            result = original_current(release)
            result[gpu_up, 0] -= coefficients @ release[gpu_pre]
            return result

        c.hybrid.current = current
    burn = steps // 4
    total = np.zeros(b.n, np.int32)
    steady = np.zeros(b.n, np.int32)
    window = np.zeros(b.n, np.int32)
    release_sum = xp.zeros(b.n, xp.float64)
    v_sum = np.zeros(len(tracked), np.float64)
    actions, spike_windows = [], []
    try:
        for step in range(steps):
            if step >= burn:
                release_sum += c.hybrid.release
            fired = b.step(eye_drive=c.retina.encode(stimulus(name, frames, step)))
            total[fired] += 1
            window[fired] += 1
            if step >= burn:
                steady[fired] += 1
                v_sum += host(b.v[tracked, 0])
            if (step + 1) % c.brain_steps == 0:
                actions.append(c.decoder.choose(window, c.brain_steps * b.dt)[0])
                spike_windows.append({key: int(window[idx].sum()) for key, idx in c.motors.items()})
                window.fill(0)
    finally:
        c.hybrid.current = original_current
    incoming, exc, inh = signed_contributions(
        matrix[tracked], host(release_sum) / (steps - burn), b.gain
    )
    cells = []
    for i, neuron in enumerate(tracked):
        by_type = Counter()
        row = incoming.getrow(i)
        for index, value in zip(row.indices, row.data, strict=True):
            by_type[str(types[index])] += float(value)
        cells.append(
            {
                "body_id": int(c.body_ids[neuron]),
                "type": str(types[neuron]),
                "hz": float(steady[neuron] / ((steps - burn) * b.dt)),
                "mean_voltage_after_reset": float(v_sum[i] / (steps - burn)),
                "mean_excitation": float(exc[i]),
                "mean_inhibition_before_ablation": float(inh[i]),
                "top_inputs": dict(sorted(by_type.items(), key=lambda x: -abs(x[1]))[:12]),
            }
        )
    return {
        "stimulus": name,
        "seed": seed,
        "steps": steps,
        "burn": burn,
        "actions": actions,
        "button_counts": count_buttons(Counter(actions)),
        "spike_windows": spike_windows,
        "neurons": cells,
        "population_hz": {
            key: float(steady[idx].mean() / ((steps - burn) * b.dt))
            for key, idx in c.groups.items()
        },
        "visual_kc_counts": steady[c.groups["visual_Kenyon_candidates"]].tolist(),
        "visual_kc_ever_active": int(
            np.count_nonzero(steady[c.groups["visual_Kenyon_candidates"]])
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays, saved = read_checkpoint(args.source_run / "latest-checkpoint.json")
    config = load_config(Path("configs/sensory-isolated-v1.json")).brain
    matrix = sparse.load_npz("fly-data/weights.npz").tocsr()
    with np.load("fly-data/brain.npz", allow_pickle=False) as data:
        types = data["cell_type"].astype(str)
    frames = test_patterns()
    with Image.open(args.source_run / "start.png") as im:
        frames["bedroom"] = np.asarray(im.convert("RGB"))
    frames["pallet"] = arrays["next_frame"][:, :, :3]
    report = {
        "source_checkpoint": saved["directory"],
        "scope": "Frozen diagnostic only",
        "rows": [],
    }
    for variant in ("original", "retained", "quiet_sensory", "remove_up_inhibition", "plastic_x4"):
        cfg = (
            replace(config, dynamics=replace(config.dynamics, quiescent_nonvisual_sensory=True))
            if variant == "quiet_sensory"
            else config
        )
        c = InternalBrain(device="cuda", config=cfg)
        if variant == "retained":
            c.restore(arrays, saved["neural"], weights_only=True)
        elif variant == "plastic_x4":
            c.plasticity.weights[:] = c.plasticity.base * 4
            c.sync_weights()
        for seed in (101, 102, 103):
            names = (
                ("black", "pallet", "left", "right", "moving_grating")
                if variant in ("original", "quiet_sensory")
                else ("pallet",)
            )
            for name in names:
                row = audit(
                    c,
                    matrix,
                    types,
                    frames,
                    name=name,
                    seed=seed,
                    remove_inhibition=variant == "remove_up_inhibition",
                )
                row["variant"] = variant
                report["rows"].append(row)
                write_json(args.output / "motor-audit.json", report)
                print(f"audit {variant} {seed} {name}: {row['button_counts']}", flush=True)
        del c


if __name__ == "__main__":
    main()
