"""Read-only motor input contrast/noise measurement from frozen neural counts.

No classifier, fitted synapses, game, reward changes or new simulation. The
measure concerns mean synaptic current over decision windows, not membrane
voltage or a validated explanation of downstream choices. Graded-input edges
are counted separately because spike counts cannot reconstruct their release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.motors import MOTOR_TYPES
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True)
    args = p.parse_args()
    source = json.loads((args.probe / "report.json").read_text())
    cfg = load_config(Path(source["config"])).brain
    data = configure_runtime()
    with np.load(data / "brain.npz", allow_pickle=False) as meta:
        ids, kinds, sides = meta["ids"], meta["cell_type"].astype(str), meta["side"].astype(str)
        graded = np.isin(kinds, ["L1", "L2", "L3", "L4", "L5"])
        graded[meta["visual"]] = True
    recorded_graded = "summed transmitter release" in source.get("recorded_activity", "")
    graph = sparse.load_npz(data / "weights.npz").tocsr()
    with np.load(args.probe / "counts.npz", allow_pickle=False) as counts:
        lookup = {int(value): i for i, value in enumerate(ids)}
        cells = np.array([lookup[int(value)] for value in counts["body_ids"]])
        values = {
            cue: np.concatenate(
                [
                    counts[f"{seed}_{cue}"][64:].astype(float) / cfg.brain_steps
                    for seed in source["seeds"]
                ]
            )
            for cue in ("left", "right")
        }
    dt = source["neural_seconds_per_decision"] / cfg.brain_steps
    probability = cfg.noise_hz * dt
    independent_noise_std = cfg.noise_amplitude * np.sqrt(
        probability * (1 - probability) / cfg.brain_steps
    )
    report = {
        "scope": __doc__,
        "source": str(args.probe),
        "counts_sha256": sha256(args.probe / "counts.npz"),
        "decision_averaged_background_current_std": float(independent_noise_std),
        "synaptic_gain": cfg.synaptic_gain,
        "recorded_activity": source.get("recorded_activity", "spike counts"),
        "rows": [],
    }
    for button, (kind, side) in MOTOR_TYPES.items():
        for target in np.flatnonzero((kinds == kind) & ((sides == side) if side else True)):
            row = graph[target]
            pre = row.indices
            assert np.isin(pre, cells).all(), "Missing input cells in the source measurement"
            vector = graph[target, cells].toarray().ravel() * cfg.synaptic_gain
            currents = {cue: response @ vector for cue, response in values.items()}
            means = {cue: float(current.mean()) for cue, current in currents.items()}
            stds = {cue: float(current.std(ddof=1)) for cue, current in currents.items()}
            contrast = means["left"] - means["right"]
            pooled_std = np.sqrt(np.mean(np.square(list(stds.values()))))
            result = {
                "button_measurement_only": button,
                "body_id": int(ids[target]),
                "graded_edges_not_reconstructed": 0 if recorded_graded else int(graded[pre].sum()),
                "cue_means": means,
                "cue_standard_deviations": stds,
                "left_minus_right_current": contrast,
                "contrast_over_synaptic_std": float(contrast / pooled_std) if pooled_std else None,
                "contrast_over_background_current_std": float(contrast / independent_noise_std)
                if independent_noise_std
                else None,
            }
            report["rows"].append(result)
            print(result, flush=True)
    output = run_directory("motor-signal-audit")
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
