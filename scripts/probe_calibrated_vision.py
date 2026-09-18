"""Standalone calibrated visual-neuron responses; no motor policy or game.

Original photoreceptors see raw generic images through the unchanged retina.
Only the visual subcircuit is active in this diagnostic; outside neural input
is explicitly zero. This cannot establish end-to-end control or learning.
"""

import argparse

import numpy as np

from pokefly.calibrated_vision import CalibratedVisualCircuit
from pokefly.checkpoint import sha256
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


def grating(angle, step, phase):
    y, x = np.indices((144, 160))
    theta = np.deg2rad(angle)
    position = x * np.cos(theta) + y * np.sin(theta) - 2 * step + phase
    pixels = (np.floor(position / 16) % 2 * 255).astype(np.uint8)
    return np.repeat(pixels[:, :, None], 3, axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", choices=("fitted", "flyvis"), default="fitted")
    parser.add_argument(
        "--fallback-release-scale", type=float, default=1.0,
        help="Standalone unit-sensitivity diagnostic ONLY; never changes a gameplay profile",
    )
    args = parser.parse_args()
    if not 0 < args.fallback_release_scale <= 1:
        parser.error("Fallback diagnostic scale must be in (0,1]")
    c = CalibratedVisualCircuit(parameters=args.parameters)
    original = c.matrix.data.copy()
    if args.fallback_release_scale != 1:
        c.matrix.data[c.fallback_edge_mask] *= np.float32(args.fallback_release_scale)
        c.propagate.brain._W.data[:] = c.xp.asarray(c.matrix.data)
        np.testing.assert_array_equal(np.sign(original), np.sign(c.matrix.data))
        np.testing.assert_array_equal(
            original[~c.fallback_edge_mask], c.matrix.data[~c.fallback_edge_mask]
        )
    data = configure_runtime()
    hashes = {name: sha256(data / name) for name in ("brain.npz", "weights.npz", "retina.npz")}
    output = run_directory("calibrated-visual-probe")
    groups = {
        kind + "_" + side: np.flatnonzero((c.types == kind) & (c.sides == side))
        for kind in ("L1", "Mi1", "Mi4", "Mi9", "T4a", "T4b", "T4c", "T4d",
                     "T5a", "T5b", "T5c", "T5d", "HSE", "HSN", "HSS", "LC4", "LPLC2")
        for side in ("L", "R")
    }
    groups = {key: value for key, value in groups.items() if len(value)}
    report = {
        "scope": __doc__, "model": c.info, "protected_source_sha256": hashes,
        "neutral_warmup_seconds": 2, "stimulus_seconds": 2.56,
        "scoring_window_seconds": [1.28, 2.56], "phases_pixels": [0, 8, 16],
        "angles_degrees": list(range(0, 360, 45)), "rows": [],
        "fallback_release_scale_diagnostic_only": args.fallback_release_scale,
        "original_fallback_edges": int(c.fallback_edge_mask.sum()),
        "reference_and_real_photo_lamina_conductances_unchanged": True,
        "gameplay_profile_modified": False,
    }
    gray = np.full((144, 160, 3), 128, np.uint8)
    for phase in report["phases_pixels"]:
        for condition in ["gray", "static", *report["angles_degrees"]]:
            c.reset()
            for _ in range(100):
                c.advance(gray)
            rates = []
            for step in range(128):
                frame = gray if condition == "gray" else grating(
                    0 if condition == "static" else condition,
                    0 if condition == "static" else step, phase,
                )
                c.advance(frame)
                if step >= 64:
                    rate = c.host_rate()
                    rates.append([float(rate[indices].mean()) for indices in groups.values()])
            values = np.array(rates)
            row = {"phase": phase, "condition": condition,
                   "mean_rates": dict(zip(groups, values.mean(axis=0).tolist(), strict=True)),
                   "saturated_fraction": float(np.mean(c.host_rate() >= c.maximum_rate))}
            report["rows"].append(row)
            write_json(output / "report.json", report)
            print("visual response", phase, condition, flush=True)
    report["tuning"] = {}
    radians = np.deg2rad(report["angles_degrees"])
    for group in groups:
        response = np.array([
            np.mean([r["mean_rates"][group] for r in report["rows"] if r["condition"] == angle])
            for angle in report["angles_degrees"]
        ])
        vector = np.sum(response * np.exp(1j * radians))
        total = response.sum()
        report["tuning"][group] = {
            "rates_by_angle": response.tolist(),
            "direction_selectivity": float(abs(vector) / total) if total else None,
            "preferred_degrees": float(np.rad2deg(np.angle(vector)) % 360) if total else None,
            "gray_rate": float(np.mean([
                r["mean_rates"][group] for r in report["rows"] if r["condition"] == "gray"
            ])),
            "static_rate": float(np.mean([
                r["mean_rates"][group] for r in report["rows"] if r["condition"] == "static"
            ])),
        }
    assert hashes == {name: sha256(data / name) for name in hashes}
    report["protected_files_unchanged"] = True
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
