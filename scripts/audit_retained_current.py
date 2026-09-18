"""Read-only first-order current audit of ROM-free acquired synapses.

Reuse ORIGINAL frozen presynaptic measurements, so this estimates a weight
change's direct current effect, not its full recurrent/behavioral effect.
No fit, weights export, new simulation, policy or gameplay intervention.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--curves", type=Path, nargs="+", required=True)
    args = p.parse_args()
    reference = json.loads((args.probe / "report.json").read_text())
    config = load_config(Path(reference["config"])).brain
    data = configure_runtime()
    with np.load(data / "brain.npz", allow_pickle=False) as z:
        ids, kinds, sides = z["ids"], z["cell_type"].astype(str), z["side"].astype(str)
    graph = sparse.load_npz(data / "weights.npz").tocsr()
    lookup = {int(body): i for i, body in enumerate(ids)}
    with np.load(args.probe / "counts.npz", allow_pickle=False) as z:
        cells = np.array([lookup[int(body)] for body in z["body_ids"]])
        activity = {
            cue: np.concatenate(
                [
                    z[f"{seed}_{cue}"][64:].astype(float) / config.brain_steps
                    for seed in reference["seeds"]
                ]
            )
            for cue in ("left", "right")
        }
    loc = np.full(len(ids), -1)
    loc[cells] = np.arange(len(cells))
    output = run_directory("retained-current-audit")
    report = {
        "scope": __doc__,
        "probe": str(args.probe),
        "counts_sha256": sha256(args.probe / "counts.npz"),
        "rows": [],
    }
    targets = np.flatnonzero(kinds == "DNa02")
    for curve in args.curves:
        source = json.loads((curve / "report.json").read_text())
        checkpoint = max(source["checkpoints"])
        assert load_config(Path(source["config"])).brain == config
        for arm in ("paired", "unpaired_within_cue"):
            path = curve / f"{arm}-{checkpoint}.npz"
            with np.load(path, allow_pickle=False) as z:
                pre, post, base, weights = (z[k].copy() for k in ("pre", "post", "base", "weights"))
            for target in targets:
                selected = post == target
                incoming = graph[target].toarray().ravel()[cells]
                assert (loc[pre[selected]] >= 0).all()
                delta = np.zeros(len(cells))
                delta[loc[pre[selected]]] = weights[selected] - base[selected]
                result = {
                    "curve": str(curve),
                    "training_seed": source["seed"],
                    "arm": arm,
                    "state_sha256": sha256(path),
                    "body_id": int(ids[target]),
                    "side": str(sides[target]),
                    "current": {},
                }
                for label, vector in (("original", incoming), ("retained", incoming + delta)):
                    currents = {
                        cue: rows @ (vector * config.synaptic_gain)
                        for cue, rows in activity.items()
                    }
                    means = {cue: float(rows.mean()) for cue, rows in currents.items()}
                    contrast = means["left"] - means["right"]
                    pooled = np.sqrt(np.mean([v.var(ddof=1) for v in currents.values()]))
                    result["current"][label] = {
                        "means": means,
                        "left_minus_right": contrast,
                        "contrast_over_original_sample_std": float(contrast / pooled),
                    }
                factor = weights[selected] / base[selected]
                result["factor_quantiles"] = np.quantile(factor, [0, 0.1, 0.5, 0.9, 1]).tolist()
                result["bound_fractions"] = {
                    "lower": float(np.mean(factor <= config.plasticity.minimum_factor + 1e-5)),
                    "upper": float(np.mean(factor >= config.plasticity.maximum_factor - 1e-5)),
                }
                report["rows"].append(result)
                print(result, flush=True)
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
