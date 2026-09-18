"""Offline frozen-input capacity bound; NEVER deploy the optimized weights.

This first-order calculation holds presynaptic activity fixed. It tests whether
the allowed positive-edge bounds can even reverse mean direct cue preference;
it does not prove a recurrent network will realize or learn that preference.
Only scalar bounds, not fitted weights, are exported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[801, 802, 803, 804])
    args = p.parse_args()
    data = configure_runtime()
    with np.load(data / "brain.npz", allow_pickle=False) as archive:
        ids, kinds, sides = archive["ids"], archive["cell_type"].astype(str), archive["side"]
    # The disk matrix is CSR; runtime propagation exposes CSC outgoing columns.
    matrix = sparse.load_npz(data / "weights.npz").tocsc()
    with np.load(args.probe / "counts.npz", allow_pickle=False) as archive:
        lookup = {int(value): index for index, value in enumerate(ids)}
        cells = np.array([lookup[int(value)] for value in archive["body_ids"]])
        difference = (
            np.mean(
                [
                    archive[f"{seed}_left"][64:].mean(axis=0)
                    - archive[f"{seed}_right"][64:].mean(axis=0)
                    for seed in args.seeds
                ],
                axis=0,
            )
            / 12
        )
    rows = []
    for target in np.flatnonzero(kinds == "DNa02"):
        edges = np.flatnonzero(matrix.indices == target)
        pre = np.searchsorted(matrix.indptr, edges, side="right") - 1
        positions = np.searchsorted(cells, pre)
        np.testing.assert_array_equal(cells[positions], pre)
        response, weights = difference[positions], matrix.data[edges]
        positive = weights > 0
        effect = weights[positive] * response[positive]
        fixed = float(weights[~positive] @ response[~positive])
        amount = weights[positive]
        for name, low, high, budget in (
            ("bounded", 0.75, 1.25, None),
            ("input_budget", 0.25, 4.0, 0.25),
            ("exact_input_budget", 0.25, 4.0, 0.0),
        ):
            bounds = []
            for direction in (1, -1):
                fit = linprog(
                    direction * effect,
                    A_ub=np.stack([amount, -amount]) if budget is not None else None,
                    b_ub=np.array([1 + budget, -(1 - budget)]) * amount.sum()
                    if budget is not None
                    else None,
                    bounds=(low, high),
                    method="highs",
                )
                if not fit.success:
                    raise RuntimeError(fit.message)
                bounds.append(3 * (fixed + float(effect @ fit.x)))
            rows.append(
                {
                    "side": str(sides[target]),
                    "body_id": int(ids[target]),
                    "profile": name,
                    "base_left_minus_right_current": float(3 * (weights @ response)),
                    "min_max_left_minus_right_current": bounds,
                    "direct_sign_reversal_possible": bounds[0] < 0 < bounds[1],
                }
            )
    output = run_directory("visual-capacity-audit")
    write_json(
        output / "report.json",
        {
            "protocol": __doc__,
            "source_probe": str(args.probe),
            "seeds": args.seeds,
            "neural_steps_per_decision": 12,
            "gain": 3.0,
            "rows": rows,
            "fitted_weights_exported_or_deployed": False,
        },
    )
    print(rows)
    print("Report:", output)


if __name__ == "__main__":
    main()
