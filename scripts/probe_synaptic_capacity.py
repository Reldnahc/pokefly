"""ROM-free oracle synaptic positive control; NEVER deploy these fitted weights.

Supervised cue labels enter a one-time OFFLINE linear program, not the learner.
Temporary existing synapses are tested in the actual recurrent fly, then
discarded. This is NOT reward learning, biological plasticity or gameplay.
No fitted weights or classifier are exported. A negative result only rejects
this first-order oracle, not the network's general representational capacity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from evaluate_learning_choices import score, warmup
from scipy.optimize import linprog

from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


class TemporaryInputs:
    """Private in-memory intervention on existing rows, never an exported learner."""

    def __init__(self, controller):
        self.c, self.b = controller, controller.brain
        self.original = self.b.weights.copy()
        self.rows = {}
        for cell in np.concatenate([controller.motors["left"], controller.motors["right"]]):
            offsets = np.flatnonzero(self.b.indices == cell)
            pre = np.searchsorted(self.b.indptr, offsets, side="right") - 1
            ptr = self.b._W.indptr.get()
            actual_pre = self.b._W.indices[ptr[cell] : ptr[cell + 1]].get()
            local = np.searchsorted(actual_pre, pre)
            np.testing.assert_array_equal(actual_pre[local], pre)
            self.rows[int(cell)] = (offsets, pre, self.b.xp.asarray(ptr[cell] + local))

    def set(self, cell, values):
        offsets, _, gpu_offsets = self.rows[int(cell)]
        values = np.asarray(values, np.float32)
        assert values.shape == offsets.shape and np.isfinite(values).all()
        np.testing.assert_array_equal(np.sign(values), np.sign(self.original[offsets]))
        self.b.weights[offsets] = values
        self.b._W.data[gpu_offsets] = self.b.xp.asarray(values)
        plastic = np.flatnonzero(self.c.plasticity.post == cell)
        self.c.plasticity.weights[plastic] = self.b.weights[self.c.csc_offsets[plastic]]

    def restore(self):
        for cell, (offsets, _, _) in self.rows.items():
            self.set(cell, self.original[offsets])
        np.testing.assert_array_equal(self.b.weights, self.original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/sensorimotor-perturb-v3.json"))
    p.add_argument("--include-inhibition", action="store_true")
    p.add_argument("--minimum-factor", type=float, default=0.25)
    p.add_argument("--maximum-factor", type=float, default=4.0)
    p.add_argument(
        "--reference-report",
        type=Path,
        help="Optional original-arm reproducibility check; not new evidence",
    )
    args = p.parse_args()
    source = json.loads((args.probe / "report.json").read_text())
    config = load_config(args.config).brain
    config = replace(
        config,
        plasticity=replace(
            config.plasticity,
            minimum_factor=args.minimum_factor,
            maximum_factor=args.maximum_factor,
        ),
    )
    source_config = load_config(Path(source["config"])).brain
    for key in (
        "brain_steps", "noise_hz", "noise_amplitude", "synaptic_gain", "dynamics",
        "intrinsic_calibration",
    ):
        assert asdict(config)[key] == asdict(source_config)[key], key
    data = configure_runtime()
    protected = {name: sha256(data / name) for name in ("brain.npz", "weights.npz")}
    c = InternalBrain(device="cuda", config=config)
    initial, initial_state = c.snapshot()
    intervention = TemporaryInputs(c)
    output = run_directory("oracle-synaptic-capacity")
    report = {
        "scope": __doc__,
        "source_probe": str(args.probe),
        "config": str(args.config),
        "recorded_activity": source.get("recorded_activity", "spike counts"),
        "counts_sha256": sha256(args.probe / "counts.npz"),
        "fit_noise_seeds": source["seeds"],
        "test_noise_seeds": [2001, 2002, 2003, 2004],
        "decisions_per_cue": 128,
        "factors": [args.minimum_factor, args.maximum_factor],
        "input_budget_fraction": 0.25,
        "allowed_synapses": "both original signs, separate E/I budgets"
        if args.include_inhibition
        else "positive only",
        "mean_input_constraint": "Preserve original predicted mean over both fit cues",
        "synthetic_neural_weights_exported": False,
        "game_used": False,
        "protected_data_sha256": protected,
        "rows": [],
    }
    reference = json.loads(args.reference_report.read_text()) if args.reference_report else None
    if reference:
        assert reference["counts_sha256"] == report["counts_sha256"]
        assert reference["test_noise_seeds"] == report["test_noise_seeds"]
        report["original_arm_repeated_validation_not_independent"] = str(args.reference_report)
    with np.load(args.probe / "counts.npz", allow_pickle=False) as archive:
        lookup = {int(body): i for i, body in enumerate(c.body_ids)}
        cells = np.array([lookup[int(body)] for body in archive["body_ids"]])
        activity = {
            cue: np.mean(
                [archive[f"{seed}_{cue}"][64:].mean(axis=0) for seed in source["seeds"]], axis=0
            )
            / c.brain_steps
            for cue in ("left", "right")
        }
    mean, contrast = (
        (activity["left"] + activity["right"]) / 2,
        activity["left"] - activity["right"],
    )
    patterns = test_patterns()
    for arm, reverse in (("original", False), ("oracle_forward", False), ("oracle_reverse", True)):
        intervention.restore()
        c.restore(initial, initial_state, weights_only=True)
        fits = []
        if arm != "original":
            for direction in ("left", "right"):
                for cell in c.motors[direction]:
                    offsets, all_pre, _ = intervention.rows[int(cell)]
                    original = intervention.original[offsets]
                    selected = original != 0 if args.include_inhibition else original > 0
                    pre = all_pre[selected]
                    positions = np.searchsorted(cells, pre)
                    np.testing.assert_array_equal(cells[positions], pre)
                    base = original[selected].astype(float)
                    mean_current, cue_current = mean[positions] * base, contrast[positions] * base
                    desired_sign = (1 if direction == "left" else -1) * (-1 if reverse else 1)
                    recorded_graded = "summed transmitter release" in report["recorded_activity"]
                    observed = (mean[positions] > 0) & (
                        True if recorded_graded else ~np.isin(pre, c.hybrid.graded_host)
                    )
                    bounds = [
                        (args.minimum_factor, args.maximum_factor) if active else (1.0, 1.0)
                        for active in observed
                    ]
                    budgets = [np.maximum(base, 0)]
                    if args.include_inhibition:
                        budgets.append(np.maximum(-base, 0))
                    budget_rows = np.stack([row for amount in budgets for row in (amount, -amount)])
                    budget_bounds = np.array(
                        [bound * amount.sum() for amount in budgets for bound in (1.25, -0.75)]
                    )
                    fit = linprog(
                        -desired_sign * cue_current,
                        A_ub=budget_rows,
                        b_ub=budget_bounds,
                        A_eq=mean_current[None, :],
                        b_eq=[mean_current.sum()],
                        bounds=bounds,
                        method="highs",
                    )
                    if not fit.success:
                        raise RuntimeError(fit.message)
                    values = original.copy()
                    values[selected] = base * fit.x
                    intervention.set(cell, values)
                    fits.append(
                        {
                            "direction": direction,
                            "body_id": int(c.body_ids[cell]),
                            "edges": len(base),
                            "active_fit_edges": int(observed.sum()),
                            "base_selected_input_contrast": float(c.brain.gain * cue_current.sum()),
                            "oracle_selected_input_contrast": float(
                                c.brain.gain * (cue_current @ fit.x)
                            ),
                            "mean_predicted_input_error": float(
                                mean_current @ fit.x - mean_current.sum()
                            ),
                            "resource_budget_ratios": [
                                float(amount @ fit.x / amount.sum())
                                for amount in budgets
                                if amount.sum()
                            ],
                        }
                    )
                    factor = values / original
                    assert factor.min() >= args.minimum_factor - 1e-6
                    assert factor.max() <= args.maximum_factor + 1e-6
        frozen = c.brain.weights.copy()
        records = []
        for seed in report["test_noise_seeds"]:
            for cue in ("left", "right"):
                warmup(c, seed)
                actions = [c.choose(c.observe(patterns[cue]))[0] for _ in range(128)]
                records.append(
                    {"cue": cue, "noise_seed": seed, "actions": dict(Counter(actions)), "n": 128}
                )
        np.testing.assert_array_equal(c.brain.weights, frozen)
        if arm == "original" and reference:
            expected = next(row for row in reference["rows"] if row["arm"] == "original")
            assert records == expected["records"], (
                "Original forward model no longer reproduces reference"
            )
        row = {
            "arm": arm,
            "reverse_mapping": reverse,
            "fit_summary": fits,
            "score": score(records, reverse),
            "records": records,
            "reverse_score": score(records, not reverse),
        }
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(arm, row["score"], flush=True)
    intervention.restore()
    c.restore(initial, initial_state, weights_only=True)
    np.testing.assert_array_equal(c.plasticity.weights, c.plasticity.base)
    assert protected == {name: sha256(data / name) for name in protected}
    report["original_weights_restored_and_source_hashes_unchanged"] = True
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
