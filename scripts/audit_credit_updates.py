"""CPU-only audit of saved internal credit and input drift, not a learning trial.

Hypothetical single rewards test whether clipping/resource bounds reintroduce
mean input into a projected update. All changes occur on private in-memory
copies. No neural exports, emulator, button labels or policy fitting are used.
"""

import argparse

import numpy as np

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.plasticity import NeuralPerturbationPlasticity, PlasticityConfig
from pokefly.runner import run_directory, write_json


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "p05": float(np.quantile(values, .05)),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "min": float(values.min()), "max": float(values.max())}


def probe_update(p, reward):
    old = p.weights.copy()
    mean = p.post_baseline[p.pre].astype(float)
    active = np.bincount(p.post, weights=p.base * mean, minlength=p.n) > 0
    eligibility = p.factor_eligibility()
    raw_delta = p.base.astype(float) * p.config.learning_rate * (
        np.tanh(reward) - float(p.reward_mean)
    ) * eligibility
    raw_per_target = np.bincount(p.post, weights=raw_delta * mean, minlength=p.n)
    edge_absolute = float(np.sum(np.abs(raw_delta * mean)))
    p.reinforce(reward)
    actual_per_target = np.bincount(p.post, weights=(p.weights - old) * mean, minlength=p.n)
    denominator = max(edge_absolute, 1e-30)
    return {
        "hypothetical_reward": reward,
        "raw_mean_residual_fraction": float(np.abs(raw_per_target).sum() / denominator),
        "applied_mean_residual_fraction": float(np.abs(actual_per_target).sum() / denominator),
        "applied_signed_input_sum": float(actual_per_target.sum()),
        "per_target_applied_input_delta": summary(actual_per_target[active]),
        "edges_at_lower_bound": int(np.count_nonzero(np.isclose(
            p.weights / p.base, p.config.minimum_factor, rtol=0, atol=1e-6))),
        "edges_at_upper_bound": int(np.count_nonzero(np.isclose(
            p.weights / p.base, p.config.maximum_factor, rtol=0, atol=1e-6))),
    }


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    output = run_directory("credit-update-audit")
    report = {"scope": __doc__, "status": "running", "rows": []}
    for path in args.checkpoints:
        arrays, saved = read_checkpoint(path)
        config = PlasticityConfig(**saved["neural"]["identity"]["config"]["plasticity"])
        if config.rule not in ("sensorimotor-perturb-v3", "sensorimotor-perturb-projected-v4"):
            raise ValueError("Audit supports only the registered paired candidate rules")
        p = NeuralPerturbationPlasticity(
            arrays["pre"], arrays["post"], arrays["base"], len(arrays["post_baseline"]), config,
        )
        p.restore(arrays, saved["neural"]["plasticity"])
        mean = p.post_baseline[p.pre].astype(float)
        base_input = np.bincount(p.post, weights=p.base * mean, minlength=p.n)
        learned_input = np.bincount(p.post, weights=p.weights * mean, minlength=p.n)
        active = base_input > 0
        row = {
            "checkpoint": saved["directory"], "sample": saved["experiment"]["sample"],
            "rule": config.rule, "estimated_mean_input_relative_change": summary(
                (learned_input[active] - base_input[active]) / base_input[active]
            ), "hypothetical_updates": [],
        }
        for reward in (0.0, 0.05, 1.0):
            p.restore(arrays, saved["neural"]["plasticity"])
            row["hypothetical_updates"].append(probe_update(p, reward))
        p.restore(arrays, saved["neural"]["plasticity"])
        np.testing.assert_array_equal(p.weights, arrays["weights"])
        row["source_brain_sha256"] = sha256(Path(saved["directory"]) / "brain.npz")
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(row, flush=True)
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
