import importlib
from pathlib import Path

import numpy as np

from pokefly.plasticity import NeuralPerturbationPlasticity, PlasticityConfig


def test_finite_update_audit_detects_projection_broken_by_a_bound(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("audit_credit_updates")
    p = NeuralPerturbationPlasticity(np.array([0, 1]), np.array([2, 2]), np.ones(2), 3,
                                    PlasticityConfig(rule="sensorimotor-perturb-projected-v4",
                                                     input_budget_fraction=.25))
    p.post_baseline[:] = [.2, .4, 0]
    p.eligibility[:] = [-2, 1]
    p.weights[:] = [.25, 1.5]
    result = module.probe_update(p, 1.0)
    assert result["raw_mean_residual_fraction"] < 1e-6
    assert result["applied_mean_residual_fraction"] > .4
    assert result["applied_signed_input_sum"] > 0
