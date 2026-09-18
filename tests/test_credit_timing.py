import importlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from pokefly.experiment import ExperimentConfig, load_config
from pokefly.plasticity import NeuralPerturbationPlasticity, PlasticityConfig


def plasticity(rule="sensorimotor-perturb-v3"):
    return NeuralPerturbationPlasticity(np.array([0, 1]), np.array([2, 2]),
                                       np.ones(2), 3, PlasticityConfig(rule=rule))


def test_latched_credit_uses_pre_action_trace_not_later_innovations():
    p = plasticity()
    p.eligibility[:] = [1, -1]
    p.latch_credit()
    with pytest.raises(ValueError, match="Previous neural credit"):
        p.latch_credit()
    p.eligibility[:] = [20, 20]  # New neural activity during the held button.
    result = p.reinforce(1)
    np.testing.assert_allclose(p.weights, 1 + .02 * np.tanh(1) * np.array([1, -1]))
    assert result["credit_timing"] == "decision-window-v1"
    assert p.pending_credit is None
    assert "pending_neural_credit" not in p.arrays()
    np.testing.assert_array_equal(p.factor_eligibility(), [20, 20])


@pytest.mark.parametrize("reward,enabled", [(0.0, True), (1.0, False)])
def test_latched_credit_does_not_defeat_no_reward_or_frozen_controls(reward, enabled):
    p = plasticity()
    p.eligibility[:] = [2, -2]
    p.latch_credit()
    p.reinforce(reward, enabled=enabled)
    np.testing.assert_array_equal(p.weights, p.base)
    assert p.pending_credit is None


def test_latched_trace_is_checkpointed_and_legacy_restore_clears_it():
    p, other = plasticity(), plasticity()
    original, state = p.arrays(), p.metrics()
    p.eligibility[:] = [1, -1]
    p.latch_credit()
    p.eligibility[:] = [10, 10]
    saved = {k: v.copy() for k, v in p.arrays().items()}
    other.restore(saved, p.metrics())
    p.reinforce(1)
    other.reinforce(1)
    np.testing.assert_array_equal(p.weights, other.weights)
    other.latch_credit()
    other.restore(original, state)
    assert other.pending_credit is None
    with pytest.raises(ValueError, match="pending neural credit"):
        other.restore({**original, "pending_neural_credit": np.array([np.nan, 1])}, state)


def test_projected_trace_latches_its_original_neural_baseline():
    p = plasticity("sensorimotor-perturb-projected-v4")
    p.post_baseline[:] = [.2, .4, 0]
    p.eligibility[:] = [2, 1]
    expected = p.factor_eligibility().copy()
    p.latch_credit()
    p.post_baseline[:] = [.4, .1, 0]
    p.eligibility[:] = [-5, 5]
    np.testing.assert_array_equal(p.factor_eligibility(), expected)
    p.reset_modulation()
    assert p.pending_credit is None


def test_credit_timing_profile_is_one_factor_and_old_checkpoints_keep_old_timing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("evaluate_visual_learning_rate")
    old = asdict(load_config(Path("configs/visual-release-wide-v3.json")))
    new = asdict(load_config(Path("configs/visual-causal-credit-v5.json")))
    assert module.validate_one_factor(old, new, "credit_timing") == (
        "feedback-boundary-v1", "decision-window-v1"
    )
    legacy = {k: v for k, v in old.items() if k != "credit_timing"}
    assert ExperimentConfig.from_dict(legacy, checkpoint=True).credit_timing == old["credit_timing"]
    new["rewards"]["new_tile"] *= 2
    with pytest.raises(ValueError, match="ONLY"):
        module.validate_one_factor(old, new, "credit_timing")
    with pytest.raises(ValueError, match="requires streamed"):
        ExperimentConfig(credit_timing="decision-window-v1")
    with pytest.raises(ValueError, match="Unknown credit timing"):
        ExperimentConfig(credit_timing="unknown")
