from dataclasses import replace

import numpy as np
import pytest

from pokefly.dynamics import DynamicsConfig
from pokefly.internal_brain import BrainConfig
from pokefly.plasticity import (
    NeuralPerturbationPlasticity,
    PlasticityConfig,
    SensorimotorPlasticity,
)


def learner(normalize=False):
    return SensorimotorPlasticity(
        np.array([0, 1]),
        np.array([2, 2]),
        np.array([0.2, 0.3]),
        3,
        PlasticityConfig(
            rule="sensorimotor-rstdp-v1",
            activity_reference_hz=1,
            eligibility_seconds=0.6,
            normalize_inputs=normalize,
        ),
    )


def test_reward_uses_neural_coactivity_without_action_information():
    p = learner()
    for _ in range(100):
        p.observe(np.array([0, 2]), 0.02)
    p.reinforce(1)
    assert p.weights[0] > p.base[0]
    assert p.weights[1] == p.base[1]
    assert p.reward_mean > 0
    p.reinforce(0)
    assert p.prediction_error < 0


@pytest.mark.parametrize("frozen", [False, True])
def test_no_reinforcement_and_frozen_controls_preserve_weights(frozen):
    p = learner(True)
    for _ in range(80):
        p.observe(np.array([0, 2]), 0.02)
        p.reinforce(1 if frozen else 0, enabled=not frozen)
    np.testing.assert_array_equal(p.weights, p.base)
    assert p.reward_mean == 0


def test_input_resource_competition_preserves_bounds_and_budget():
    p = learner(True)
    for _ in range(80):
        for _ in range(12):
            p.observe(np.array([0, 2]), 0.02)
        p.reinforce(1)
    assert p.weights[0] > p.base[0] and p.weights[1] < p.base[1]
    assert np.isclose(p.weights.sum(), p.base.sum(), rtol=1e-4)
    factors = p.weights / p.base
    assert np.all(factors >= p.config.minimum_factor)
    assert np.all(factors <= p.config.maximum_factor)


def test_bounded_input_budget_allows_selectivity_without_unbounded_motor_gain():
    p = learner()
    p.config = replace(p.config, input_budget_fraction=0.25)
    for _ in range(200):
        for _ in range(12):
            p.observe(np.array([0, 2]), 0.02)
        p.reinforce(1)
    assert p.weights[0] > p.base[0] * 1.25
    assert p.weights.sum() <= p.base.sum() * 1.2501
    assert p.weights.sum() >= p.base.sum() * 0.7499
    assert np.all(p.weights / p.base >= p.config.minimum_factor)
    assert np.all(p.weights / p.base <= p.config.maximum_factor)


def test_feedback_state_restores_exactly():
    p, q = learner(True), learner(True)
    p.observe(np.array([0, 2]), 0.02)
    p.reinforce(1)
    q.restore(p.arrays(), p.metrics())
    for reward in [0, 1, -1, 0] * 10:
        for model in (p, q):
            model.observe(np.array([1, 2]), 0.02)
            model.reinforce(reward)
        for key in p.arrays():
            np.testing.assert_array_equal(p.arrays()[key], q.arrays()[key])
    with pytest.raises(ValueError, match="bounds"):
        q.restore(dict(p.arrays(), reward_mean=np.array(2.0)), p.metrics())


def test_intrinsic_calibration_is_explicit_and_cannot_change_baseline_silently():
    assert BrainConfig().intrinsic_calibration is None
    with pytest.raises(ValueError, match="hybrid"):
        BrainConfig(intrinsic_calibration="candidate.npz")
    c = BrainConfig(dynamics=DynamicsConfig(profile="hybrid-v1"))
    assert replace(c, intrinsic_calibration="candidate.npz").intrinsic_calibration
    with pytest.raises(ValueError, match="explicit file"):
        replace(c, intrinsic_calibration="")


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_input_normalization_requires_a_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        PlasticityConfig(normalize_inputs=value)


def test_full_sensorimotor_updates_match_numpy_reference(monkeypatch):
    import pokefly.fast_plasticity as fast

    kernel = fast.sensorimotor_eligibility

    def reference(pre, post, trace, baseline, spike, eligibility, scale, decay):
        incoming = np.clip(trace[pre] / scale, 0, 5)
        outgoing = (spike[post] - baseline[post]) / scale
        eligibility *= decay
        eligibility += (1 - decay) * incoming * outgoing
        np.clip(eligibility, -5, 5, out=eligibility)

    rng = np.random.default_rng(927)
    n, edges = 1000, 5000
    args = (
        rng.integers(n, size=edges),
        rng.integers(n, size=edges),
        rng.uniform(0.001, 0.1, edges),
        n,
    )
    cfg = PlasticityConfig(rule="sensorimotor-rstdp-v1", activity_reference_hz=1)
    p, q = SensorimotorPlasticity(*args, cfg), SensorimotorPlasticity(*args, cfg)
    for _ in range(20):
        for _ in range(12):
            fired = np.flatnonzero(rng.random(n) < 0.04)
            monkeypatch.setattr(fast, "sensorimotor_eligibility", kernel)
            p.observe(fired, 0.02)
            monkeypatch.setattr(fast, "sensorimotor_eligibility", reference)
            q.observe(fired, 0.02)
        reward = rng.choice([0, 0.05, 1.0])
        assert p.reinforce(reward) == q.reinforce(reward)
        for key in p.arrays():
            np.testing.assert_array_equal(p.arrays()[key], q.arrays()[key])


def test_perturbation_credit_uses_actual_neural_noise_not_output_action():
    p = NeuralPerturbationPlasticity(
        np.array([0, 1]),
        np.array([2, 2]),
        np.array([0.2, 0.3]),
        3,
        PlasticityConfig(rule="sensorimotor-perturb-v1", activity_reference_hz=1),
    )
    for _ in range(50):
        p.observe(np.array([0]), 0.02, perturbation=np.array([0, 0, 1]), probability=0.024)
    p.reinforce(1)
    assert p.weights[0] > p.base[0]
    assert p.weights[1] == p.base[1]
    with pytest.raises(ValueError, match="Actual neural perturbations"):
        p.observe(np.array([0]), 0.02)
    with pytest.raises(ValueError, match="mask"):
        p.observe(np.array([0]), 0.02, perturbation=np.array([0, 0, 0.5]), probability=0.024)


def test_perturbation_capture_requires_supported_dynamics():
    with pytest.raises(ValueError, match="requires hybrid"):
        BrainConfig(plasticity=PlasticityConfig(rule="sensorimotor-perturb-v1"))
