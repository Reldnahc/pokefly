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


def test_linear_perturbation_trace_is_conditionally_zero_mean():
    means = {}
    for rule in ("sensorimotor-perturb-v1", "sensorimotor-perturb-v2", "sensorimotor-perturb-v3"):
        outcomes = []
        for event in (False, True):
            p = NeuralPerturbationPlasticity(
                np.array([0]),
                np.array([1]),
                np.array([0.2]),
                2,
                PlasticityConfig(rule=rule, activity_reference_hz=1, eligibility_seconds=0.6),
            )
            p.pre_trace[0] = 0.1
            p.observe(
                np.array([], dtype=int),
                0.02,
                perturbation=np.array([False, event]),
                probability=0.024,
            )
            outcomes.append(float(p.eligibility[0]))
        means[rule] = 0.976 * outcomes[0] + 0.024 * outcomes[1]
    assert abs(means["sensorimotor-perturb-v2"]) < 1e-6
    assert abs(means["sensorimotor-perturb-v3"]) < 1e-6
    assert means["sensorimotor-perturb-v1"] < -0.01


def test_centered_input_perturbation_uses_past_local_activity_and_allows_negative_contrast():
    p = NeuralPerturbationPlasticity(
        np.array([0]),
        np.array([1]),
        np.array([0.2]),
        2,
        PlasticityConfig(rule="sensorimotor-perturb-v3", activity_reference_hz=1),
    )
    p.pre_trace[0] = 0.01
    p.post_baseline[0] = 0.05
    p.observe(np.array([0]), 0.02, perturbation=np.array([False, True]), probability=0.024)
    assert p.eligibility[0] < 0  # This step's presynaptic spike is not read early.


def test_dual_eligibility_retains_delayed_activity_and_restores_exactly():
    cfg = replace(learner().config, slow_eligibility_seconds=30)
    args = (np.array([0]), np.array([1]), np.array([0.2]), 2, cfg)
    p, q = SensorimotorPlasticity(*args), SensorimotorPlasticity(*args)
    for _ in range(12):
        p.observe(np.array([0, 1]), 0.02)
    for _ in range(500):
        p.observe(np.array([], dtype=int), 0.02)
    assert abs(p.slow_eligibility[0]) > abs(p.eligibility[0]) * 100
    q.restore(p.arrays(), p.metrics())
    for reward in (1, 0, 0, 1):
        for model in (p, q):
            model.observe(np.array([0, 1]), 0.02)
            model.reinforce(reward)
        for key in p.arrays():
            np.testing.assert_array_equal(p.arrays()[key], q.arrays()[key])
    assert p.weights[0] != p.base[0]
    p.reset_modulation()
    assert not p.slow_eligibility.any()


def test_centered_covariance_credits_past_input_contrast_not_current_cofiring():
    p = SensorimotorPlasticity(
        np.array([0]),
        np.array([1]),
        np.array([0.2]),
        2,
        PlasticityConfig(rule="sensorimotor-rstdp-v2", activity_reference_hz=1),
    )
    p.pre_trace[0], p.post_baseline[0] = 0.01, 0.05
    p.observe(np.array([0, 1]), 0.02)
    assert p.eligibility[0] < 0
    p.reinforce(1)
    assert p.weights[0] < p.base[0]


def test_impulse_balanced_mix_has_equal_event_gains_and_preserves_white_noise_variance():
    from scipy.integrate import quad

    fast, slow = 0.6, 30.0
    ratio = slow / fast
    scale = 1 / np.sqrt(1 + ratio + 4 * ratio / (1 + ratio))
    assert 1 / fast == pytest.approx(ratio / slow)
    variance, _ = quad(
        lambda t: (scale * (np.exp(-t / fast) / fast + ratio * np.exp(-t / slow) / slow)) ** 2,
        0,
        np.inf,
    )
    assert variance == pytest.approx(1 / (2 * fast), rel=1e-9)
    config = replace(
        learner().config,
        eligibility_seconds=fast,
        slow_eligibility_seconds=slow,
        trace_mixing="impulse-balanced-v2",
    )
    p = SensorimotorPlasticity(np.array([0]), np.array([1]), np.array([0.2]), 2, config)
    p.eligibility[:] = np.exp(-7.68 / fast) / fast
    p.slow_eligibility[:] = np.exp(-7.68 / slow) / slow
    old_mix = 0.5 * (p.eligibility + p.slow_eligibility)
    assert p.factor_eligibility()[0] > 10 * old_mix[0]
    q = SensorimotorPlasticity(p.pre, p.post, p.base, 2, config)
    q.restore(p.arrays(), p.metrics())
    np.testing.assert_array_equal(p.factor_eligibility(), q.factor_eligibility())
    for model in (p, q):
        model.reinforce(1)
    np.testing.assert_array_equal(p.weights, q.weights)


def test_new_mix_requires_two_traces_and_legacy_mix_remains_exact():
    with pytest.raises(ValueError, match="requires a slow"):
        PlasticityConfig(trace_mixing="impulse-balanced-v2")
    with pytest.raises(ValueError, match="Unknown eligibility"):
        PlasticityConfig(trace_mixing="guess")
    p = SensorimotorPlasticity(
        np.array([0]),
        np.array([1]),
        np.array([0.2]),
        2,
        replace(learner().config, slow_eligibility_seconds=30),
    )
    p.eligibility[:] = 0.031234
    p.slow_eligibility[:] = -0.391182
    np.testing.assert_array_equal(
        p.factor_eligibility(), 0.5 * (p.eligibility + p.slow_eligibility)
    )


def test_hebbian_event_tag_stays_positive_after_spiking_and_uses_reward_centering():
    config = replace(
        learner().config,
        rule="sensorimotor-event-v1",
        eligibility_seconds=0.6,
        slow_eligibility_seconds=30,
        trace_mixing="impulse-balanced-v2",
    )
    p = SensorimotorPlasticity(np.array([0]), np.array([1]), np.array([0.2]), 2, config)
    p.observe(np.array([0, 1]), 0.02)
    for _ in range(384):
        p.observe(np.array([], dtype=int), 0.02)
    assert p.eligibility[0] > 0 and p.slow_eligibility[0] > 0
    p.reinforce(1)
    assert p.weights[0] > p.base[0]
    assert p.reward_mean > 0
    p.reinforce(0)
    assert p.prediction_error < 0  # Internal expectation, not a new game penalty.
    assert "hebbian-event" in p.metrics()["rule"]
