from dataclasses import replace

import numpy as np
import pytest

from pokefly.dynamics import DynamicsConfig
from pokefly.fast_plasticity import likelihood_eligibility
from pokefly.internal_brain import BrainConfig
from pokefly.plasticity import PlasticityConfig
from pokefly.score_plasticity import LikelihoodPlasticity


def learner():
    return LikelihoodPlasticity(
        np.array([0]),
        np.array([1]),
        np.array([0.13], np.float32),
        2,
        PlasticityConfig(rule="sensorimotor-score-v1", learning_rate=0.1),
    )


def test_local_score_matches_finite_difference_of_complete_spike_history():
    # Fixed noise/input/spike trajectory: derivative includes membrane history
    # and both the spike and silence likelihood; postsynaptic reset is essential.
    release = np.array([1.0, 0.0, 0.3, 1.0, 0.0, 0.8, 0.0])
    fired = np.array([0, 0, 1, 0, 0, 1, 0])
    decay, gain, temperature, base = 0.82, 3.0, 0.15, 0.13

    def log_likelihood(factor, collect=False):
        voltage, total, probabilities = 0.0, 0.0, []
        for incoming, spike in zip(release, fired, strict=True):
            voltage = decay * voltage + gain * base * factor * incoming + 0.14
            p = 1 / (1 + np.exp(-(voltage - 1) / temperature))
            probabilities.append(p)
            total += np.log(p if spike else 1 - p)
            if spike:
                voltage = 0.0
        return probabilities if collect else total

    membrane, eligibility = np.zeros(1), np.zeros(1)
    for incoming, spike, p in zip(release, fired, log_likelihood(1, True), strict=True):
        likelihood_eligibility(
            np.array([0]),
            np.array([1]),
            np.array([base]),
            np.array([incoming, 0.0]),
            np.array([0.0, spike]),
            np.array([0.0, p]),
            membrane,
            eligibility,
            decay,
            1.0,
            gain,
            temperature,
        )
    eps = 1e-5
    expected = (log_likelihood(1 + eps) - log_likelihood(1 - eps)) / (2 * eps)
    assert eligibility[0] == pytest.approx(expected, rel=1e-8)


def test_expected_instantaneous_score_is_zero_and_has_no_cue_labels():
    p = 0.13
    scores = []
    for spike in [0, 1]:
        e = np.zeros(1)
        likelihood_eligibility(
            np.array([0]),
            np.array([1]),
            np.array([0.2]),
            np.array([0.7, 0]),
            np.array([0, spike]),
            np.array([0, p]),
            np.zeros(1),
            e,
            0.8,
            0.9,
            3.0,
            0.05,
        )
        scores.append(e[0])
    assert (1 - p) * scores[0] + p * scores[1] == pytest.approx(0, abs=1e-12)


def test_v2_preconditioning_uses_only_original_synapse_size():
    p = learner()
    observe(p)
    raw_score = p.eligibility.copy()
    p.config = replace(p.config, rule="sensorimotor-score-v2")
    np.testing.assert_array_equal(p.factor_eligibility(), raw_score / p.base)
    before = p.weights.copy()
    p.reinforce(1)
    np.testing.assert_allclose(
        p.weights - before, p.config.learning_rate * np.tanh(1) * raw_score, rtol=1e-6
    )


def centered_learner():
    return LikelihoodPlasticity(
        np.array([0]),
        np.array([1]),
        np.array([0.13], np.float32),
        2,
        PlasticityConfig(rule="sensorimotor-score-centered-v4", learning_rate=0.002),
    )


def test_centered_innovation_uses_prior_release_average_not_stimulus_labels():
    p = centered_learner()
    p.release_baseline[0] = 0.4
    observe(p)
    assert p.eligibility[0] == pytest.approx(0, abs=1e-7)  # float32 stored baseline
    p.release_baseline[0] = 0.6
    observe(p)
    assert p.eligibility[0] < 0
    assert 0.4 < p.release_baseline[0] < 0.6
    np.testing.assert_array_equal(p.factor_eligibility(), p.eligibility / p.base)


def test_centered_innovation_has_conditional_zero_mean_for_spike_and_silence():
    scores = []
    for fired in ([], [1]):
        p = centered_learner()
        p.release_baseline[0] = 0.7
        p.observe(
            np.array(fired, dtype=int),
            0.02,
            release=np.array([0.4, 0]),
            probability=np.array([0, 0.13]),
            membrane_decay=0.82,
            gain=3,
            temperature=0.05,
        )
        scores.append(p.eligibility[0])
    assert 0.87 * scores[0] + 0.13 * scores[1] == pytest.approx(0, abs=1e-7)


def test_centered_negative_trace_checkpoint_and_reset_are_exact():
    p = centered_learner()
    p.release_baseline[:] = 0.7
    p.observe(
        np.array([], dtype=int),
        0.02,
        release=np.array([0.4, 0]),
        probability=np.array([0, 0.13]),
        membrane_decay=0.82,
        gain=3,
        temperature=0.05,
    )
    assert p.membrane_trace[0] < 0
    q = centered_learner()
    q.restore({k: v.copy() for k, v in p.arrays().items()}, p.metrics())
    for _ in range(4):
        observe(p)
        observe(q)
        p.reinforce(0.4)
        q.reinforce(0.4)
    for key in p.arrays():
        np.testing.assert_array_equal(p.arrays()[key], q.arrays()[key])
    broken = {k: v.copy() for k, v in q.arrays().items()}
    broken["release_baseline"][0] = 2
    with pytest.raises(ValueError, match="baseline"):
        p.restore(broken, q.metrics())
    p.reset_modulation()
    assert not p.release_baseline.any() and not p.membrane_trace.any()


def test_signed_score_preserves_signs_and_separate_inhibitory_resource_budget():
    p = LikelihoodPlasticity(
        np.array([0, 1, 0, 1]),
        np.array([2, 2, 2, 2]),
        np.array([0.2, 0.3, -0.1, -0.4], np.float32),
        3,
        PlasticityConfig(
            rule="sensorimotor-score-v3", learning_rate=0.01, input_budget_fraction=0.25
        ),
    )
    for _ in range(20):
        p.observe(
            np.array([2]),
            0.02,
            release=np.array([0.4, 0.2, 0.0]),
            probability=np.array([0.0, 0.0, 0.1]),
            membrane_decay=0.82,
            gain=3.0,
            temperature=0.05,
        )
        p.reinforce(1)
    assert p.weights[0] > p.base[0]
    assert abs(p.weights[2]) < abs(p.base[2])  # Positive outcome weakens active inhibition.
    np.testing.assert_array_equal(np.sign(p.weights), np.sign(p.base))
    for mask in (p.base > 0, p.base < 0):
        ratio = abs(p.weights[mask]).sum() / abs(p.base[mask]).sum()
        assert 0.75 - 1e-5 <= ratio <= 1.25 + 1e-5
    q = LikelihoodPlasticity(p.pre, p.post, p.base, 3, p.config)
    q.restore(p.arrays(), p.metrics())
    np.testing.assert_array_equal(q.weights, p.weights)


def observe(p):
    p.observe(
        np.array([1]),
        0.02,
        release=np.array([0.4, 0.0]),
        probability=np.array([0.0, 0.1]),
        membrane_decay=0.82,
        gain=3.0,
        temperature=0.05,
    )


def test_reward_frozen_zero_reward_reset_and_snapshot():
    p = learner()
    observe(p)
    p.reinforce(0)
    np.testing.assert_array_equal(p.base, p.weights)
    p.reinforce(1, enabled=False)
    np.testing.assert_array_equal(p.base, p.weights)
    p.reinforce(1)
    assert p.weights[0] > p.base[0]
    q = learner()
    q.restore({k: v.copy() for k, v in p.arrays().items()}, p.metrics())
    for _ in range(4):
        observe(p)
        observe(q)
        p.reinforce(0.4)
        q.reinforce(0.4)
    for k in p.arrays():
        np.testing.assert_array_equal(p.arrays()[k], q.arrays()[k])
    p.membrane_trace[:] = 1
    p.reset_modulation()
    assert not np.any(p.membrane_trace)


def test_score_rule_requires_matching_stochastic_model():
    config = PlasticityConfig(rule="sensorimotor-score-v1")
    with pytest.raises(ValueError):
        BrainConfig(plasticity=config)
    assert BrainConfig(
        plasticity=config,
        dynamics=DynamicsConfig(profile="hybrid-v1", spike_temperature=0.05),
    )
    with pytest.raises(ValueError):
        BrainConfig(
            plasticity=replace(config, slow_eligibility_seconds=30),
            dynamics=DynamicsConfig(profile="hybrid-v1", spike_temperature=0.05),
        )


def test_toy_two_cue_circuit_learns_with_positive_scalar_reward():
    # Algorithm unit test ONLY: this four-cell toy is never the fly controller.
    # A cue is supplied as two alternate presynaptic transmitter patterns.
    p = LikelihoodPlasticity(
        np.array([0, 1, 0, 1]),
        np.array([2, 2, 3, 3]),
        np.full(4, 0.05, np.float32),
        4,
        PlasticityConfig(rule="sensorimotor-score-v1", learning_rate=0.5, eligibility_seconds=0.6),
    )
    rng = np.random.default_rng(123)
    voltage = np.zeros(4)
    correct, choices = 0, 0
    for decision in range(1600):
        cue = (decision // 16) % 2
        release = np.zeros(4)
        release[cue] = 0.25
        counts = np.zeros(4)
        for _ in range(12):
            incoming = np.bincount(p.post, weights=p.weights * release[p.pre], minlength=4)
            voltage = 0.82 * voltage + 3 * incoming + 0.14
            probability = 1 / (1 + np.exp(-(voltage - 1) / 0.05))
            probability[:2] = 0
            fired = np.flatnonzero(rng.random(4) < probability)
            counts[fired] += 1
            voltage[fired] = 0
            p.observe(
                fired,
                0.02,
                release=release,
                probability=probability,
                membrane_decay=0.82,
                gain=3.0,
                temperature=0.05,
            )
        choice = np.argmax(counts[2:]) if counts[2] != counts[3] else None
        if decision < 1200:
            p.reinforce(float(choice == cue))
        elif choice is not None:
            choices += 1
            correct += choice == cue
    assert choices >= 200
    assert correct / choices > 0.8
