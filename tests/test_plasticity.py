import numpy as np
import pytest

from pokefly.plasticity import EligibilityPlasticity, PlasticityConfig


def circuit():
    return EligibilityPlasticity(
        np.array([0, 0, 1]), np.array([1, 2, 2]), np.array([0.2, -0.1, 0.3]), 3
    )


def prime(model):
    for _ in range(20):
        model.observe(np.array([0, 1]), 0.02)


def test_three_factors_are_required_and_no_synthetic_edges():
    p = circuit()
    p.reinforce(1)
    np.testing.assert_array_equal(p.weights, p.base)  # No eligible neural activity.
    prime(p)
    p.reinforce(0)
    np.testing.assert_array_equal(p.weights, p.base)
    p.reinforce(1, enabled=False)
    np.testing.assert_array_equal(p.weights, p.base)
    p.reinforce(1)
    assert p.weights[0] > p.base[0]
    assert p.weights[1] == p.base[1] and p.weights[2] == p.base[2]  # No postsynaptic activity.
    assert len(p.weights) == 3


def test_bounds_preserve_signed_wiring_and_reward_validation():
    p = circuit()
    p.eligibility[:] = [1e8, 1e8, -1e8]
    p.reinforce(1e9)
    np.testing.assert_array_equal(np.sign(p.weights), np.sign(p.base))
    np.testing.assert_allclose(p.weights / p.base, [4, 4, 0.25])
    with pytest.raises(ValueError):
        p.reinforce(float("nan"))
    with pytest.raises(ValueError):
        PlasticityConfig(maximum_factor=0.1)


def test_learning_is_retained_and_checkpoint_rejects_wrong_edges():
    p = circuit()
    prime(p)
    p.reinforce(2)
    restored = circuit()
    restored.restore(p.arrays(), p.metrics())
    for _ in range(3):
        for model in (p, restored):
            model.observe(np.array([1, 2]), 0.02)
            model.reinforce(-0.2)
        for name, values in p.arrays().items():
            np.testing.assert_array_equal(values, restored.arrays()[name])
    bad = dict(p.arrays(), pre=np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="identity"):
        restored.restore(bad, p.metrics())


def test_eligibility_decays_and_weights_change_internal_current():
    p = circuit()
    prime(p)
    before = p.eligibility.copy()
    for _ in range(1000):
        p.observe(np.array([], dtype=int), 0.02)
    assert np.max(np.abs(p.eligibility)) < np.max(np.abs(before)) * 0.01
    prime(p)
    p.reinforce(1)
    pre_spikes = np.array([1, 0, 0])
    base_current = np.bincount(p.post, p.base * pre_spikes[p.pre], minlength=3)
    learned_current = np.bincount(p.post, p.weights * pre_spikes[p.pre], minlength=3)
    assert learned_current[1] > base_current[1]
