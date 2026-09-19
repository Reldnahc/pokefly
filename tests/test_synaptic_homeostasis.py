import numpy as np
import pytest

from pokefly.plasticity import (
    NeuralPerturbationPlasticity,
    PlasticityConfig,
    SensorimotorPlasticity,
)
from pokefly.synaptic_homeostasis import bounded_mean_factors, target_order


def constrain(proposed, base, mean, post, n, budget=.25):
    return bounded_mean_factors(
        np.asarray(proposed, np.float32), np.asarray(base, np.float32),
        np.asarray(mean, np.float32), *target_order(np.asarray(post), n), .25, 4., budget,
    )


def test_bounds_do_not_reintroduce_mean_drive_and_accumulated_drift_is_removed():
    base, mean, post = np.ones(2, np.float32), np.array([.2, .4], np.float32), [2, 2]
    result = constrain([-.5, 1.6], base, mean, post, 3)
    assert (result >= .25).all() and (result <= 4).all()
    assert result[0] == .25
    np.testing.assert_allclose(base * mean @ (result - 1), 0, atol=2e-8)
    assert result[1] < 1.6
    # Moving mean references constrain total learned weights, not only a step.
    result = constrain(result, base, [.4, .2], post, 3)
    np.testing.assert_allclose(np.array([.4, .2]) @ (result - 1), 0, atol=2e-8)


@pytest.mark.parametrize("budget", [-1, 0, .05, .25])
def test_large_random_updates_preserve_mean_box_and_resource_constraints(budget):
    rng = np.random.default_rng(681)
    n, edges = 101, 1200
    post = rng.integers(n - 3, size=edges)  # Empty targets are harmless.
    base = rng.uniform(.001, 1, edges).astype(np.float32)
    mean = rng.uniform(0, 1, edges).astype(np.float32)
    mean[::7] = 0
    proposed = rng.normal(1, 30, edges).astype(np.float32)
    result = constrain(proposed, base, mean, post, n, budget)
    assert np.isfinite(result).all()
    assert ((result >= .25) & (result <= 4)).all()
    reference = np.bincount(post, weights=base.astype(float) * mean, minlength=n)
    actual = np.bincount(post, weights=base.astype(float) * mean * result, minlength=n)
    np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=1e-10)
    if budget >= 0:
        original = np.bincount(post, weights=base, minlength=n)
        resource = np.bincount(post, weights=base.astype(float) * result, minlength=n)
        assert (resource <= original * (1 + budget + 1e-7)).all()
        assert (resource >= original * (1 - budget - 1e-7)).all()


def test_zero_release_adds_no_fabricated_constraint_and_targets_are_local():
    base, mean, post = np.array([.3, .2, .4]), [0, 0, .2], [1, 1, 2]
    proposed = np.array([1.2, .5, 1.8])
    first = constrain(proposed, base, mean, post, 3, -1)
    np.testing.assert_array_equal(first[:2], proposed[:2].astype(np.float32))
    assert first[2] == 1  # A single nonzero input has no mean-preserving direction.
    changed = constrain([-10, 10, 1.8], base, mean, post, 3, -1)
    assert changed[2] == first[2]


def test_constraint_is_idempotent_at_float_precision_and_keeps_original_weights():
    base, mean, post = [.2, .7, .3, .4], [.01, .04, .4, .8], [2, 2, 3, 3]
    first = constrain([-2, 2, 3, .3], base, mean, post, 4)
    np.testing.assert_allclose(constrain(first, base, mean, post, 4), first, atol=1e-7)
    np.testing.assert_array_equal(constrain(np.ones(4), base, mean, post, 4), np.ones(4))


def test_moving_context_constraint_can_erase_selectivity_unlike_a_fixed_reference():
    # Capacity warning, NOT a fly learning experiment: these are hand-set toy
    # factors. Repeatedly enforcing ORIGINAL mean input in each distinct
    # context can remove a useful contrast while satisfying every numeric bound.
    # A fixed mixed-context reference has a different feasible set. No such
    # frozen reference has yet been calibrated or installed in a game model.
    chosen = np.array([1.5, .5], np.float32)
    moving, fixed = chosen.copy(), chosen.copy()
    for _ in range(32):
        for context in ([.8, .2], [.2, .8]):
            moving = constrain(moving, [1., 1.], context, [0, 0], 1)
            fixed = constrain(fixed, [1., 1.], [.5, .5], [0, 0], 1)
    np.testing.assert_allclose(moving, np.ones(2), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(fixed, chosen)


@pytest.mark.parametrize("tiny", [1e-20, 1e-35, 1e-45])
def test_nearly_silent_release_means_do_not_corrupt_dual_bracketing(tiny):
    base = np.array([.1, .2, .4, .7], np.float32)
    mean = np.array([.1, tiny, tiny, tiny], np.float32)
    post = np.array([0, 0, 1, 1])
    result = constrain([1.8, .4, .5, 2.], base, mean, post, 2)
    reference = np.bincount(post, weights=base.astype(float) * mean)
    actual = np.bincount(post, weights=base.astype(float) * mean * result)
    np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=0)


def learner(rule="sensorimotor-perturb-homeostatic-v5", **options):
    return NeuralPerturbationPlasticity(
        np.array([0, 1, 0, 1]), np.array([2, 2, 3, 3]),
        np.array([.1, .2, .3, .4]), 4,
        PlasticityConfig(rule=rule, learning_rate=.2, input_budget_fraction=.25),
        **options,
    )


def anchored(reference=None):
    return learner("sensorimotor-perturb-anchored-v6", fixed_release_reference=reference)


@pytest.mark.parametrize("reference", [
    None, [0., 0., 0., 0.], [.2, .3], [[.2], [.3], [.4], [.5]],
    [.2, .3, np.nan, .5], [.2, .3, np.inf, .5], [-.1, .3, .4, .5], [1.1, .3, .4, .5],
])
def test_anchored_rule_requires_a_valid_physical_reference(reference):
    with pytest.raises(ValueError, match="fixed release reference"):
        anchored(reference)
    with pytest.raises(ValueError, match="explicit anchored rule"):
        learner(fixed_release_reference=np.ones(4))


def test_fixed_reference_is_copied_readonly_and_independent_of_game_context():
    reference = np.full(4, .5, np.float32)
    model = anchored(reference)
    reference[:] = 1
    np.testing.assert_array_equal(model.fixed_release_reference, np.full(4, .5, np.float32))
    with pytest.raises(ValueError, match="read-only"):
        model.fixed_release_reference[0] = 1
    arrays = model.arrays()
    arrays["fixed_release_reference"][:] = 0
    assert model.fixed_release_reference[0] == .5
    chosen = np.array([1.5, .75, 1.4, .7], np.float32)
    for context in ([.8, .2, 0, 0], [.2, .8, 1, 1]):
        model.post_baseline[:] = context
        np.testing.assert_allclose(model.constrain_factors(chosen), chosen, rtol=0, atol=1e-7)


def test_anchored_rule_does_not_change_physical_credit_observation():
    candidate, original = anchored(np.full(4, .5)), learner()
    for model in (candidate, original):
        for _ in range(4):
            model.observe(np.array([3]), .02, perturbation=np.array([0, 1, 0, 1]),
                          probability=.024, graded_release=(np.array([0, 1]), [.2, .8]))
    for key, value in original.arrays().items():
        np.testing.assert_array_equal(value, candidate.arrays()[key])
    assert set(candidate.arrays()) - set(original.arrays()) == {"fixed_release_reference"}


@pytest.mark.parametrize("reward,enabled", [(1., False), (0., True)])
def test_anchored_reference_never_normalizes_frozen_or_zero_error_weights(reward, enabled):
    model = anchored(np.full(4, .5))
    model.weights *= np.array([.5, 1.8, 1.5, .7])
    model.eligibility[:] = 10
    before = model.weights.copy()
    model.reinforce(reward, enabled=enabled)
    np.testing.assert_array_equal(before, model.weights)


def test_anchored_learning_and_checkpoint_resume_are_exact():
    p, q = anchored([.02, .08, .04, .03]), anchored([.02, .08, .04, .03])
    p.eligibility[:] = [-30, 8, 20, -20]
    p.feedback_elapsed[...] = .24
    p.reinforce(.5)
    assert p.last_changed > 0
    q.restore({k: v.copy() for k, v in p.arrays().items()}, p.metrics())
    assert not q.fixed_release_reference.flags.writeable
    for reward in (0., .05, 1., -.2):
        for model in (p, q):
            model.observe(np.array([0, 2]), .02,
                          perturbation=np.array([0, 0, 1, 0]), probability=.024)
            model.reinforce(reward)
            means = model.fixed_release_reference[model.pre].astype(float)
            actual = np.bincount(model.post, weights=model.weights.astype(float) * means,
                                 minlength=model.n)
            reference = np.bincount(model.post, weights=model.base.astype(float) * means,
                                    minlength=model.n)
            np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-10)
        for key, value in p.arrays().items():
            np.testing.assert_array_equal(value, q.arrays()[key])


def test_checkpoint_cannot_replace_or_drop_the_frozen_reference():
    model = anchored([.2, .3, .4, .5])
    arrays = model.arrays()
    arrays["weights"] *= 1.1
    before = model.weights.copy()
    with pytest.raises(ValueError, match="reference mismatch"):
        learner().restore(arrays, model.metrics())
    arrays["fixed_release_reference"][0] = .3
    with pytest.raises(ValueError, match="reference mismatch"):
        model.restore(arrays, model.metrics())
    del arrays["fixed_release_reference"]
    with pytest.raises(ValueError, match="reference mismatch"):
        model.restore(arrays, model.metrics())
    np.testing.assert_array_equal(before, model.weights)


def test_internal_rule_preserves_physical_observation_and_frozen_weights():
    candidate, original = learner(), learner("sensorimotor-perturb-projected-v4")
    for p in (candidate, original):
        p.observe(np.array([0, 3]), .02, perturbation=np.array([0, 1, 0, 1]), probability=.024)
    for key, value in candidate.arrays().items():
        np.testing.assert_array_equal(value, original.arrays()[key])
    candidate.weights[:] *= np.array([.5, 1.8, 1.5, .7])
    before = candidate.weights.copy()
    candidate.reinforce(1, enabled=False)
    np.testing.assert_array_equal(before, candidate.weights)


def test_internal_homeostasis_retains_learning_and_resumes_exactly():
    p, q = learner(), learner()
    p.post_baseline[:] = [.02, .08, .04, .03]
    p.eligibility[:] = [-30, 8, 20, -20]
    p.feedback_elapsed[...] = .24
    p.reinforce(.5)
    assert np.any(p.weights != p.base)
    assert p.last_changed > 0
    q.restore({k: v.copy() for k, v in p.arrays().items()}, p.metrics())
    for reward in (0., .05, 1., -.2):
        for model in (p, q):
            model.observe(
                np.array([0, 2]), .02, perturbation=np.array([0, 0, 1, 0]), probability=.024,
            )
            model.reinforce(reward)
            mean = model.post_baseline[model.pre].astype(float)
            actual = np.bincount(
                model.post, weights=model.weights.astype(float) * mean,
                minlength=model.n,
            )
            reference = np.bincount(model.post, weights=model.base.astype(float) * mean,
                                    minlength=model.n)
            np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-10)
        for key, value in p.arrays().items():
            np.testing.assert_array_equal(value, q.arrays()[key])


@pytest.mark.parametrize("normalize,budget", [(False, 0), (True, 0), (False, .25)])
@pytest.mark.parametrize("signed", [False, True])
def test_legacy_constraint_refactor_preserves_exact_operation_order(normalize, budget, signed):
    rng = np.random.default_rng(275)
    base = rng.uniform(.001, .3, 1000).astype(np.float32)
    if signed:
        base[::2] *= -1
    model = SensorimotorPlasticity(
        rng.integers(100, size=1000), rng.integers(100, size=1000), base, 100,
        PlasticityConfig(rule="sensorimotor-score-v3" if signed else "sensorimotor-perturb-v3",
                         normalize_inputs=normalize, input_budget_fraction=budget),
    )
    proposed = rng.normal(1, 3, 1000).astype(np.float32)
    expected = np.clip(proposed.copy(), .25, 4.)
    if normalize or budget:
        for _ in range(12):
            total = np.bincount(model.resource_group, weights=model.resource_base * expected,
                                minlength=model.resource_size)
            target = np.clip(total[model.resource_group],
                             model.input_budget[model.resource_group] * (1 - budget),
                             model.input_budget[model.resource_group] * (1 + budget))
            ratio = total[model.resource_group] / target
            if np.max(np.abs(ratio - 1)) < 1e-5:
                break
            expected /= ratio
            np.clip(expected, .25, 4., out=expected)
    np.testing.assert_array_equal(model.constrain_factors(proposed), expected)
