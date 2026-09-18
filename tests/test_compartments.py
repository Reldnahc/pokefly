from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from pokefly.compartments import compartment_gates
from pokefly.plasticity import CompartmentPlasticity, PlasticityConfig


def plasticity():
    return CompartmentPlasticity(
        np.array([0, 1, 0]),
        np.array([2, 2, 3]),
        np.array([0.2, 0.3, 0.4]),
        4,
        PlasticityConfig(rule="compartment-ema-v1"),
        gates=np.eye(3, dtype=np.float32),
    )


def test_partial_anatomy_excludes_unmapped_types_and_keeps_existing_edges():
    types = np.array(
        ["KCg-d", "KCab-p", "MBON01", "MBON07", "MBON11", "PAM01", "PAM11", "PPL101", "MBON02"]
    )
    matrix = sparse.csc_matrix(
        (np.ones(18), (np.tile([0, 1, 2, 3, 4, 8], 3), np.repeat([5, 6, 7], 6))), shape=(9, 9)
    )
    brain = SimpleNamespace(
        n=9, cell_type=types, weights=matrix.data, indices=matrix.indices, indptr=matrix.indptr
    )
    gates = compartment_gates(brain, np.array([0, 1, 0, 0]), np.array([2, 3, 4, 8]))
    np.testing.assert_array_equal(gates[:, :3], np.eye(3))
    assert not gates[:, 3].any()


def test_compartment_reward_sign_and_omission_are_not_the_same_update():
    p = plasticity()
    for _ in range(100):
        p.observe(np.array([0]), 0.02)
    p.reinforce(1)
    assert p.weights[0] < p.base[0]
    assert p.weights[1] == p.base[1] and p.weights[2] == p.base[2]
    assert p.reward_expectation[0] > 0 and p.prediction_error[0] > 0
    old = p.weights.copy()
    for _ in range(12):
        p.observe(np.array([0]), 0.02)
    p.reinforce(0)
    assert p.prediction_error[0] < 0 and p.weights[0] > old[0]
    p.reinforce(-1)
    assert p.weights[2] < p.base[2]


def test_frozen_absent_reward_and_ineligible_edges_stay_fixed():
    for frozen in (False, True):
        p = plasticity()
        for _ in range(100):
            p.observe(np.array([0, 1]), 0.02)
            p.reinforce(1 if frozen else 0, enabled=not frozen)
        np.testing.assert_array_equal(p.weights, p.base)


def test_compartment_state_restores_exactly_and_bounds_hold():
    a, b = plasticity(), plasticity()
    for _ in range(100):
        a.observe(np.array([0]), 0.02)
    a.reinforce(1)
    b.restore(a.arrays(), a.metrics())
    for reward in [0, 1, 0, -1] * 20:
        for p in (a, b):
            p.observe(np.array([0, 1]), 0.02)
            p.reinforce(reward)
        for key in a.arrays():
            np.testing.assert_array_equal(a.arrays()[key], b.arrays()[key])
    factors = a.weights / a.base
    assert (factors >= a.config.minimum_factor).all()
    assert (factors <= a.config.maximum_factor).all()
    assert (a.weights > 0).all()
    with pytest.raises(ValueError, match="identity"):
        b.restore(dict(a.arrays(), compartment_gates=np.zeros((3, 3))), a.metrics())
