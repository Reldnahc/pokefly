from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from pokefly.plastic_edges import csr_offsets_for_edges, sensorimotor_targets
from pokefly.plasticity import PlasticityConfig


def test_premotor_scope_selects_anatomy_not_buttons_or_edge_signs():
    classes = np.array(
        [
            "descending_neuron",
            "cb_motor",
            "vnc_motor",
            "cb_intrinsic",
            "cb_intrinsic",
            "visual_projection",
            "sensory",
            "cb_intrinsic",
        ]
    )
    # Cell 3 projects to a DN; 4 is two hops away and must NOT be added.
    # Sensory/visual parents are excluded. Cell 7 only projects to a motor.
    graph = sparse.csc_matrix(([1, -1, 1, 1, 1], ([0, 3, 0, 0, 1], [3, 4, 5, 6, 7])), shape=(8, 8))
    brain = SimpleNamespace(
        superclass=classes, indices=graph.indices, indptr=graph.indptr, weights=graph.data
    )
    np.testing.assert_array_equal(
        np.flatnonzero(sensorimotor_targets(brain, "motor-inputs-v1")), [0, 1, 2]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(sensorimotor_targets(brain, "premotor-one-hop-v2")), [0, 1, 2, 3]
    )
    graph.data[graph.indptr[3] : graph.indptr[4]] *= -1
    np.testing.assert_array_equal(
        np.flatnonzero(sensorimotor_targets(brain, "premotor-one-hop-v2")), [0, 1, 2, 3]
    )
    with pytest.raises(ValueError, match="scope"):
        sensorimotor_targets(brain, "buttons")


def test_sparse_correspondence_preserves_unordered_edges_exactly():
    rng = np.random.default_rng(1234)
    graph = sparse.random(40, 40, density=0.3, random_state=rng, format="csc")
    outgoing = np.arange(graph.nnz)[::3][::-1]
    pre = np.searchsorted(graph.indptr, outgoing, side="right") - 1
    post = graph.indices[outgoing]
    incoming = graph.tocsr()
    indices = csr_offsets_for_edges(pre, post, incoming.indptr, incoming.indices)
    np.testing.assert_array_equal(incoming.data[indices], graph.data[outgoing])
    assert csr_offsets_for_edges([], [], incoming.indptr, incoming.indices).size == 0


def test_missing_sparse_edges_and_invalid_scope_rejected():
    graph = sparse.eye(3, format="csr")
    with pytest.raises(ValueError, match="missing"):
        csr_offsets_for_edges([2], [0], graph.indptr, graph.indices)
    with pytest.raises(ValueError, match="align"):
        csr_offsets_for_edges([0, 1], [0], graph.indptr, graph.indices)
    with pytest.raises(ValueError, match="scope"):
        PlasticityConfig(scope="all-buttons")
    with pytest.raises(ValueError, match="sensorimotor"):
        PlasticityConfig(scope="premotor-one-hop-v2")
    PlasticityConfig(rule="sensorimotor-perturb-v3", scope="premotor-one-hop-v2")
