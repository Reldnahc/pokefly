"""Anatomical learning scopes and exact sparse-format index correspondence.

Selection is independent of the motor/button registry, pixels, actions and
rewards. The optional one-hop scope is a model hypothesis, not a measured map
of biologically plastic synapses.
"""

import numpy as np


def sensorimotor_targets(brain, scope):
    classes = brain.superclass.astype(str)
    selected = np.isin(classes, ["descending_neuron", "cb_motor", "vnc_motor"])
    if scope == "motor-inputs-v1":
        return selected
    if scope != "premotor-one-hop-v2":
        raise ValueError("Unknown plasticity scope")
    descending = classes == "descending_neuron"
    edges = np.flatnonzero(descending[brain.indices] & (brain.weights != 0))
    parents = np.unique(np.searchsorted(brain.indptr, edges, side="right") - 1)
    parents = parents[classes[parents] == "cb_intrinsic"]
    selected[parents] = True
    return selected


def csr_offsets_for_edges(pre, post, indptr, indices):
    """Map existing (pre, post) edges to sorted incoming-CSR storage.

    Group once instead of scanning millions of plastic edges for every target.
    Edge order and all arithmetic on neural weights remain unchanged.
    """
    pre, post = np.asarray(pre), np.asarray(post)
    if pre.ndim != 1 or pre.shape != post.shape:
        raise ValueError("Plastic edge arrays do not align")
    if len(post) and (post.min() < 0 or post.max() >= len(indptr) - 1):
        raise ValueError("Plastic target index out of range")
    result = np.empty(len(pre), np.int64)
    order = np.argsort(post, kind="stable")
    grouped_post = post[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(grouped_post)) + 1, len(pre)]
    for start, end in zip(cuts[:-1], cuts[1:], strict=True):
        if start == end:
            continue
        chosen = order[start:end]
        cell = post[chosen[0]]
        incoming = indices[indptr[cell] : indptr[cell + 1]]
        local = np.searchsorted(incoming, pre[chosen])
        if np.any(local >= len(incoming)) or not np.array_equal(incoming[local], pre[chosen]):
            raise ValueError("Plastic edge missing from incoming sparse graph")
        result[chosen] = indptr[cell] + local
    return result
