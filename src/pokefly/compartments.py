"""Conservative, PARTIAL type-level mushroom-body compartment assignments.

Li et al. 2020, eLife 62576 (Figures 22, 26, 30): PAM01/gamma5 ->
MBON01/MBON27; PAM11/alpha1 -> MBON07; PPL101/gamma1pedc -> MBON11.
https://elifesciences.org/articles/62576

These standardized types exist in our MaleCNS metadata. Their fine spatial
synapse compartments and DAN subtypes are NOT present in the installed data.
Only existing shared DAN targets and appropriate visual-KC types are admitted.
This remains a cross-dataset, type-level approximation, not resolved anatomy.
No MBON is assigned a button, behavioral valence, or new fast connection here.
"""

from __future__ import annotations

import numpy as np

COMPARTMENTS = (
    ("gamma5", "PAM01", ("MBON01", "MBON27"), "KCg-d", 1),
    ("alpha1", "PAM11", ("MBON07",), "KCab-p", 1),
    ("gamma1pedc", "PPL101", ("MBON11",), "KCg-d", -1),
)


def compartment_gates(brain, pre, post):
    types = brain.cell_type.astype(str)
    gates = np.zeros((len(COMPARTMENTS), len(pre)), np.float32)
    for k, (name, dan_type, mbon_types, kc_type, _) in enumerate(COMPARTMENTS):
        dans = np.flatnonzero(types == dan_type)
        eligible = np.isin(types[post], mbon_types) & (types[pre] == kc_type)
        if not len(dans) or not eligible.any():
            raise ValueError(f"Required compartment types missing: {name}")
        for dan in dans:
            edges = slice(brain.indptr[dan], brain.indptr[dan + 1])
            targets = np.zeros(brain.n, np.float32)
            targets[brain.indices[edges]] = np.abs(brain.weights[edges])
            gates[k] += np.sqrt(targets[pre] * targets[post]) * eligible
        maximum = gates[k].max(initial=0)
        if maximum == 0:
            raise ValueError(f"No shared existing DAN targets for {name}")
        gates[k] /= maximum
    return gates
