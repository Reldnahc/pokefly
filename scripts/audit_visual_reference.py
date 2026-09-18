"""Read-only comparison of calibrated visual reference and the ORIGINAL fly.

Checks anatomical identities and existing-edge/sign agreement. No weights,
neurons, decoder, retina, game or original files are changed. This precedes
any attempt to import neural parameters into an opt-in experimental model.
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from pokefly.checkpoint import sha256
from pokefly.plastic_edges import csr_offsets_for_edges
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime
from pokefly.visual_reference import VisualReference


def main():
    data = configure_runtime()
    source = Path("fly-data/visual-reference-c28066a5")
    reference = VisualReference(source)
    with np.load(data / "brain.npz", allow_pickle=False) as z:
        body_ids, types, classes = z["ids"], z["cell_type"].astype(str), z["superclass"].astype(str)
    matched = reference.match_body_ids(body_ids)
    graph = sparse.load_npz(data / "weights.npz").tocsr()
    pre = matched[reference.arrays["edges.pre"]]
    post = matched[reference.arrays["edges.post"]]
    offsets = csr_offsets_for_edges(pre, post, graph.indptr, graph.indices)
    original = graph.data[offsets]
    counts = reference.arrays["edges.weight"]
    assert (counts > 0).all() and (original != 0).all()
    ratio = np.abs(original).astype(float) / counts
    minimum, maximum = np.full(len(body_ids), np.inf), np.zeros(len(body_ids))
    np.minimum.at(minimum, post, ratio)
    np.maximum.at(maximum, post, ratio)
    active = maximum > 0
    normalization_spread = (maximum[active] - minimum[active]) / maximum[active]
    pair_sign = {(p["pre"], p["post"]): p["sign"] for p in reference.parameters["pairs"]}
    ref_pre, ref_post = reference.arrays["edges.pre"], reference.arrays["edges.post"]
    tp, tq = reference.canonical_types[ref_pre], reference.canonical_types[ref_post]
    types_in_pairs = sorted(set(tp) | set(tq))
    ti = {kind: i for i, kind in enumerate(types_in_pairs)}
    sign_table = np.zeros((len(ti), len(ti)), np.int8)
    for (a, b), sign in pair_sign.items():
        if a in ti and b in ti:
            sign_table[ti[a], ti[b]] = sign
    type_index = np.array([ti[t] for t in reference.canonical_types])
    fitted_sign = sign_table[type_index[ref_pre], type_index[ref_post]]
    covered = fitted_sign != 0
    mismatched = covered & (np.sign(original) != fitted_sign)
    disagreements = {}
    for a, b in set(zip(tp[mismatched], tq[mismatched], strict=True)):
        mask = mismatched & (tp == a) & (tq == b)
        disagreements[f"{a}->{b}"] = int(mask.sum())
    report = {
        "scope": __doc__, "reference": str(source), "revision": reference.manifest["revision"],
        "neurons": reference.n, "existing_edges_verified": len(offsets),
        "all_reference_neurons_and_edges_present": True,
        "exact_type_matches": int(np.count_nonzero(types[matched] == reference.types)),
        "original_neuron_classes": {
            str(name): int(np.count_nonzero(classes[matched] == name))
            for name in np.unique(classes[matched])
        },
        "count_to_original_weight_relative_spread_max": float(normalization_spread.max()),
        "count_to_original_weight_relative_spread_p99": float(
            np.quantile(normalization_spread, .99)
        ),
        "fitted_pair_edges": int(covered.sum()), "sign_disagreement_edges": int(mismatched.sum()),
        "sign_disagreement_pairs": disagreements,
        "protected_original_hashes": {
            name: sha256(data / name) for name in ("brain.npz", "weights.npz", "retina.npz")
        },
        "next_constraint": "Use only existing anatomical neurons/edges; preserve original signs. "
        "Never import reference motor readout or direct virtual-photoreceptor lamina injection.",
    }
    output = run_directory("visual-reference-audit")
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
