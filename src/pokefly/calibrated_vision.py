"""Opt-in visual-neuron rate-model prototype, NOT yet a gameplay controller.

Only original fly neurons and existing edges are used. Type parameters come
from the pinned licensed visual reference. Original edge signs are retained.
Actual photoreceptors receive the unchanged raw-pixel retina; their EXISTING
inputs to lamina neurons are calibrated to the reference's aggregate strength.
There is no virtual receptor, direct image-to-lamina bypass, motion detector,
action output, game state, or reward in this module.

The reference rate equation/parameter conventions derive from Abijah Kajabika's
MIT-licensed model and flyvis; see THIRD_PARTY_NOTICES.md. Transfer is a new
engineering hypothesis, not a reproduction of their graph's reported results.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from pokefly.checkpoint import sha256
from pokefly.deterministic import DeterministicCUDAInput
from pokefly.plastic_edges import csr_offsets_for_edges
from pokefly.runtime import configure_runtime
from pokefly.vision import RetinaMap
from pokefly.visual_reference import VisualReference


class CalibratedVisualCircuit:
    def __init__(self, *, device="cuda", parameters="fitted", data=None):
        data = Path(data or configure_runtime())
        reference = VisualReference(data / "visual-reference-c28066a5")
        if parameters not in ("fitted", "flyvis"):
            raise ValueError("Unknown visual parameter source")
        params = reference.parameters if parameters == "fitted" else reference.original_parameters
        with np.load(data / "brain.npz", allow_pickle=False) as z:
            ids, types = z["ids"], z["cell_type"].astype(str)
            classes, visual = z["superclass"].astype(str), z["visual"]
        original = sparse.load_npz(data / "weights.npz").tocsr()
        matched = reference.match_body_ids(ids)
        supported = np.isin(reference.canonical_types, list(params["types"])) & np.isin(
            classes[matched], ["ol_intrinsic", "visual_projection"]
        )
        lamina = np.flatnonzero(np.isin(types, ["L1", "L2", "L3", "L4", "L5"]))
        selected = np.unique(np.concatenate([matched[supported], visual, lamina]))
        local = np.full(len(ids), -1, np.int64)
        local[selected] = np.arange(len(selected))
        ref_local = local[matched]
        self.global_indices, self.body_ids = selected, ids[selected]
        self.n = len(selected)
        self.retina = RetinaMap.load(data / "retina.npz", data / "brain.npz")
        self.photo_indices = local[visual]
        self.types = types[selected].copy()
        self.types[ref_local[supported]] = reference.canonical_types[supported]
        self.sides = np.full(self.n, "", dtype="U1")
        self.sides[ref_local[supported]] = reference.sides[supported]
        self.photo_mask = np.zeros(self.n, bool)
        self.photo_mask[self.photo_indices] = True
        tau, bias = np.full(self.n, 0.03, np.float32), np.zeros(self.n, np.float32)
        for kind, values in params["types"].items():
            mask = self.types == kind
            tau[mask], bias[mask] = values["tau"], values["bias"]
        photo = params["photoreceptor"]
        tau[self.photo_indices] = photo["tau"]
        bias[self.photo_indices] = photo["restOffset"]
        if not np.isfinite(tau).all() or (tau <= 0).any() or not np.isfinite(bias).all():
            raise ValueError("Invalid calibrated neuronal constants")

        # Every original connection between selected cells remains present.
        # Uncovered connections retain their original normalized gain (3).
        matrix = (original[selected][:, selected] * np.float32(3)).tocsr()
        matrix.sort_indices()
        source_pre, source_post = reference.arrays["edges.pre"], reference.arrays["edges.post"]
        keep = (ref_local[source_pre] >= 0) & (ref_local[source_post] >= 0)
        pre, post = ref_local[source_pre[keep]], ref_local[source_post[keep]]
        offsets = csr_offsets_for_edges(pre, post, matrix.indptr, matrix.indices)
        signs = np.sign(matrix.data[offsets])
        kinds = sorted(set(reference.canonical_types))
        type_number = {kind: i for i, kind in enumerate(kinds)}
        table = np.full((len(kinds), len(kinds)), np.nan, np.float32)
        for pair in params["pairs"]:
            if pair["pre"] in type_number and pair["post"] in type_number:
                table[type_number[pair["pre"]], type_number[pair["post"]]] = pair["strength"]
        ti = np.array([type_number[t] for t in reference.canonical_types])
        strength = table[ti[source_pre[keep]], ti[source_post[keep]]]
        pooling = (reference.roles[source_post[keep]] == "input") | np.char.startswith(
            reference.canonical_types[source_post[keep]], "LPi"
        )
        strength = np.where(np.isfinite(strength), strength, np.where(pooling, 0.001, 0.02))
        if (strength < 0).any() or not np.isfinite(strength).all():
            raise ValueError("Calibrated strengths must preserve every existing sign")
        positive = strength > 0
        matrix.data[offsets[positive]] = (
            signs[positive] * reference.arrays["edges.weight"][keep][positive] * strength[positive]
        )
        # Calibrate aggregate conductance on REAL R1-6 -> L1/L2/L3 edges.
        # Preserve relative original synapse counts, positions, and all signs.
        r16 = self.types == "R1-6"
        photo_edges, photo_targets = 0, 0
        for kind, aggregate in photo["laminaInput"].items():
            for cell in np.flatnonzero(self.types == kind):
                indices = np.arange(matrix.indptr[cell], matrix.indptr[cell + 1])
                edges = indices[r16[matrix.indices[indices]]]
                if not len(edges):
                    continue
                if aggregate >= 0 or (matrix.data[edges] >= 0).any():
                    raise ValueError("Expected original histaminergic photoreceptor signs")
                matrix.data[edges] *= abs(aggregate) / np.abs(matrix.data[edges]).sum()
                photo_edges += len(edges)
                photo_targets += 1
        self.matrix = matrix
        expected = original[selected][:, selected].tocsr()
        expected.sort_indices()
        if not (
            np.array_equal(matrix.indptr, expected.indptr)
            and np.array_equal(matrix.indices, expected.indices)
            and np.array_equal(np.sign(matrix.data), np.sign(expected.data))
        ):
            raise ValueError("Calibration changed original connectivity or edge signs")
        # Preserve ALL actual incoming boundary edges for whole-brain coupling.
        boundary = original[selected].tocoo()
        outside = local[boundary.col] < 0
        self.boundary = sparse.csr_matrix(
            (boundary.data[outside], (selected[boundary.row[outside]], boundary.col[outside])),
            shape=original.shape,
        )
        self.dt, self.maximum_rate = 0.004, 5.0
        self.photo_gain = float(photo["stimGain"])
        self.device = device
        if device == "cuda":
            import cupy as cp
            import cupyx.scipy.sparse as csparse

            self.xp = cp
            wrapper = SimpleNamespace(n=self.n, batch=1, xp=cp, _W=csparse.csr_matrix(matrix))
            self.propagate = DeterministicCUDAInput(wrapper)
            boundary_wrapper = SimpleNamespace(
                n=len(ids), batch=1, xp=cp, _W=csparse.csr_matrix(self.boundary)
            )
            self.propagate_boundary = DeterministicCUDAInput(boundary_wrapper)
        elif device == "cpu":
            self.xp = np
            self.propagate = None
            self.propagate_boundary = None
        else:
            raise ValueError("Explicit CPU or CUDA device required")
        xp = self.xp
        self.tau, self.bias = xp.asarray(tau), xp.asarray(bias)
        self.alpha = xp.asarray(np.minimum(1, self.dt / tau).astype(np.float32))
        self.photo_device = xp.asarray(self.photo_indices)
        self.global_device = xp.asarray(self.global_indices)
        self.info = {
            "parameters": parameters, "reference_revision": reference.manifest["revision"],
            "neurons": self.n, "photoreceptors": len(self.photo_indices),
            "existing_internal_edges": int(matrix.nnz), "reference_internal_edges": int(keep.sum()),
            "photoreceptor_lamina_edges_calibrated": photo_edges,
            "lamina_targets_with_real_photoreceptor_input": photo_targets,
            "substep_seconds": self.dt, "maximum_rate": self.maximum_rate,
            "original_signs_retained": True, "original_files_modified": False,
            "reference_zero_strength_edges_kept_at_original_gain": int((~positive).sum()),
            "incoming_boundary_edges": int(self.boundary.nnz),
            "source_sha256": {
                name: sha256(data / name) for name in ("brain.npz", "weights.npz", "retina.npz")
            },
            "reference_asset_sha256": {
                name: item["sha256"] for name, item in reference.manifest["assets"].items()
            },
            "outside_circuit_boundary": "standalone assay: zero; full fly: all existing inputs",
            "limitations": "No gain/readout fit to game or buttons. Parameters were trained on "
            "a different subset/photoreceptor convention; transfer is not validated physiology. "
            "All selected-cell edges retained, unlike the smaller reference extraction.",
        }
        self.reset()

    def reset(self):
        self.voltage = self.bias.copy()
        self.rate = self.xp.clip(self.voltage, 0, self.maximum_rate)
        self.steps = 0

    def advance(self, frame, seconds=0.02, *, external_current=None):
        return self.advance_drive(
            self.retina.encode(frame), seconds, external_current=external_current
        )

    def incoming_current(self, release, gain):
        values = (
            self.propagate_boundary.dense(release)[:, 0]
            if self.propagate_boundary else self.boundary @ release
        )
        return values[self.global_device] * np.float32(gain)

    def advance_drive(self, eye_drive, seconds=0.02, *, external_current=None):
        substeps = round(seconds / self.dt)
        if substeps < 1 or not np.isclose(substeps * self.dt, seconds, atol=1e-9, rtol=0):
            raise ValueError("Visual interval must be an exact multiple of the calibrated substep")
        xp = self.xp
        drive = xp.zeros(self.n, xp.float32)
        eye_drive = xp.asarray(eye_drive, dtype=xp.float32)
        if eye_drive.shape != (len(self.photo_indices),):
            raise ValueError("One raw retinal value per original photoreceptor required")
        drive[self.photo_device] = eye_drive * np.float32(self.photo_gain)
        if external_current is not None:
            if external_current.shape != (self.n,):
                raise ValueError("External neural current shape mismatch")
            drive += xp.asarray(external_current, dtype=xp.float32)
        for _ in range(substeps):
            current = (
                self.propagate.dense(self.rate)[:, 0]
                if self.propagate else self.matrix @ self.rate
            )
            self.voltage += self.alpha * (current + self.bias + drive - self.voltage)
            self.rate = xp.clip(self.voltage, 0, self.maximum_rate)
            self.steps += 1
        return self.rate

    def host_rate(self):
        return self.rate.copy() if self.xp is np else self.rate.get()

    def restore_voltage(self, voltage):
        voltage = np.asarray(voltage, np.float32)
        if voltage.shape != (self.n,) or not np.isfinite(voltage).all():
            raise ValueError("Invalid calibrated visual voltage")
        self.voltage = self.xp.asarray(voltage.copy())
        self.rate = self.xp.clip(self.voltage, 0, self.maximum_rate)
