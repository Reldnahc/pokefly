"""Frozen generic-stimulus neural release means, never a learned policy layer."""

import hashlib
import json
from dataclasses import asdict

import numpy as np

from pokefly.checkpoint import sha256
from pokefly.runtime import configure_runtime

REFERENCE_RULE = "generic-visual-release-reference-v1"
REFERENCE_PLASTICITY = "sensorimotor-perturb-anchored-v6"
INTRINSIC_KEYS = ("bias", "mask", "body_ids", "protocol")


def source_data_hashes(data, config):
    names = ["brain.npz", "weights.npz", "retina.npz"]
    if config.dynamics.visual_model == "calibrated-rate-v1":
        names += [f"visual-reference-c28066a5/{name}" for name in (
            "manifest.json", "optic-v2.json", "optic-v2.bin",
            "fitted-params.json", "flyvis-params.json",
        )]
    return {name: sha256(data / name) for name in names}


def physical_config(config):
    values = asdict(config)
    # The collector calls only neural steps; neither decoder nor plasticity
    # participates. The intrinsic payload is checked separately, bit for bit.
    for name in ("plasticity", "motor", "intrinsic_calibration", "brain_steps"):
        values.pop(name)
    return values


def intrinsic_payload_sha256(arrays):
    digest = hashlib.sha256()
    for key in INTRINSIC_KEYS:
        value = np.asarray(arrays[key])
        digest.update(json.dumps([key, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def verify_intrinsic_copy(source, extended):
    """A new reference must not quietly alter the frozen physical calibration."""
    with np.load(source, allow_pickle=False) as original, np.load(
        extended, allow_pickle=False,
    ) as augmented:
        if (not set(INTRINSIC_KEYS) <= set(original.files)
                or not set((*INTRINSIC_KEYS, "mean_release", "reference_protocol"))
                <= set(augmented.files) or "reference_protocol" in original):
            raise ValueError("Reference extension requires an original unaugmented calibration")
        protocol = json.loads(str(augmented["reference_protocol"]))
        payload = intrinsic_payload_sha256(original)
        if (payload != intrinsic_payload_sha256(augmented)
                or protocol.get("intrinsic_payload_sha256") != payload
                or protocol.get("source_intrinsic_sha256") != sha256(source)
                or protocol.get("rule") != REFERENCE_RULE):
            raise ValueError(
                "Reference extension must preserve the exact original intrinsic payload"
            )
    return {"original_sha256": sha256(source), "extended_sha256": sha256(extended),
            "intrinsic_payload_sha256": payload}


def load_release_reference(path, body_ids, config, *, data=None):
    with np.load(path, allow_pickle=False) as artifact:
        if not set((*INTRINSIC_KEYS, "mean_release", "reference_protocol")) <= set(artifact.files):
            raise ValueError("Anchored plasticity requires a frozen generic release reference")
        if not np.array_equal(artifact["body_ids"], body_ids):
            raise ValueError("Release reference neuron identity mismatch")
        mean = np.asarray(artifact["mean_release"], np.float32)
        if (mean.shape != (len(body_ids),) or not np.isfinite(mean).all()
                or (mean < 0).any() or (mean > 1).any() or not (mean > 0).any()):
            raise ValueError("Invalid fixed neural release reference")
        protocol = json.loads(str(artifact["reference_protocol"]))
        if (protocol.get("rule") != REFERENCE_RULE
                or protocol.get("game_frames_used") is not False
                or protocol.get("rewards_used") is not False
                or protocol.get("actions_selected") != 0
                or protocol.get("physical_config") != physical_config(config)
                or protocol.get("intrinsic_payload_sha256") != intrinsic_payload_sha256(artifact)):
            raise ValueError("Release reference provenance or frozen physical model mismatch")
        source_hashes = protocol.get("data_sha256", {})
        data = data or configure_runtime()
        if source_hashes != source_data_hashes(data, config):
            raise ValueError("Release reference source asset identity mismatch")
        return mean.copy(), {
            "rule": REFERENCE_RULE, "sha256": sha256(path),
            "scope": "fixed generic-stimulus means; no gameplay context, rewards or button fitting",
        }
