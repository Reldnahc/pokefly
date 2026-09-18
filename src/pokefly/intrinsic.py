"""Versioned frozen neural excitability artifacts, not an action calibration."""

import hashlib
import json

import numpy as np

WIDE_RULE = "uniform-neutral-rate-homeostasis-wide-v3"


def load_intrinsic(path, body_ids, allowed):
    allowed = np.asarray(allowed)
    if allowed.dtype != np.bool_ or allowed.shape != (len(body_ids),):
        raise ValueError("Intrinsic calibration needs an explicit anatomical eligibility mask")
    with np.load(path, allow_pickle=False) as calibration:
        if not np.array_equal(calibration["body_ids"], body_ids):
            raise ValueError("Intrinsic calibration neuron identity mismatch")
        protocol = json.loads(str(calibration["protocol"])) if "protocol" in calibration else {}
        wide = protocol.get("rule") == WIDE_RULE
        bounds = [-0.5, 0.5] if wide else [-0.14, 0.2]
        if protocol.get("bounds", bounds) != bounds:
            raise ValueError("Intrinsic bounds require their explicit calibration version")
        bias = np.asarray(calibration["bias"], np.float32)
        if (
            bias.shape != (len(body_ids), 1) or not np.isfinite(bias).all()
            or (bias < bounds[0] - 1e-6).any() or (bias > bounds[1] + 1e-6).any()
            or np.any(bias[~allowed])
        ):
            raise ValueError("Invalid intrinsic calibration offsets")
    return bias.copy(), {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        # Preserve exact historical checkpoint identity (including reset-v2).
        "rule": WIDE_RULE if wide else "uniform-neutral-rate-homeostasis-v1",
        "scope": "frozen internal excitability offsets; no game/reward/motor labels",
        **({"bounds": bounds} if wide else {}),
    }
