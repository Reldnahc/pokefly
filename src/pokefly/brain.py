"""Screen -> explicit visual injections -> frozen connectome -> neural features.

The hand-written encoder is an experimental interface, not a model of fly vision.
It does not read RAM, choose buttons, or contain a Pokemon route.
"""

from __future__ import annotations

import hashlib

import numpy as np

from pokefly.runtime import configure_runtime

ENCODER_VERSION = "visual-halves-v1"


def image_features(frame: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    gray = frame[:, :, :3].astype(np.float32).mean(axis=2) / 255.0
    motion = np.zeros_like(gray) if previous is None else np.abs(gray - previous)
    h, w = gray.shape
    out = []
    for left, right in ((0, w // 2), (w // 2, w)):
        half = gray[:, left:right]
        out.extend(
            (
                1.0 - half[: h // 2].mean(),  # upper darkness -> LC10a
                1.0 - half[h // 2 :].mean(),  # lower darkness -> LPLC1
                min(1.0, 4.0 * motion[:, left:right].mean()),  # motion -> LC4
                min(1.0, 4.0 * np.abs(np.diff(half, axis=1)).mean()),  # contrast -> LPLC2
            )
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class FlyController:
    def __init__(self, *, device: str = "auto", seed: int = 64, brain_steps: int = 12) -> None:
        data = configure_runtime()
        from flybrain import FlyBrain, Trace, has_data

        if not has_data(data):
            raise RuntimeError(f"Brain data missing: run 'pokefly download' (destination: {data})")
        if brain_steps < 1:
            raise ValueError("brain_steps must be positive")
        self.brain = FlyBrain(data=data, device=device, seed=seed)
        self.steps_per_action = brain_steps
        self.trace = Trace(self.brain, types=["descending_neuron"], tau=0.2)
        self.groups = [
            self.brain.cells([cell], side=side)
            for side in ("L", "R")
            for cell in ("LC10a", "LPLC1", "LC4", "LPLC2")
        ]
        if not len(self.trace.idx) or any(len(group) == 0 for group in self.groups):
            raise RuntimeError("Downloaded brain is missing the required neuron populations")
        self.previous: np.ndarray | None = None
        self.last_spikes = 0
        self.last_drives = np.zeros(8, np.float32)
        self.last_dn_spikes = 0
        self.feature_count = len(self.trace.idx) + 1  # Constant bias, no game state.
        self.signature = (
            f"{ENCODER_VERSION}:{brain_steps}:"
            + hashlib.sha256(self.trace.idx.astype("<i8").tobytes()).hexdigest()
        )

    def observe(self, frame: np.ndarray) -> np.ndarray:
        self.last_drives = image_features(frame, self.previous)
        self.previous = frame[:, :, :3].astype(np.float32).mean(axis=2) / 255.0
        injection = [
            (group, float(drive) * 0.8)
            for group, drive in zip(self.groups, self.last_drives, strict=True)
        ]
        self.last_spikes = 0
        self.last_dn_spikes = 0
        for _ in range(self.steps_per_action):
            fired = self.brain.step(inject=injection)
            self.trace.observe(fired)
            self.last_spikes += len(fired)
            self.last_dn_spikes += int(np.count_nonzero(self.trace.slot[fired] >= 0))
        # Scale the decaying spike trace to [0,1], then bound the vector norm.
        values = self.trace.features() * (1.0 - self.trace.decay)
        features = np.append(values, np.float32(1.0)).astype(np.float32)
        features /= max(1.0, float(np.linalg.norm(features)))
        return features
