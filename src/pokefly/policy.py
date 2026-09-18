"""Online linear SARSA(lambda) readout. Only these weights learn."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokefly.emulator import ACTIONS


class LinearReadout:
    def __init__(
        self,
        features: int,
        *,
        seed: int = 64,
        epsilon: float = 0.2,
        learning_rate: float = 0.03,
        signature: str = "",
    ) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between zero and one")
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 0.01, (len(ACTIONS), features)).astype(np.float32)
        self.eligibility = np.zeros_like(self.weights)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = 0.98
        self.trace_decay = 0.8
        self.updates = 0
        self.signature = signature

    def choose(self, features: np.ndarray) -> tuple[int, bool]:
        exploratory = self.rng.random() < self.epsilon
        if exploratory:
            return int(self.rng.integers(len(ACTIONS))), True
        values = self.weights @ features
        ties = np.flatnonzero(values == values.max())
        return int(self.rng.choice(ties)), False

    def learn(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        next_action: int,
        *,
        terminal: bool = False,
    ) -> float:
        target = reward
        if not terminal:
            target += self.discount * float(self.weights[next_action] @ next_features)
        error = target - float(self.weights[action] @ features)
        self.eligibility *= self.discount * self.trace_decay
        self.eligibility[action] += features
        self.weights += self.learning_rate * np.clip(error, -5, 5) * self.eligibility
        self.updates += 1
        if terminal:
            self.eligibility.fill(0)
        return float(error)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            weights=self.weights,
            actions=np.asarray(ACTIONS),
            updates=self.updates,
            signature=self.signature,
            schema=1,
        )

    def load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            if (
                int(checkpoint["schema"]) != 1
                or tuple(checkpoint["actions"]) != ACTIONS
                or str(checkpoint["signature"]) != self.signature
            ):
                raise ValueError("Checkpoint was created with a different encoder or action space")
            weights = checkpoint["weights"]
            if weights.shape != self.weights.shape or not np.isfinite(weights).all():
                raise ValueError("Checkpoint weights are incompatible or non-finite")
            self.weights[:] = weights
            self.updates = int(checkpoint["updates"])
        self.eligibility.fill(0)
