import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_probe_excludes_warmup_and_never_reinforces(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("probe_game_motor_drift")

    class FakeBrain:
        brain = SimpleNamespace(n=2, dt=0.02)
        brain_steps = 12
        plasticity = SimpleNamespace(weights=np.ones(1))
        motors = {"up": np.array([0]), "b": np.array([1])}

        def reset_dynamics(self, seed):
            self.seed, self.observed = seed, 0

        def observe(self, frame):
            self.observed += 1
            return SimpleNamespace(counts=np.array([self.observed, 2 * self.observed]))

        def choose(self, observation):
            return "up+b", {}

    c = FakeBrain()  # No reinforce method: calling one would fail the test.
    row = module.probe_rates(c, None, 55, warmup=2, decisions=2)
    assert c.seed == 55 and c.observed == 4
    assert row["motor_counts"] == {"up": [7], "b": [14]}
    assert row["motor_hz"] == {"up": 7 / 0.48, "b": 14 / 0.48}
    assert {k: v for k, v in row["button_counts"].items() if v} == {"up": 2, "b": 2}
    assert row["weights_unchanged"]
