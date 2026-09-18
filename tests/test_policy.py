from pathlib import Path

import numpy as np
import pytest

from pokefly.policy import LinearReadout


def test_terminal_reward_updates_only_chosen_action_without_bootstrap() -> None:
    policy = LinearReadout(3)
    policy.weights.fill(0)
    policy.weights[4].fill(100)  # Terminal update must ignore this next-state value.
    x = np.array([0.0, 1.0, 0.0], np.float32)
    before = policy.weights.copy()
    error = policy.learn(x, 2, 1.0, x, 4, terminal=True)
    assert error == pytest.approx(1)
    assert policy.weights[2, 1] > before[2, 1]
    np.testing.assert_array_equal(policy.weights[4], before[4])
    assert not policy.eligibility.any()


def test_checkpoint_roundtrip_and_compatibility(tmp_path: Path) -> None:
    policy = LinearReadout(3, epsilon=0, signature="test-v1")
    x = np.array([0.2, 0.3, 1.0], np.float32)
    policy.learn(x, 2, 1, x, 3)
    path = tmp_path / "readout.npz"
    policy.save(path)
    restored = LinearReadout(3, epsilon=0, seed=17, signature="test-v1")
    restored.load(path)
    np.testing.assert_array_equal(restored.weights, policy.weights)
    assert restored.choose(x) == policy.choose(x)
    assert restored.updates == 1
    with pytest.raises(ValueError, match="different encoder"):
        LinearReadout(3, signature="changed").load(path)
