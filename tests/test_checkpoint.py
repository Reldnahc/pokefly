import json
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import StopAtBoundary
from pokefly.internal_brain import CheckpointNoise, InternalBrain
from pokefly.plasticity import EligibilityPlasticity


def fixture(tmp_path):
    np.savez_compressed(tmp_path / "brain.npz", weights=np.array([0.2], np.float32))
    (tmp_path / "game.state").write_bytes(b"test fixture, not a real game state")
    (tmp_path / "state.json").write_text(json.dumps({"format": 1}), encoding="utf-8")
    manifest = {name: sha256(tmp_path / name) for name in ("brain.npz", "game.state", "state.json")}
    (tmp_path / "complete.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_complete_checkpoint_and_pointer(tmp_path):
    directory = fixture(tmp_path)
    pointer = tmp_path / "latest-checkpoint.json"
    pointer.write_text(json.dumps({"path": str(directory)}), encoding="utf-8")
    arrays, state = read_checkpoint(pointer)
    np.testing.assert_array_equal(arrays["weights"], np.array([0.2], np.float32))
    assert state["directory"] == str(directory.resolve())


def test_changed_or_incomplete_checkpoint_fails_closed(tmp_path):
    fixture(tmp_path)
    (tmp_path / "game.state").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="modified"):
        read_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_checkpoint(tmp_path / "missing")


def test_manifest_cannot_select_arbitrary_paths(tmp_path):
    (tmp_path / "complete.json").write_text('{"../private":"ignored"}', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        read_checkpoint(tmp_path)


def test_ctrl_c_is_deferred_and_prior_handler_restored():
    previous = signal.getsignal(signal.SIGINT)
    with StopAtBoundary() as stop:
        assert not stop.requested
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert stop.requested  # No exception interrupts the neural/emulator transaction.
    assert signal.getsignal(signal.SIGINT) == previous


def test_neural_snapshot_owns_plastic_arrays_for_isolated_control_trials():
    c = InternalBrain.__new__(InternalBrain)
    c.plasticity = EligibilityPlasticity(np.array([0]), np.array([1]), np.array([0.2]), 2)
    c.brain = SimpleNamespace(
        v=np.zeros((2, 1), np.float32), fired=np.array([0]), steps=1, rng=CheckpointNoise(np, 64)
    )
    c.hybrid = None
    c.identity = lambda: {}
    c.decoder = SimpleNamespace(state=lambda: {})
    arrays, _ = c.snapshot()
    for name, value in c.plasticity.arrays().items():
        assert not np.shares_memory(arrays[name], value), name
    c.plasticity.weights *= 2
    np.testing.assert_array_equal(arrays["weights"], arrays["base"])
