from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import pokefly.calibrated_vision as module
from pokefly.calibrated_vision import CalibratedVisualCircuit


@pytest.fixture
def circuit(tmp_path, monkeypatch):
    original = sparse.csr_matrix(np.array([
        [0, 0, 0, 0, 0],
        [-0.2, 0, -0.1, 0.4, 0],
        [0, 0.3, 0, 0, 0],
        [0, 0, 0.1, 0, 0],
        [0, 0, 0, 0, 0],
    ], np.float32))
    sparse.save_npz(tmp_path / "weights.npz", original)
    np.savez(
        tmp_path / "brain.npz", ids=np.arange(11, 16),
        cell_type=np.array(["R1-6", "L1", "Mi1", "other", "other"]),
        superclass=np.array(["sensory", "ol_intrinsic", "ol_intrinsic", "cb_intrinsic", "other"]),
        visual=np.array([0]),
    )
    np.savez(tmp_path / "retina.npz", unused=np.array([1]))
    params = {
        "types": {"L1": {"tau": 0.02, "bias": 0.5}, "Mi1": {"tau": 0.04, "bias": 0.3}},
        "pairs": [{"pre": "L1", "post": "Mi1", "strength": 0.25},
                  {"pre": "Mi1", "post": "L1", "strength": 0.5}],
        "photoreceptor": {"tau": 0.02, "restOffset": 0.2, "stimGain": 0.5,
                          "laminaInput": {"L1": -1.0}},
    }
    ref = SimpleNamespace(
        parameters=params, original_parameters=params,
        match_body_ids=lambda ids: np.array([1, 2]),
        canonical_types=np.array(["L1", "Mi1"]), sides=np.array(["L", "L"]),
        roles=np.array(["visual", "visual"]),
        arrays={"edges.pre": np.array([0, 1]), "edges.post": np.array([1, 0]),
                "edges.weight": np.array([2, 3])},
        manifest={"revision": "test", "assets": {}},
    )
    monkeypatch.setattr(module, "VisualReference", lambda path: ref)
    monkeypatch.setattr(
        module.RetinaMap, "load", lambda *args: SimpleNamespace(
            encode=lambda frame: np.array([frame.mean() / 255], np.float32)
        ),
    )
    return CalibratedVisualCircuit(device="cpu", data=tmp_path), original


def test_calibration_only_uses_real_cells_edges_and_original_signs(circuit):
    c, original = circuit
    np.testing.assert_array_equal(c.body_ids, [11, 12, 13])
    np.testing.assert_array_equal(np.sign(c.matrix.toarray()), np.sign(original[:3, :3].toarray()))
    assert c.matrix[1, 0] == -1.0  # Real receptor -> lamina, no pixel bypass.
    assert c.matrix[2, 1] == 0.5
    assert c.matrix[1, 2] == -1.5
    assert c.info["existing_internal_edges"] == 3
    np.testing.assert_array_equal(c.photo_indices, [0])
    release = np.array([0.8, 0.6, 0.7, 1, 0], np.float32)
    np.testing.assert_allclose(c.incoming_current(release, 3), [0, 1.2, 0])
    assert c.boundary.nnz == 1  # No double-count of visual-internal edges.


def test_rate_model_restore_is_exact_and_light_enters_photoreceptors(circuit):
    c, _ = circuit
    dark, bright = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 255, np.uint8)
    c.advance(dark, seconds=0.004)
    dark_rate = c.host_rate()
    c.reset()
    c.advance(bright, seconds=0.004)
    bright_rate = c.host_rate()
    assert bright_rate[0] > dark_rate[0]
    np.testing.assert_array_equal(bright_rate[1:], dark_rate[1:])
    for _ in range(20):
        c.advance(bright)
    saved = c.voltage.copy()
    expected = c.advance(dark).copy()
    c.restore_voltage(saved)
    np.testing.assert_array_equal(c.advance(dark), expected)
    assert np.isfinite(c.voltage).all()
    assert np.all((expected >= 0) & (expected <= 5))
    assert c.voltage[1] < 0  # Negative internal voltages are legitimate state.
    with pytest.raises(ValueError, match="multiple"):
        c.advance(bright, seconds=0.003)
    with pytest.raises(ValueError, match="voltage"):
        c.restore_voltage(np.full(c.n, np.nan))
    with pytest.raises(ValueError, match="photoreceptor"):
        c.advance_drive(np.zeros(3))
