import json

import numpy as np
import pytest

from pokefly.checkpoint import sha256
from pokefly.intrinsic import WIDE_RULE, load_intrinsic


def calibration(tmp_path, bias, protocol=None):
    path = tmp_path / "calibration.npz"
    np.savez(path, bias=np.array(bias, np.float32)[:, None], body_ids=np.array([10, 11]),
             **({"protocol": np.array(json.dumps(protocol))} if protocol else {}))
    return path


def test_legacy_calibration_identity_and_values_remain_unchanged(tmp_path):
    path = calibration(tmp_path, [0, -.12])
    values, metadata = load_intrinsic(path, np.array([10, 11]), np.array([False, True]))
    np.testing.assert_array_equal(values, np.array([0, -.12], np.float32)[:, None])
    assert metadata == {
        "sha256": sha256(path), "rule": "uniform-neutral-rate-homeostasis-v1",
        "scope": "frozen internal excitability offsets; no game/reward/motor labels",
    }


def test_wider_calibration_is_explicit_and_still_excludes_sensory_and_graded_cells(tmp_path):
    path = calibration(tmp_path, [0, -.4], {"rule": WIDE_RULE, "bounds": [-.5, .5]})
    values, metadata = load_intrinsic(path, np.array([10, 11]), np.array([False, True]))
    assert values[1, 0] == np.float32(-.4)
    assert metadata["rule"] == WIDE_RULE and metadata["bounds"] == [-.5, .5]
    with pytest.raises(ValueError, match="offsets"):
        load_intrinsic(path, np.array([10, 11]), np.array([True, False]))
    with pytest.raises(ValueError, match="identity"):
        load_intrinsic(path, np.array([11, 10]), np.array([True, True]))
    path = calibration(tmp_path, [0, -.4])
    with pytest.raises(ValueError, match="offsets"):
        load_intrinsic(path, np.array([10, 11]), np.array([False, True]))
    path = calibration(tmp_path, [0, -.4], {"rule": WIDE_RULE, "bounds": [-1, 1]})
    with pytest.raises(ValueError, match="version"):
        load_intrinsic(path, np.array([10, 11]), np.array([False, True]))


@pytest.mark.parametrize("value", [np.nan, np.inf, -.501, .501])
def test_wider_calibration_remains_finite_and_bounded(tmp_path, value):
    path = calibration(tmp_path, [0, value], {"rule": WIDE_RULE, "bounds": [-.5, .5]})
    with pytest.raises(ValueError, match="offsets"):
        load_intrinsic(path, np.array([10, 11]), np.array([False, True]))
