import numpy as np
import pytest

from pokefly.visual_reference import VisualReference, typed_array


def test_reference_array_reader_rejects_wrong_dtype_bounds_alignment():
    data = np.arange(4, dtype="<f4").tobytes()
    np.testing.assert_array_equal(
        typed_array(data, {"dtype": "float32", "length": 2, "offset": 4}), [1, 2]
    )
    for spec in (
        {"dtype": "object", "length": 2, "offset": 0},
        {"dtype": "float32", "length": 20, "offset": 0},
        {"dtype": "float32", "length": 2, "offset": 1},
        {"dtype": "float32", "length": -1, "offset": 0},
    ):
        with pytest.raises(ValueError):
            typed_array(data, spec)


def test_body_mapping_uses_anatomy_not_file_order_or_virtual_units():
    ref = VisualReference.__new__(VisualReference)
    ref.body_ids = np.array([22, 14])
    actual = np.array([14, 31, 22])
    np.testing.assert_array_equal(ref.match_body_ids(actual), [2, 0])
    with pytest.raises(ValueError, match="missing"):
        ref.match_body_ids(np.array([14, 31]))
