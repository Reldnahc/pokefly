from types import SimpleNamespace

import numpy as np
import pytest

from pokefly.deterministic import KERNEL, ROW_KERNEL, DeterministicCUDAInput


class FakeCUDA:
    float32 = np.float32
    asarray = staticmethod(np.asarray)
    empty = staticmethod(np.empty)
    zeros = staticmethod(np.zeros)

    def RawKernel(self, source, name, options):
        self.compiled = source, name, options

        def run(grid, block, args):
            self.called = grid, block, args

        return run


def fake_brain():
    return SimpleNamespace(
        n=6, batch=1, xp=FakeCUDA(),
        _W=SimpleNamespace(
            indices=np.array([0, 1], np.int32),
            indptr=np.zeros(7, np.int32), data=np.ones(2, np.float32),
        ),
    )


@pytest.mark.parametrize("rows", [[], [2, 2], [3, 1], [-1, 1], [6], [[1]], [1.5]])
def test_selected_rows_reject_invalid_indices(rows):
    with pytest.raises(ValueError, match="Selected rows"):
        DeterministicCUDAInput(fake_brain(), rows=rows)


def test_default_propagation_signature_is_unchanged():
    b = fake_brain()
    operation = DeterministicCUDAInput(b)
    result = operation.dense(np.zeros(6, np.float32))
    assert b.xp.compiled == (KERNEL, "propagate", ("--fmad=false",))
    grid, block, args = b.xp.called
    assert grid == (2,) and block == (128,) and len(args) == 6 and args[-1] == 6
    assert result.shape == (6, 1)


def test_selected_rows_own_indices_and_zero_discarded_output():
    b = fake_brain()
    rows = np.array([1, 4], np.int64)
    operation = DeterministicCUDAInput(b, rows=rows)
    rows[0] = 0
    result = operation.dense(np.zeros(6, np.float32))
    assert b.xp.compiled == (ROW_KERNEL, "propagate_rows", ("--fmad=false",))
    grid, block, args = b.xp.called
    assert grid == (1,) and block == (128,) and len(args) == 7 and args[-1] == 2
    np.testing.assert_array_equal(args[-2], [1, 4])
    np.testing.assert_array_equal(result, np.zeros((6, 1), np.float32))
    # The physical accumulation and reduction body is textually unchanged.
    original_body = KERNEL[KERNEL.index("    float sum"):]
    assert original_body == ROW_KERNEL[ROW_KERNEL.index("    float sum"):]
