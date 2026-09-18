import numpy as np
import pytest

from pokefly.cuda_plasticity import CUDAEligibility
from pokefly.fast_plasticity import sensorimotor_eligibility
from pokefly.runtime import configure_runtime


@pytest.fixture
def gpu():
    configure_runtime()
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:
        pytest.skip("CUDA eligibility check requires a CUDA device")
    return cp


@pytest.mark.parametrize("clipped", [False, True])
@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("floor", [-5.0, 0.0])
def test_cuda_eligibility_matches_every_serial_bit_and_keeps_no_dynamic_state(
    gpu, clipped, dual, floor
):
    rng = np.random.default_rng(2121)
    n, edges = 37, 1101
    pre, post = [rng.integers(0, n, edges) for _ in range(2)]
    a, b = [rng.normal(0, 4, edges).astype(np.float32) for _ in range(2)]
    expected, actual = a.copy(), a.copy()
    slow_expected, slow_actual = b.copy(), b.copy()
    cuda = CUDAEligibility(pre, post, gpu)
    for step in range(25):
        trace = rng.normal(0, 0.2, n).astype(np.float32)
        baseline = rng.uniform(0, 0.05, n).astype(np.float32)
        spike = (rng.random(n) < 0.05).astype(np.float32)
        options = dict(clip_eligibility=clipped, incoming_floor=floor)
        if dual:
            options.update(slow_decay=np.exp(-0.02 / 30))
        if step == 12:
            # Simulate replacing arrays during a checkpoint restore.
            actual, slow_actual = actual.copy(), slow_actual.copy()
        for kernel, values, slow in (
            (sensorimotor_eligibility, expected, slow_expected),
            (cuda, actual, slow_actual),
        ):
            kernel(
                pre,
                post,
                trace,
                baseline,
                spike,
                values,
                0.02,
                np.exp(-0.02 / 0.6),
                **options,
                **({"slow_eligibility": slow} if dual else {}),
            )
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(slow_expected, slow_actual)


def test_cuda_eligibility_rejects_changed_edges_and_misaligned_arrays(gpu):
    pre, post = np.array([0]), np.array([1])
    cuda = CUDAEligibility(pre, post, gpu)
    vector, eligible = np.zeros(2, np.float32), np.zeros(1, np.float32)
    with pytest.raises(ValueError, match="identity"):
        cuda(pre.copy(), post, vector, vector, vector, eligible, 0.02, 0.8)
    with pytest.raises(ValueError, match="sizes"):
        cuda(pre, post, vector[:1], vector, vector, eligible, 0.02, 0.8)
    with pytest.raises(ValueError, match="slow"):
        cuda(pre, post, vector, vector, vector, eligible, 0.02, 0.8, slow_eligibility=vector)
