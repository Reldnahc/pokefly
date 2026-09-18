import numpy as np
import pytest

from pokefly.fast_plasticity import sensorimotor_eligibility


@pytest.mark.parametrize("scale", [0.02, 0.1, 0.031234567])
@pytest.mark.parametrize("decay", [np.exp(-0.02 / 0.6), np.exp(-0.02 / 0.15)])
def test_fused_eligibility_matches_numpy_bit_for_bit(scale, decay):
    rng = np.random.default_rng(927)
    pre = rng.integers(1000, size=20000)
    post = rng.integers(1000, size=20000)
    reference = rng.uniform(-5, 5, len(pre)).astype(np.float32)
    fused = reference.copy()
    for _ in range(30):
        trace = rng.random(1000, dtype=np.float32) * 0.2
        baseline = rng.random(1000, dtype=np.float32) * 0.1
        spike = (rng.random(1000) < 0.025).astype(np.float32)
        incoming = np.clip(trace[pre] / scale, 0, 5)
        outgoing = (spike[post] - baseline[post]) / scale
        reference *= decay
        reference += (1 - decay) * incoming * outgoing
        np.clip(reference, -5, 5, out=reference)
        sensorimotor_eligibility(pre, post, trace, baseline, spike, fused, scale, decay)
        np.testing.assert_array_equal(fused, reference)
