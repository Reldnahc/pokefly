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


@pytest.mark.parametrize("clip", [True, False])
def test_dual_trace_kernel_exactly_matches_two_independent_traces(clip):
    rng = np.random.default_rng(555)
    pre, post = rng.integers(100, size=(2, 1000))
    fast, slow, joint_fast, joint_slow = [np.zeros(1000, np.float32) for _ in range(4)]
    fast_decay, slow_decay = np.exp(-0.02 / 0.6), np.exp(-0.02 / 30)
    for _ in range(30):
        trace = rng.random(100, dtype=np.float32) * 0.2
        baseline = np.full(100, 0.024, np.float32)
        spike = (rng.random(100) < 0.024).astype(np.float32)
        args = (pre, post, trace, baseline, spike)
        sensorimotor_eligibility(*args, fast, 0.02, fast_decay, clip)
        sensorimotor_eligibility(*args, slow, 0.02, slow_decay, clip)
        sensorimotor_eligibility(
            *args,
            joint_fast,
            0.02,
            fast_decay,
            clip,
            slow_eligibility=joint_slow,
            slow_decay=slow_decay,
        )
        np.testing.assert_array_equal(fast, joint_fast)
        np.testing.assert_array_equal(slow, joint_slow)
