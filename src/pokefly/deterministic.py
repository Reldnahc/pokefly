"""Fixed-order CUDA sparse accumulation for repeatable neural checkpoints.

One warp per postsynaptic row, lane-strided ordered sums then an ordered warp
reduction. No cross-thread atomic additions. Connectivity and equations are
unchanged; float32 rounding order differs from upstream cuSPARSE's default.
"""

from __future__ import annotations

import numpy as np

KERNEL = r"""
extern "C" __global__ void propagate(
    const int* ptr, const int* col, const float* weight,
    const float* spikes, float* output, int n) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= n) return;
    float sum = 0.0f;
    for (int edge = ptr[row] + lane; edge < ptr[row + 1]; edge += 32) {
        if (spikes[col[edge]] != 0.0f) sum += weight[edge] * spikes[col[edge]];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) output[row] = sum;
}
"""


class DeterministicCUDAInput:
    def __init__(self, brain):
        self.brain = brain
        if brain.batch != 1 or brain._W.indices.dtype != np.int32:
            raise ValueError("Deterministic CUDA propagation requires single-fly int32 CSR")
        self.kernel = brain.xp.RawKernel(KERNEL, "propagate", options=("--fmad=false",))

    def __call__(self, fired):
        brain, xp = self.brain, self.brain.xp
        spikes = xp.zeros(brain.n, xp.float32)
        spikes[fired] = 1.0
        return self.dense(spikes)

    def dense(self, spikes):
        """Also supports actual continuous transmitter release in the hybrid model."""
        brain, xp = self.brain, self.brain.xp
        spikes = xp.asarray(spikes, dtype=xp.float32)
        if spikes.shape != (brain.n,):
            raise ValueError("One release value per neuron required")
        current = xp.empty((brain.n, 1), xp.float32)
        self.kernel(
            ((brain.n + 3) // 4,),
            (128,),
            (brain._W.indptr, brain._W.indices, brain._W.data, spikes, current, np.int32(brain.n)),
        )
        return current
