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

# The selected-row variant uses EXACTLY the same inner sum/reduction. Skipped
# rows are zero, so it may only replace a full pass where those currents are
# discarded by the caller. It does not remove a graph edge or change weights.
ROW_KERNEL = KERNEL.replace(
    "propagate(", "propagate_rows("
).replace(
    "float* output, int n)", "float* output, const int* rows, int n)"
).replace(
    "const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;",
    "const int work = (blockIdx.x * blockDim.x + threadIdx.x) / 32;",
).replace(
    "if (row >= n) return;", "if (work >= n) return;\n    const int row = rows[work];"
)


class DeterministicCUDAInput:
    def __init__(self, brain, *, rows=None):
        self.brain = brain
        if brain.batch != 1 or brain._W.indices.dtype != np.int32:
            raise ValueError("Deterministic CUDA propagation requires single-fly int32 CSR")
        self.rows = None
        self.output_rows = brain.n
        if rows is not None:
            rows = np.asarray(rows)
            if (
                rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows)
                or (rows < 0).any() or (rows >= brain.n).any()
                or (rows[1:] <= rows[:-1]).any()
            ):
                raise ValueError("Selected rows must be nonempty, sorted unique neuron indices")
            self.rows = brain.xp.asarray(rows.astype(np.int32, copy=True))
            self.output_rows = len(rows)
        self.kernel = brain.xp.RawKernel(
            KERNEL if self.rows is None else ROW_KERNEL,
            "propagate" if self.rows is None else "propagate_rows", options=("--fmad=false",),
        )

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
        current = (xp.empty if self.rows is None else xp.zeros)((brain.n, 1), xp.float32)
        args = (brain._W.indptr, brain._W.indices, brain._W.data, spikes, current)
        if self.rows is not None:
            args += (self.rows,)
        self.kernel(
            ((self.output_rows + 3) // 4,),
            (128,),
            (*args, np.int32(self.output_rows)),
        )
        return current
