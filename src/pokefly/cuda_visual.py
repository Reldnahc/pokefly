"""Optional fused visual update with the original float32 operation order.

This changes kernel dispatch only, not neurons, equations, parameters, time
steps or learning. Separate round-to-nearest operations prohibit FMA changes.
"""

import numpy as np

KERNEL = r"""
extern "C" __global__ void visual_update(
    const float* current, const float* bias, const float* drive,
    const float* alpha, float* voltage, float* rate, int n, float maximum) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float delta = __fadd_rn(current[i], bias[i]);
    delta = __fadd_rn(delta, drive[i]);
    delta = __fsub_rn(delta, voltage[i]);
    delta = __fmul_rn(alpha[i], delta);
    const float value = __fadd_rn(voltage[i], delta);
    voltage[i] = value;
    rate[i] = value < 0.0f ? 0.0f : (value > maximum ? maximum : value);
}
"""


class CUDAVisualUpdate:
    def __init__(self, xp):
        self.xp = xp
        self.kernel = xp.RawKernel(KERNEL, "visual_update", options=("--fmad=false",))

    def __call__(self, current, bias, drive, alpha, voltage, maximum):
        if any(
            a.dtype != self.xp.float32 or a.shape != voltage.shape or not a.flags.c_contiguous
            for a in (current, bias, drive, alpha, voltage)
        ):
            raise ValueError("Aligned contiguous float32 visual arrays required")
        if voltage.ndim != 1 or not 0 < maximum <= 5:
            raise ValueError("One-dimensional visual voltage and a positive rate cap required")
        rate = self.xp.empty_like(voltage)
        self.kernel(
            ((voltage.size + 255) // 256,),
            (256,),
            (
                current,
                bias,
                drive,
                alpha,
                voltage,
                rate,
                np.int32(voltage.size),
                np.float32(maximum),
            ),
        )
        return rate
