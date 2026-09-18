"""Experimental bit-matched CUDA eligibility kernel; no learning-rule change.

Host arrays remain authoritative. Only immutable anatomical edge indices are
cached on the device; each call transfers current values in and out. This
avoids hidden state or checkpoint/cache invalidation requirements.
"""

import numpy as np

KERNEL = r"""
extern "C" __global__ void eligibility_step(
    const int* pre, const int* post, const float* pre_trace,
    const float* baseline, const float* spike, float* eligibility,
    float* slow, const float scale, const double decay, const double slow_decay,
    const float floor_value, const int clip, const int has_slow, const int edges) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= edges) return;
    float incoming = pre_trace[pre[edge]] / scale;
    incoming = fminf(5.0f, fmaxf(floor_value, incoming));
    float outgoing = (spike[post[edge]] - baseline[post[edge]]) / scale;
    float previous = (float)((double)eligibility[edge] * decay);
    double coincidence = ((1.0 - decay) * (double)incoming) * (double)outgoing;
    float value = (float)((double)previous + coincidence);
    eligibility[edge] = clip ? fminf(5.0f, fmaxf(-5.0f, value)) : value;
    if (has_slow) {
        float prior_slow = (float)((double)slow[edge] * slow_decay);
        double event_slow = ((1.0 - slow_decay) * (double)incoming) * (double)outgoing;
        float value_slow = (float)((double)prior_slow + event_slow);
        slow[edge] = clip ? fminf(5.0f, fmaxf(-5.0f, value_slow)) : value_slow;
    }
}
"""


class CUDAEligibility:
    def __init__(self, pre, post, xp):
        if len(pre) != len(post) or len(pre) > np.iinfo(np.int32).max:
            raise ValueError("Invalid CUDA eligibility edge count")
        for indices in (pre, post):
            if (
                indices.dtype.kind not in "iu"
                or (indices < 0).any()
                or (len(indices) and indices.max() > np.iinfo(np.int32).max)
            ):
                raise ValueError("CUDA eligibility requires nonnegative int32-sized indices")
        self.host_pre, self.host_post, self.xp = pre, post, xp
        self.required_n = int(max(pre.max(initial=0), post.max(initial=0))) + 1
        self.pre = xp.asarray(pre, dtype=xp.int32)
        self.post = xp.asarray(post, dtype=xp.int32)
        self.kernel = xp.RawKernel(KERNEL, "eligibility_step", options=("--fmad=false",))

    def __call__(
        self,
        pre,
        post,
        pre_trace,
        baseline,
        spike,
        eligibility,
        scale,
        decay,
        clip_eligibility=True,
        incoming_floor=0.0,
        slow_eligibility=None,
        slow_decay=0.0,
    ):
        if pre is not self.host_pre or post is not self.host_post:
            raise ValueError("CUDA eligibility edge identity changed")
        arrays = (pre_trace, baseline, spike, eligibility)
        if any(a.dtype != np.float32 or a.ndim != 1 for a in arrays):
            raise ValueError("CUDA eligibility expects one-dimensional float32 arrays")
        if (
            not pre_trace.shape == baseline.shape == spike.shape
            or len(spike) < self.required_n
            or len(eligibility) != len(pre)
        ):
            raise ValueError("CUDA eligibility array sizes do not match anatomy")
        if slow_eligibility is not None and (
            slow_eligibility.dtype != np.float32 or slow_eligibility.shape != eligibility.shape
        ):
            raise ValueError("Invalid slow CUDA eligibility array")
        xp = self.xp
        values = [xp.asarray(a) for a in arrays]
        slow = xp.asarray(slow_eligibility) if slow_eligibility is not None else values[-1]
        if len(pre):
            self.kernel(
                ((len(pre) + 255) // 256,),
                (256,),
                (
                    self.pre,
                    self.post,
                    *values,
                    slow,
                    np.float32(scale),
                    np.float64(decay),
                    np.float64(slow_decay),
                    np.float32(incoming_floor),
                    np.int32(clip_eligibility),
                    np.int32(slow_eligibility is not None),
                    np.int32(len(pre)),
                ),
            )
        values[-1].get(out=eligibility)
        if slow_eligibility is not None:
            slow.get(out=slow_eligibility)
