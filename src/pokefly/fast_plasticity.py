"""Fused CPU eligibility update, preserving the NumPy 2 rounding sequence.

This is numerical acceleration only: no different signals or learning rule.
No fast-math, parallel reduction, random draws, or changed summation order.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def sensorimotor_eligibility(pre, post, pre_trace, baseline, spike, eligibility, scale, decay):
    divisor = np.float32(scale)
    gain = 1.0 - decay
    for edge in range(len(pre)):
        incoming = np.float32(pre_trace[pre[edge]] / divisor)
        incoming = min(np.float32(5.0), max(np.float32(0.0), incoming))
        outgoing = np.float32(np.float32(spike[post[edge]] - baseline[post[edge]]) / divisor)
        previous = np.float32(np.float64(eligibility[edge]) * decay)
        coincidence = (gain * np.float64(incoming)) * np.float64(outgoing)
        value = np.float32(np.float64(previous) + coincidence)
        eligibility[edge] = min(np.float32(5.0), max(np.float32(-5.0), value))
