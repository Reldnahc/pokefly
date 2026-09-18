"""Fused CPU eligibility update, preserving the NumPy 2 rounding sequence.

This is numerical acceleration only: no different signals or learning rule.
No fast-math, parallel reduction, random draws, or changed summation order.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def likelihood_eligibility(
    pre, post, base, release, spike, probability, membrane_trace, eligibility,
    membrane_decay, eligibility_decay, gain, temperature,
    release_baseline=None,
):
    """Local log-likelihood score with respect to each original-weight factor.

    Given the recorded spike history, membrane sensitivity integrates actual
    incoming transmitter release and resets AFTER evaluating the emitted spike.
    No button, stimulus identity, target spike, or external gradient is used.
    """
    for edge in range(len(pre)):
        j = post[edge]
        incoming = release[pre[edge]]
        if release_baseline is not None:
            # Optional centered innovation, not the original likelihood gradient.
            incoming -= release_baseline[pre[edge]]
        trace = membrane_decay * membrane_trace[edge] + incoming
        score = (spike[j] - probability[j]) * base[edge] * gain * trace / temperature
        eligibility[edge] = eligibility_decay * eligibility[edge] + score
        membrane_trace[edge] = 0.0 if spike[j] else trace


@njit(cache=True, nogil=True)
def sensorimotor_eligibility(
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
    divisor = np.float32(scale)
    gain = 1.0 - decay
    for edge in range(len(pre)):
        incoming = np.float32(pre_trace[pre[edge]] / divisor)
        incoming = min(np.float32(5.0), max(np.float32(incoming_floor), incoming))
        outgoing = np.float32(np.float32(spike[post[edge]] - baseline[post[edge]]) / divisor)
        previous = np.float32(np.float64(eligibility[edge]) * decay)
        coincidence = (gain * np.float64(incoming)) * np.float64(outgoing)
        value = np.float32(np.float64(previous) + coincidence)
        if clip_eligibility:
            eligibility[edge] = min(np.float32(5.0), max(np.float32(-5.0), value))
        else:
            eligibility[edge] = value
        if slow_eligibility is not None:
            previous_slow = np.float32(np.float64(slow_eligibility[edge]) * slow_decay)
            coincidence_slow = ((1.0 - slow_decay) * np.float64(incoming)) * np.float64(outgoing)
            value_slow = np.float32(np.float64(previous_slow) + coincidence_slow)
            if clip_eligibility:
                slow_eligibility[edge] = min(np.float32(5.0), max(np.float32(-5.0), value_slow))
            else:
                slow_eligibility[edge] = value_slow
