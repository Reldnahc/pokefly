"""Experimental local mean-input constraint on existing positive synapses.

This is a modeling hypothesis, not calibrated fly physiology. Preserve the
original-weight estimated input at the CURRENT local release means, rather
than only projecting an incremental update before clipping. Nothing here
receives pixels, reward, action labels, desired motor rates or game state.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _mean_total(proposed, base, mean, order, start, end, multiplier, peak, minimum, maximum):
    total, slope = 0.0, 0.0
    for offset in range(start, end):
        edge = order[offset]
        m = np.float64(mean[edge]) / peak
        unbounded = np.float64(proposed[edge]) - multiplier * m
        factor = min(maximum, max(minimum, unbounded))
        total += np.float64(base[edge]) * m * factor
        if minimum < unbounded < maximum:
            slope += np.float64(base[edge]) * m * m
    return total, slope


@njit(cache=True, nogil=True)
def bounded_mean_factors(proposed, base, mean, order, indptr, minimum, maximum, budget):
    """Bounded positive-metric projection, then a feasible resource contraction.

For each target, solve f=clip(proposed-lambda*mean) with sum(base*mean*f)
equal to sum(base*mean). The positive base-weight metric gives this scalar
dual form. Zero-mean inputs are unconstrained by that equality, not deleted.

If the ordinary total-input budget is exceeded, contract towards ORIGINAL
factors (all ones). Both endpoints satisfy the mean constraint and box, so
this also satisfies the budget without breaking either previous constraint.
It is not the joint minimum-distance projection onto all three constraints.
budget<0 disables that last constraint; budget=0 requests exact total input.
Float32 output has ordinary final-rounding residuals, not exact real arithmetic.
"""
    result = np.empty(len(proposed), np.float32)
    for target in range(len(indptr) - 1):
        start, end = indptr[target], indptr[target + 1]
        if start == end:
            continue
        reference, resource, peak = 0.0, 0.0, 0.0
        for offset in range(start, end):
            peak = max(peak, np.float64(mean[order[offset]]))
        for offset in range(start, end):
            edge = order[offset]
            b = np.float64(base[edge])
            m = np.float64(mean[edge]) / peak if peak > 0 else 0.0
            reference += b * m
            resource += b
        multiplier = 0.0
        if reference > 0:
            # Normalize within a target and bracket from zero. Extrema of
            # (proposed-bound)/mean give unusably huge brackets when a silent
            # cell's EMA has decayed to a float32 subnormal (~1e-45).
            args = (proposed, base, mean, order, start, end)
            total, slope = _mean_total(*args, 0.0, peak, minimum, maximum)
            if abs(total - reference) > 1e-12 * reference:
                if total > reference:
                    lower, upper = 0.0, 1.0
                    for _ in range(256):
                        if _mean_total(*args, upper, peak, minimum, maximum)[0] <= reference:
                            break
                        upper *= 2.0
                else:
                    lower, upper = -1.0, 0.0
                    for _ in range(256):
                        if _mean_total(*args, lower, peak, minimum, maximum)[0] >= reference:
                            break
                        lower *= 2.0
                for _ in range(64):
                    if abs(total - reference) <= 1e-12 * reference:
                        break
                    if total > reference:
                        lower = multiplier
                    else:
                        upper = multiplier
                    # This dual is piecewise linear. Newton usually resolves
                    # an unchanged active set in one step; retain bisection
                    # when clipping removes every local slope or leaves bracket.
                    candidate = multiplier + (total - reference) / slope if slope > 0 else np.inf
                    multiplier = (
                        candidate if lower < candidate < upper
                        else lower + (upper - lower) * 0.5
                    )
                    total, slope = _mean_total(*args, multiplier, peak, minimum, maximum)
        total = 0.0
        for offset in range(start, end):
            edge = order[offset]
            m = np.float64(mean[edge]) / peak if peak > 0 else 0.0
            factor = min(maximum, max(
                minimum, np.float64(proposed[edge]) - multiplier * m
            ))
            # Retain float64 until both constraints have been applied below.
            total += np.float64(base[edge]) * factor
        scale = 1.0
        if budget >= 0 and total != resource:
            scale = min(1.0, budget * resource / abs(total - resource))
        for offset in range(start, end):
            edge = order[offset]
            m = np.float64(mean[edge]) / peak if peak > 0 else 0.0
            factor = min(maximum, max(
                minimum, np.float64(proposed[edge]) - multiplier * m
            ))
            result[edge] = 1.0 + scale * (factor - 1.0)
    return result


def target_order(post, n):
    """Deterministic target grouping, computed once, not checkpoint state."""
    order = np.argsort(post, kind="stable")
    indptr = np.concatenate(([0], np.cumsum(np.bincount(post, minlength=n))))
    return order, indptr
