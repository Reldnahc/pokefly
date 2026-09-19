"""Scalar feedback timing for ROM-free assays; never a gameplay controller.

Delayed feedback refers to an earlier observed choice without storing or
replacing neural eligibility. Unmatured feedback stays in the saved history;
there is no special post-training flush or frozen-test reward.
"""

import math

import numpy as np


def validate_delay(value):
    if type(value) is not int or value < 0:
        raise ValueError('Feedback delay must be a nonnegative integer')
    return value


def earned_feedback(row, delay):
    value = row['earned_reward'] if delay else row['reward']
    if not math.isfinite(value):
        raise ValueError('Assay feedback must be finite')
    return float(value)


def delivered_feedback(history, earned, delay):
    """Reward at this decision, not a retroactive edit to the old neural state."""
    validate_delay(delay)
    if not math.isfinite(earned):
        raise ValueError('Assay feedback must be finite')
    if not delay:
        return float(earned)
    return 0.0 if len(history) < delay else earned_feedback(history[-delay], delay)


def feedback_row(action, cue, earned, history, delay):
    delivered = delivered_feedback(history, earned, delay)
    return {'action': action, 'cue': cue, 'reward': delivered,
            **({'earned_reward': float(earned)} if delay else {})}


def validate_history(history, delay):
    validate_delay(delay)
    for index, row in enumerate(history):
        earned = earned_feedback(row, delay)
        expected = (earned if not delay else 0.0 if index < delay
                    else earned_feedback(history[index - delay], delay))
        if row['reward'] != expected:
            raise ValueError('Saved feedback does not match its registered delay')


def shuffled_feedback(history, checkpoints, cues, seed, delay=0, *, starting_step=0):
    """Match feedback by originating cue AND delivery checkpoint interval.

The final pending tail is shuffled separately. Otherwise shuffling rewards
across that boundary would change how much feedback an arm actually receives.
At delay zero this preserves the historical permutation and RNG call order.
"""
    validate_history(history, delay)
    if len(history) != checkpoints[-1]:
        raise ValueError('Complete paired history required')
    values = np.array([earned_feedback(row, delay) for row in history], dtype=float)
    labels = np.array([row['cue'] for row in history])
    rng = np.random.default_rng(seed + 712345)
    previous = starting_step
    boundaries = sorted({max(starting_step, end - delay) for end in checkpoints}
                        | {checkpoints[-1]})
    for end in boundaries:
        if end <= previous:
            continue
        for cue in cues:
            indices = np.flatnonzero(labels[previous:end] == cue) + previous
            values[indices] = rng.permutation(values[indices])
        previous = end
    return values
