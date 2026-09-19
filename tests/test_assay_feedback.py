"""Timing/provenance fixtures, not evidence that any neural model learns."""

import importlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    return importlib.import_module('assay_feedback')


def history(m, delay, size=64):
    rows = []
    for i in range(size):
        rows.append(m.feedback_row(
            'left', ('left', 'right')[(i // 16) % 2], float(i % 3 == 0), rows, delay,
        ))
    return rows


@pytest.mark.parametrize('delay', [0, 1, 8, 20])
def test_feedback_matures_without_retroactive_credit_or_tail_flush(monkeypatch, delay):
    m = module(monkeypatch)
    rows = history(m, delay)
    earned = [float(i % 3 == 0) for i in range(64)]
    expected = [0.0] * delay + earned[:64 - delay]
    assert [r['reward'] for r in rows] == expected
    assert ('earned_reward' in rows[0]) == bool(delay)
    m.validate_history(rows, delay)
    # Exact continuation delivers the old pending queue; no zeroed restart.
    row = m.feedback_row('right', 'left', 0.05, rows, delay)
    assert row['reward'] == (earned[-delay] if delay else 0.05)


@pytest.mark.parametrize('delay', [1, 8, 20, 40])
def test_shuffling_matches_delivered_feedback_and_separate_pending_tail(monkeypatch, delay):
    m = module(monkeypatch)
    paired = history(m, delay)
    schedule = m.shuffled_feedback(paired, [32, 64], ('left', 'right'), 5, delay)
    control = []
    for row, earned in zip(paired, schedule, strict=True):
        control.append(m.feedback_row('up', row['cue'], earned, control, delay))
    m.validate_history(control, delay)
    for end in (32, 64):
        delivered = max(0, end - delay)
        for cue in ('left', 'right'):
            assert Counter(r['earned_reward'] for r in paired[:delivered] if r['cue'] == cue) == (
                Counter(r['earned_reward'] for r in control[:delivered] if r['cue'] == cue)
            )
        assert sum(r['reward'] for r in paired[:end]) == sum(r['reward'] for r in control[:end])
    for cue in ('left', 'right'):
        assert Counter(r['earned_reward'] for r in paired[-delay:] if r['cue'] == cue) == (
            Counter(r['earned_reward'] for r in control[-delay:] if r['cue'] == cue)
        )


def test_zero_delay_keeps_historical_permutation_exact(monkeypatch):
    m = module(monkeypatch)
    rows = history(m, 0)
    expected = np.array([r['reward'] for r in rows])
    rng = np.random.default_rng(5 + 712345)
    for start, end in ((0, 32), (32, 64)):
        for cue in ('left', 'right'):
            indices = np.array([i for i in range(start, end) if rows[i]['cue'] == cue])
            expected[indices] = rng.permutation(expected[indices])
    np.testing.assert_array_equal(
        m.shuffled_feedback(rows, [32, 64], ('left', 'right'), 5), expected,
    )


@pytest.mark.parametrize('delay', [-1, True, 0.5, '8'])
def test_invalid_delay_rejected(monkeypatch, delay):
    m = module(monkeypatch)
    with pytest.raises(ValueError, match='nonnegative integer'):
        m.delivered_feedback([], 1.0, delay)


def test_corrupt_delayed_history_cannot_resume(monkeypatch):
    m = module(monkeypatch)
    rows = history(m, 8)
    rows[8]['reward'] = 0.5
    with pytest.raises(ValueError, match='registered delay'):
        m.validate_history(rows, 8)
    del rows[0]['earned_reward']
    with pytest.raises(KeyError):
        m.validate_history(rows, 8)


@pytest.mark.parametrize('delay', ['-1', '32', '1.5'])
def test_curve_rejects_invalid_delay_before_creating_brain(monkeypatch, delay):
    module(monkeypatch)
    curve = importlib.import_module('probe_visual_curve')
    monkeypatch.setattr(curve, 'InternalBrain', lambda **kw: pytest.fail('Started a brain'))
    monkeypatch.setattr(sys, 'argv', ['probe_visual_curve.py', '--checkpoints', '32',
                                     '--reward-delay-decisions', delay])
    with pytest.raises(SystemExit):
        curve.main()
