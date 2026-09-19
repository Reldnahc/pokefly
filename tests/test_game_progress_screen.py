"""Fabricated outcomes test accounting only; never evidence of neural learning."""

import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    return importlib.import_module('screen_game_retention')


def panels():
    before = {
        'starter': True, 'route1_visited': False, 'first_starter': 50, 'first_battle': 80,
        'completed_outcomes': {'battle_win': 0, 'rival_win': 0, 'capture': 0, 'gym_win': 0},
    }
    rows = []
    for source in (401, 402):
        for seed in (4801, 4802):
            after = deepcopy(before)
            if source == 401:
                after['first_starter'], after['first_battle'] = 40, 70
                after['completed_outcomes']['battle_win'] = 1
                after['completed_outcomes']['rival_win'] = 1
            rows.append({'source_launch_seed': source, 'seed': seed,
                         'original': deepcopy(before), 'retained': after})
    return rows


def test_same_lineage_must_preserve_progress_and_improve_both_seeds(module):
    result = module.screen(panels(), [4801, 4802], 100)
    assert result['passed'] and not result['new_trials']
    assert [row['passed'] for row in result['candidates']] == [True, False]
    assert result['independent_confirmation_still_required']
    assert not result['statistical_significance_claimed']


def test_speed_without_any_additional_progress_does_not_pass(module):
    rows = panels()
    for row in rows:
        row['retained']['completed_outcomes'] = deepcopy(row['original']['completed_outcomes'])
    assert not module.screen(rows, [4801, 4802], 100)['passed']


def test_cherry_picking_a_different_lineage_on_each_seed_does_not_pass(module):
    rows = panels()
    rows[3]['retained'] = deepcopy(rows[1]['retained'])
    rows[1]['retained'] = deepcopy(rows[1]['original'])
    assert not module.screen(rows, [4801, 4802], 100)['passed']


def test_cannot_cherry_pick_a_different_efficiency_endpoint_per_seed(module):
    rows = panels()
    rows[0]['retained']['first_battle'] = 80
    rows[1]['retained']['first_starter'] = 50
    assert not module.screen(rows, [4801, 4802], 100)['passed']


def test_lost_milestone_on_one_seed_cannot_be_offset_by_extra_wins(module):
    rows = panels()
    for row in rows:
        if row['seed'] == 4802:
            row['original']['route1_visited'] = True
    rows[0]['retained']['completed_outcomes']['battle_win'] = 100
    assert not module.screen(rows, [4801, 4802], 100)['passed']


def test_paid_unfinished_reward_is_not_completed_progress(module):
    rows = panels()
    for row in rows:
        after = row['retained']
        after['awarded_outcomes'] = after['completed_outcomes']
        after['completed_outcomes'] = deepcopy(row['original']['completed_outcomes'])
    assert not module.screen(rows, [4801, 4802], 100)['passed']


def test_censored_arrival_uses_conservative_budget_bound_and_exact_threshold(module):
    result = module.arrival_gain(None, 90, 100)
    assert result['original_right_censored'] and result['at_least_ten_percent']
    assert result['gain_fraction_lower_bound'] == pytest.approx(.1)
    assert not module.arrival_gain(None, 91, 100)['at_least_ten_percent']
    assert not module.arrival_gain(None, None, 100)['at_least_ten_percent']
    assert module.arrival_gain(50, 45, 100)['at_least_ten_percent']
    assert not module.arrival_gain(50, 46, 100)['at_least_ten_percent']


@pytest.mark.parametrize('value', [0, -1, 101, 1.5, True])
def test_out_of_budget_or_invalid_arrival_is_rejected(module, value):
    with pytest.raises(ValueError, match='sampled decision'):
        module.arrival_gain(50, value, 100)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'different_original', 'one_source'])
def test_incomplete_or_selected_panel_is_rejected(module, change):
    rows = panels()
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif change == 'different_original':
        rows[0]['original']['first_starter'] = 60
    else:
        rows = rows[:2]
    with pytest.raises(ValueError):
        module.screen(rows, [4801, 4802], 100)
