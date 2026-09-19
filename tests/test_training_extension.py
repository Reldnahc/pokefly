"""Fabricated outcomes validate dose-test accounting, not neural learning."""

import argparse
import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

FAMILIES = [(4501, 61004), (4502, 62004)]
SEEDS = [63001, 63002]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    return importlib.import_module('screen_training_extension')


def panels():
    original = {
        'starter': True, 'route1_visited': False, 'first_starter': 50, 'first_battle': 80,
        'completed_outcomes': {'battle_win': 0, 'rival_win': 0, 'capture': 0, 'gym_win': 0},
    }
    rows = []
    for source in (4501, 4502, 61004, 62004):
        for seed in SEEDS:
            after = deepcopy(original)
            after['first_starter'], after['first_battle'] = 40, 60
            after['completed_outcomes'].update(battle_win=1, rival_win=1)
            if source == 61004:
                after.update(first_starter=30, first_battle=50, route1_visited=True)
            rows.append({'source_launch_seed': source, 'seed': seed,
                         'original': deepcopy(original), 'retained': after})
    return rows


def family_rows(rows, source):
    return [row['retained'] for row in rows if row['source_launch_seed'] == source]


def test_child_must_improve_against_original_and_its_own_parent(module):
    rows = panels()
    before = deepcopy(rows)
    result = module.screen_extension(rows, SEEDS, 100, FAMILIES)
    assert rows == before
    assert [row['passed'] for row in result['families']] == [True, False]
    assert result['passed'] and result['related_checkpoint_families'] == 2
    assert result['independent_confirmation_still_required']
    assert not result['new_trials'] and not result['statistical_significance_claimed']


def test_parent_only_success_cannot_qualify_additional_training(module):
    rows = panels()
    for row in rows:
        if row['source_launch_seed'] in (61004, 62004):
            row['retained'] = deepcopy(row['original'])
    assert not module.screen_extension(rows, SEEDS, 100, FAMILIES)['passed']


def test_consistent_parent_relative_efficiency_can_qualify_without_new_milestone(module):
    rows = panels()
    for after in family_rows(rows, 61004):
        after['route1_visited'] = False
    candidate = module.screen_extension(rows, SEEDS, 100, FAMILIES)['families'][0]
    assert candidate['passed'] and not candidate['additional_parent_progress']
    assert candidate['consistently_faster_parent_relative_arrivals'] == [
        'first_starter', 'first_battle',
    ]


def test_additional_parent_progress_can_qualify_without_faster_parent_arrival(module):
    rows = panels()
    for after in family_rows(rows, 61004):
        after.update(first_starter=40, first_battle=60)
    candidate = module.screen_extension(rows, SEEDS, 100, FAMILIES)['families'][0]
    assert candidate['passed'] and candidate['additional_parent_progress']
    assert not candidate['consistently_faster_parent_relative_arrivals']


def test_parent_regression_cannot_be_offset_by_extra_wins(module):
    rows = panels()
    for before in family_rows(rows, 4501):
        before['route1_visited'] = True
    for after in family_rows(rows, 61004):
        after['route1_visited'] = False
        after['completed_outcomes']['battle_win'] = 100
    assert not module.screen_extension(rows, SEEDS, 100, FAMILIES)['passed']


def test_parent_relative_speed_endpoint_must_match_on_both_seeds(module):
    rows = panels()
    first, second = family_rows(rows, 61004)
    first.update(route1_visited=False, first_battle=60)
    second.update(route1_visited=False, first_starter=40)
    assert not module.screen_extension(rows, SEEDS, 100, FAMILIES)['passed']


def test_missing_parent_arrival_is_censored_not_zero(module):
    rows = panels()
    for before in family_rows(rows, 4501):
        before.update(starter=False, first_starter=None, first_battle=None)
        before['completed_outcomes'].update(battle_win=0, rival_win=0)
    candidate = module.screen_extension(rows, SEEDS, 100, FAMILIES)['families'][0]
    assert candidate['passed']
    arrival = candidate['parent_comparisons'][0]['parent_relative_arrival']['first_starter']
    assert arrival['original_right_censored'] and arrival['original'] is None
    assert arrival['gain_fraction_lower_bound'] == pytest.approx(.7)


@pytest.mark.parametrize('families', [
    [(4501, 61004)], [(4501, 61004), (4501, 62004)],
    [(4501, 61004), (4502, 999)], [(4501,), (4502, 62004)],
    [(True, 61004), (4502, 62004)],
])
def test_selected_missing_or_malformed_families_fail_closed(module, families):
    with pytest.raises(ValueError):
        module.screen_extension(panels(), SEEDS, 100, families)


@pytest.mark.parametrize('value', ['401', 'a:402', '1:2:3'])
def test_family_cli_requires_explicit_seed_pair(module, value):
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_family(value)
    assert module.parse_family('4501:61004') == (4501, 61004)


def ancestry(module, tmp_path, monkeypatch):
    runs = [tmp_path / name for name in ('parent', 'middle', 'child')]
    checkpoints = [path / 'checkpoints' / 'pinned' for path in runs]
    for index, (run, checkpoint) in enumerate(zip(runs, checkpoints, strict=True)):
        checkpoint.mkdir(parents=True)
        (checkpoint / 'brain.npz').write_bytes(f'fixture-{index}'.encode())
        options = {} if index == 0 else {
            'resume' if index == 1 else 'weights': str(checkpoints[index - 1]),
        }
        (run / 'config.json').write_text(json.dumps({'options': options}))
    monkeypatch.setattr(module, 'read_checkpoint', lambda path: ({}, {'directory': str(path)}))
    parent = {'checkpoint': str(checkpoints[0]), 'run': str(runs[0]),
              'source_launch_seed': 4501,
              'brain_sha256': module.sha256(checkpoints[0] / 'brain.npz')}
    child = {'checkpoint': str(checkpoints[2]), 'run': str(runs[2]), 'source_launch_seed': 61004}
    return parent, child, runs, checkpoints


def test_family_proof_follows_exact_checkpoint_links_and_resume(module, tmp_path, monkeypatch):
    parent, child, _, _ = ancestry(module, tmp_path, monkeypatch)
    result = module.verify_parent_checkpoint(parent, child)
    assert result['exact_parent_checkpoint_verified'] and len(result['links']) == 2
    assert result['parent'] == 4501 and result['child'] == 61004


@pytest.mark.parametrize('kind', ['unrelated', 'cycle', 'ambiguous', 'checksum', 'copied'])
def test_family_proof_rejects_unrelated_or_damaged_ancestry(
    module, tmp_path, monkeypatch, kind,
):
    parent, child, runs, checkpoints = ancestry(module, tmp_path, monkeypatch)
    options = {'resume': str(checkpoints[0])}
    if kind == 'unrelated':
        options = {}
    elif kind == 'cycle':
        options = {'weights': str(checkpoints[2])}
    elif kind == 'ambiguous':
        options['weights'] = str(checkpoints[0])
    elif kind == 'checksum':
        parent['brain_sha256'] = 'changed'
    else:
        options = {'weights': str(tmp_path / 'copied' / 'checkpoint')}
    (runs[1] / 'config.json').write_text(json.dumps({'options': options}))
    with pytest.raises(ValueError):
        module.verify_parent_checkpoint(parent, child)
