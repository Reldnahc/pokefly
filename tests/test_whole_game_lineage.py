'''Ancestry fixtures test provenance only, not evidence of learned behavior.'''

import importlib
import json
from pathlib import Path

import pytest


def lineage(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    module = importlib.import_module('train_game_series')
    runs = [tmp_path / name for name in ('initial', 'practice', 'continued')]
    snapshots = {}
    for index, run in enumerate(runs):
        run.mkdir()
        checkpoint = run / 'checkpoints' / 'step-00000016-fixture'
        checkpoint.mkdir(parents=True)
        options = {'mode': 'learn', 'seed': 401 + index, 'intro': index < 2, 'load_state': None,
                   'weights': str(runs[0] / 'latest-checkpoint.json') if index == 1 else None,
                   'resume': str(runs[1] / 'latest-checkpoint.json') if index == 2 else None}
        stored = {'options': options, 'config': {}, 'rom_sha1': 'same-rom'}
        (run / 'config.json').write_text(json.dumps(stored))
        snapshots[str(run / 'latest-checkpoint.json')] = ({}, {
            'directory': str(checkpoint), 'rom_sha1': 'same-rom',
            'experiment': {'mode': 'learn', 'sample': 16, 'config': {}},
        })
    monkeypatch.setattr(module, 'read_checkpoint', lambda path: snapshots[str(path)])
    return module, runs, snapshots


def change(run, change_record):
    path = run / 'config.json'
    stored = json.loads(path.read_text())
    change_record(stored)
    path.write_text(json.dumps(stored))


def test_whole_game_weights_and_exact_resume_keep_original_provenance(tmp_path, monkeypatch):
    module, runs, _ = lineage(tmp_path, monkeypatch)
    before = [(run / 'config.json').read_bytes() for run in runs]
    for run in runs:
        module.verify_whole_game_lineage(run)
    assert before == [(run / 'config.json').read_bytes() for run in runs]


@pytest.mark.parametrize('index', [0, 1, 2])
def test_stage_start_rejected_at_every_ancestral_depth(tmp_path, monkeypatch, index):
    module, runs, _ = lineage(tmp_path, monkeypatch)
    change(runs[index], lambda value: value['options'].update(load_state='battle.state'))
    with pytest.raises(ValueError, match='Stage-specific'):
        module.verify_whole_game_lineage(runs[-1])


@pytest.mark.parametrize('kind', ['cycle', 'ambiguous', 'rom', 'mode', 'model', 'missing'])
def test_malformed_or_cross_model_ancestry_fails_closed(tmp_path, monkeypatch, kind):
    module, runs, snapshots = lineage(tmp_path, monkeypatch)
    if kind == 'cycle':
        cycle = str(runs[2] / 'latest-checkpoint.json')
        change(runs[0], lambda v: v['options'].update(weights=cycle))
    elif kind == 'ambiguous':
        change(runs[1], lambda v: v['options'].update(resume='another-checkpoint'))
    elif kind == 'rom':
        change(runs[0], lambda v: v.update(rom_sha1='different-rom'))
    elif kind == 'mode':
        change(runs[0], lambda v: v['options'].update(mode='frozen'))
    elif kind == 'model':
        change(runs[2], lambda v: v['config'].update(frames=48))
    else:
        snapshots[str(runs[1] / 'latest-checkpoint.json')][1]['directory'] = str(tmp_path / 'copy')
    with pytest.raises(ValueError):
        module.verify_whole_game_lineage(runs[-1])


def test_fresh_power_on_is_whole_game_without_scripted_intro(tmp_path, monkeypatch):
    module, runs, _ = lineage(tmp_path, monkeypatch)
    change(runs[0], lambda v: v['options'].update(intro=False))
    module.verify_whole_game_lineage(runs[-1])


def test_completed_source_checks_ancestry_before_accepting_brain(tmp_path, monkeypatch):
    module, runs, _ = lineage(tmp_path, monkeypatch)
    (runs[-1] / 'summary.json').write_text(json.dumps({'reason': 'step_limit', 'samples': 16}))
    change(runs[0], lambda value: value['options'].update(load_state='battle.state'))
    with pytest.raises(ValueError, match='Stage-specific'):
        module.completed_game_source(runs[-1])


@pytest.mark.parametrize('field', ['rom', 'model'])
def test_final_checkpoint_must_match_its_run_identity(tmp_path, monkeypatch, field):
    module, runs, snapshots = lineage(tmp_path, monkeypatch)
    (runs[-1] / 'summary.json').write_text(json.dumps({'reason': 'step_limit', 'samples': 16}))
    saved = snapshots[str(runs[-1] / 'latest-checkpoint.json')][1]
    if field == 'rom':
        saved['rom_sha1'] = 'different-rom'
    else:
        saved['experiment']['config']['frames'] = 48
    with pytest.raises(ValueError, match='final completed'):
        module.completed_game_source(runs[-1])


def test_damaged_parent_checkpoint_is_not_treated_as_fresh_brain(tmp_path, monkeypatch):
    module, runs, _ = lineage(tmp_path, monkeypatch)

    def corrupt(path):
        raise ValueError('Incomplete or modified checkpoint')

    monkeypatch.setattr(module, 'read_checkpoint', corrupt)
    with pytest.raises(ValueError, match='Incomplete or modified'):
        module.verify_whole_game_lineage(runs[-1])


@pytest.mark.parametrize('seed', [401, 402])
def test_heldout_seeds_exclude_every_ancestral_fresh_game(tmp_path, monkeypatch, seed):
    module, runs, _ = lineage(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='overlaps ancestral'):
        module.verify_whole_game_lineage(runs[-1], heldout_seeds=[seed, 9001])


def test_unseen_seeds_and_unused_exact_resume_launch_seed_are_allowed(tmp_path, monkeypatch):
    module, runs, _ = lineage(tmp_path, monkeypatch)
    before = [(run / 'config.json').read_bytes() for run in runs]
    module.verify_whole_game_lineage(runs[-1], heldout_seeds=[403, 9001])
    assert before == [(run / 'config.json').read_bytes() for run in runs]


@pytest.mark.parametrize('seed', [None, True, '401'])
def test_unknown_ancestral_training_seed_fails_closed_when_testing_holdout(
    tmp_path, monkeypatch, seed,
):
    module, runs, _ = lineage(tmp_path, monkeypatch)
    change(runs[0], lambda value: value['options'].update(seed=seed))
    with pytest.raises(ValueError, match='valid training launch seed'):
        module.verify_whole_game_lineage(runs[-1], heldout_seeds=[9001])


def test_completed_source_checks_heldout_ancestry_before_loading_weights(tmp_path, monkeypatch):
    module, runs, _ = lineage(tmp_path, monkeypatch)
    (runs[-1] / 'summary.json').write_text(json.dumps({'reason': 'step_limit', 'samples': 16}))
    with pytest.raises(ValueError, match='overlaps ancestral'):
        module.completed_game_source(runs[-1], heldout_seeds=[401, 9001])


@pytest.mark.parametrize('seeds', [[True], ['401'], [1, True]])
def test_holdout_seed_types_are_explicit(tmp_path, monkeypatch, seeds):
    module, runs, _ = lineage(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='must be integers'):
        module.verify_whole_game_lineage(runs[-1], heldout_seeds=seeds)
