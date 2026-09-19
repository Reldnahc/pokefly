import copy
import importlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("derive_game_baseline")


def fixtures(monkeypatch):
    m = script(monkeypatch)
    original_config = "configs/visual-release-wide-v3.json"
    projected_config = "configs/visual-wide-projected-v4.json"
    fixed = asdict(m.load_config(Path(projected_config)))
    controls = [{"mode": mode, "run": mode} for mode in ("frozen", "learn")]
    row = {"run": "resumed", "samples": 6000, "tiles": 200}
    data = {
        "recovered.json": {
            "status": "completed", "rows": [row], "source_report": "candidate.json",
            "source_report_sha256": "digest", "seed": 401, "steps": 6000,
            "candidate_config": projected_config, "reused_controls_not_new_trials": controls,
            "source_trajectory_sha256": "digest", "source_checkpoint": "checkpoint",
            "source_brain_sha256": "digest", "overlap_decisions_exact": 20,
            "segments": [
                {"run": "partial", "first_sample": 1, "last_sample": 1000},
                {"run": "resumed", "first_sample": 1001, "last_sample": 6000},
            ],
        },
        "candidate.json": {
            "factor": "rule", "prefix_verification": {"all_fields_exact": True},
            "baseline_report": "base.json", "baseline_report_sha256": "digest",
            "config": projected_config, "config_sha256": "digest",
            "seed": 401, "steps": 6000, "reused_controls_not_new_trials": controls,
            "status": "running", "rows": [],
        },
        "base.json": {
            "status": "completed", "rows": controls, "seed": 401, "steps": 6000,
            "config": original_config, "config_sha256": "digest",
        },
        "partial/config.json": {
            "options": {"mode": "learn", "intro": True, "seed": 401, "steps": 6000},
            "config": fixed, "rom_sha1": "rom",
        },
        "resumed/config.json": {
            "options": {"resume": "checkpoint", "steps": 5000, "seed": 401},
        },
        "resumed/summary.json": {"reason": "step_limit"},
    }

    def checkpoint(path):
        old = path == Path("checkpoint")
        return {}, {"directory": "checkpoint" if old else "final-checkpoint", "rom_sha1": "rom",
                    "experiment": {"mode": "learn", "sample": 1000 if old else 6000,
                                   "config": fixed}}

    monkeypatch.setattr(m, "sha256", lambda path: "digest")
    monkeypatch.setattr(m, "read", lambda path: data[Path(path).as_posix()])
    monkeypatch.setattr(m, "read_checkpoint", checkpoint)
    monkeypatch.setattr(m, "read_rows", lambda *a, **kw: [])
    monkeypatch.setattr(m, "stitch", lambda *a: ([], 20))
    monkeypatch.setattr(m, "measure_complete", lambda *a: row)
    return m, data


def test_derived_baseline_keeps_shared_control_and_exact_candidate_identity(monkeypatch):
    m, data = fixtures(monkeypatch)
    before = copy.deepcopy(data)
    result = m.compose(Path("recovered.json"))
    assert result["new_trials"] is False
    assert result["frozen_control_shared_not_independent"]
    assert result["candidate_resume_segments_not_independent"]
    assert result["rows"][0] == data["base.json"]["rows"][0]
    assert result["rows"][1] == {"mode": "learn", **data["recovered.json"]["rows"][0]}
    assert result["config"] == data["candidate.json"]["config"]
    assert len(result["provenance"]) == 3 and data == before


@pytest.mark.parametrize("file,key,value", [
    ("recovered.json", "status", "running"),
    ("recovered.json", "source_report_sha256", "changed"),
    ("recovered.json", "seed", 999),
    ("recovered.json", "source_brain_sha256", "changed"),
    ("recovered.json", "overlap_decisions_exact", 19),
    ("candidate.json", "factor", "learning_rate"),
    ("candidate.json", "config_sha256", "changed"),
    ("base.json", "continuation_source", "different-trial"),
    ("resumed/summary.json", "reason", "interrupted"),
])
def test_derived_baseline_rejects_altered_or_incomplete_evidence(monkeypatch, file, key, value):
    m, data = fixtures(monkeypatch)
    data[file][key] = value
    with pytest.raises(ValueError):
        m.compose(Path("recovered.json"))


def test_projected_dose_candidate_changes_only_internal_learning_rate(monkeypatch):
    m = script(monkeypatch)
    base = asdict(m.load_config(Path("configs/visual-wide-projected-v4.json")))
    candidate = asdict(m.load_config(Path("configs/visual-wide-projected-fast-v6.json")))
    assert m.validate_one_factor(base, candidate, "learning_rate") == (0.02, 0.2)
    candidate["rewards"]["new_tile"] *= 2
    with pytest.raises(ValueError, match="ONLY"):
        m.validate_one_factor(base, candidate, "learning_rate")


def test_fixed_reference_slow_candidate_only_changes_update_rate(monkeypatch):
    m = script(monkeypatch)
    base = asdict(m.load_config(Path('configs/visual-wide-anchored-v9.json')))
    candidate = asdict(m.load_config(Path('configs/visual-wide-anchored-slow-v10.json')))
    assert m.validate_one_factor(base, candidate, 'learning_rate') == (.2, .02)
    assert candidate['brain']['intrinsic_calibration'] == base['brain']['intrinsic_calibration']


def fresh_fixtures(monkeypatch):
    m = script(monkeypatch)
    old_config = "configs/visual-wide-projected-v4.json"
    new_config = "configs/visual-wide-projected-fast-v6.json"
    fixed = asdict(m.load_config(Path(new_config)))
    control = {"run": "control", "samples": 64, "weight_updates_this_evaluation": 0}
    learned = {"run": "candidate", "samples": 64, "weight_updates_this_evaluation": 3}
    controls = [{"mode": "frozen", **control}, {"mode": "learn", "run": "old-learning"}]
    data = {
        "fresh.json": {
            "status": "completed", "seed": 401, "steps": 64, "rows": [learned],
            "baseline_report": "base.json", "baseline_report_sha256": "digest",
            "config": new_config, "config_sha256": "digest", "factor": "learning_rate",
            "old_learning_rate": .02, "new_learning_rate": .2,
            "reused_controls_not_new_trials": controls,
            "prefix_verification": {"samples": 64, "all_fields_exact": True, "run": "prefix"},
        },
        "base.json": {"status": "completed", "seed": 401, "steps": 64, "rows": controls,
                      "config": old_config, "config_sha256": "digest"},
    }
    snapshots, trajectories = {}, {}
    for game, mode in (("control", "frozen"), ("candidate", "learn")):
        model = fixed if mode == "learn" else asdict(m.load_config(
            Path("configs/visual-release-wide-v3.json")))
        data[f"{game}/config.json"] = {
            "config": model, "rom_sha1": "rom", "options": {
                "intro": True, "mode": mode, "steps": 64, "seed": 401,
            },
        }
        data[f"{game}/summary.json"] = {"reason": "step_limit"}
        snapshots[f"{game}/latest-checkpoint.json"] = ({
            "base": np.array([1.0]), "weights": np.array([1.0 if mode == "frozen" else 1.1]),
        }, {"directory": f"{game}/final", "rom_sha1": "rom", "experiment": {
            "mode": mode, "sample": 64, "config": model,
        }})
        trajectories[game] = [{
            "sample": i, "action_source": "fly", "action": "up", "buttons": ["up"],
            "spikes_total": 2, "groups": {}, "motor_rates_hz": {}, "telemetry": {},
            "reward": 0, "reward_events": [], "input_window": {},
        } for i in range(1, 65)]
    trajectories["prefix"] = copy.deepcopy(trajectories["control"])
    monkeypatch.setattr(m, "read", lambda path: data[Path(path).as_posix()])
    monkeypatch.setattr(m, "sha256", lambda path: "digest")
    monkeypatch.setattr(m, "measure", lambda path: control if str(path) == "control" else learned)
    monkeypatch.setattr(m, "read_rows", lambda path: trajectories[str(path)])
    monkeypatch.setattr(m, "read_checkpoint", lambda path: snapshots[Path(path).as_posix()])
    return m, data, snapshots, trajectories


def test_completed_candidate_can_supply_next_one_factor_baseline(monkeypatch):
    m, data, _, _ = fresh_fixtures(monkeypatch)
    result = m.compose(Path("fresh.json"))
    assert result["rows"] == [data["base.json"]["rows"][0],
                              {"mode": "learn", **data["fresh.json"]["rows"][0]}]
    assert result["raw_prefix_rechecked"] and result["frozen_control_shared_not_independent"]
    assert not result["new_trials"] and not result["candidate_resume_segments_not_independent"]
    assert result["candidate_final_checkpoint"] == str(Path("candidate/final"))


def anchored_fixtures(monkeypatch):
    m, data, snapshots, trajectories = fresh_fixtures(monkeypatch)
    old = 'configs/visual-wide-homeostatic-v8.json'
    new = 'configs/visual-wide-anchored-v9.json'
    data['base.json']['config'] = old
    data['fresh.json'].update(config=new, factor='release_reference',
                             old_release_reference='sensorimotor-perturb-homeostatic-v5',
                             new_release_reference='sensorimotor-perturb-anchored-v6')
    model = asdict(m.load_config(Path(new)))
    data['candidate/config.json']['config'] = model
    snapshots['candidate/latest-checkpoint.json'][1]['experiment']['config'] = model
    copies = []

    def verify(source, extended):
        copies.append((source, extended))

    checker = importlib.import_module('evaluate_visual_learning_rate')
    monkeypatch.setattr(checker, 'verify_intrinsic_copy', verify)
    monkeypatch.setattr(m, 'verify_intrinsic_copy', verify)
    return m, data, snapshots, copies


def test_reference_composition_requires_verified_unchanged_intrinsic_payload(monkeypatch):
    m, _, _, copies = anchored_fixtures(monkeypatch)
    result = m.compose(Path('fresh.json'))
    assert not result['new_trials'] and result['raw_prefix_rechecked']
    expected = (Path('fly-data/intrinsic-neutral-visual-release-wide-v3.npz'),
                Path('fly-data/intrinsic-neutral-visual-release-wide-anchored-v9.npz'))
    assert copies == [expected, expected]


@pytest.mark.parametrize('changed', ['physical', 'intrinsic'])
def test_reference_composition_does_not_ignore_changed_original_dynamics(monkeypatch, changed):
    m, data, _, _ = anchored_fixtures(monkeypatch)
    if changed == 'physical':
        data['control/config.json']['config']['frames'] += 1
    else:
        def reject_copy(*args):
            raise ValueError('Original intrinsic payload changed')

        monkeypatch.setattr(m, 'verify_intrinsic_copy', reject_copy)
    with pytest.raises(ValueError):
        m.compose(Path('fresh.json'))


@pytest.mark.parametrize("change", ["partial", "factor", "weights", "resume", "prefix",
                                    "missing", "human", "physical", "checkpoint"])
def test_fresh_composition_rejects_altered_protocol_or_evidence(monkeypatch, change):
    m, data, snapshots, trajectories = fresh_fixtures(monkeypatch)
    if change == "partial":
        data["fresh.json"]["status"] = "running"
    elif change == "factor":
        data["fresh.json"]["new_learning_rate"] = .3
    elif change == "weights":
        snapshots["control/latest-checkpoint.json"][0]["weights"][0] = 1.1
    elif change == "resume":
        data["candidate/config.json"]["options"]["resume"] = "stage-reset"
    elif change == "prefix":
        trajectories["prefix"][0]["action"] = "down"
    elif change == "missing":
        trajectories["candidate"].pop()
    elif change == "human":
        trajectories["candidate"][0]["action_source"] = "human"
    elif change == "physical":
        data["control/config.json"]["config"]["frames"] += 1
    elif change == "checkpoint":
        snapshots["candidate/latest-checkpoint.json"][1]["experiment"]["sample"] = 63
    with pytest.raises((ValueError, AssertionError)):
        m.compose(Path("fresh.json"))
