import copy
import importlib
from dataclasses import asdict
from pathlib import Path

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
