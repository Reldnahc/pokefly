import copy
import importlib
import json
from collections import Counter
from pathlib import Path

import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("summarize_visual_retention")


def panel(m, left_correct, right_correct, reverse=False):
    rows = {}
    for arm in m.ARMS:
        raw = []
        correct = (left_correct, right_correct) if arm == "paired" else (50, 50)
        for i, cue in enumerate(("left", "right")):
            target = ("left", "right")[i ^ int(reverse)]
            other = "right" if target == "left" else "left"
            raw.append({"cue": cue, "actions": [target] * correct[i] + [other] * (100-correct[i])})
        rows[arm] = {
            "raw": raw,
            "overall": m.score([
                {"cue": r["cue"], "actions": dict(Counter(r["actions"]))} for r in raw
            ], reverse),
        }
    return rows


def test_both_cues_and_both_controls_must_pass(monkeypatch):
    m = module(monkeypatch)
    assert m.evaluate_panel(panel(m, 65, 65))["passed"]
    assert not m.evaluate_panel(panel(m, 99, 51))["passed"]
    assert not m.evaluate_panel(panel(m, 59, 59))["passed"]
    rows = panel(m, 65, 65)
    rows["unpaired_within_cue"] = copy.deepcopy(rows["paired"])
    assert not m.evaluate_panel(rows)["passed"]


def test_reversal_scores_the_actual_reversed_target(monkeypatch):
    m = module(monkeypatch)
    assert m.evaluate_panel(panel(m, 70, 80, reverse=True), reverse=True)["passed"]


def test_all_raw_choices_and_controls_are_required(monkeypatch):
    m = module(monkeypatch)
    rows = panel(m, 65, 65)
    rows["paired"]["raw"][0]["actions"][0] = "right"
    with pytest.raises(ValueError, match="full raw choices"):
        m.evaluate_panel(rows)
    del rows["unpaired_within_cue"]
    with pytest.raises(ValueError, match="All original"):
        m.evaluate_panel(rows)


def artifact(m, tmp_path, training_seed):
    # Synthetic file/protocol fixture only; never evidence about neural learning.
    curve, output = tmp_path / f"curve-{training_seed}", tmp_path / f"test-{training_seed}"
    curve.mkdir()
    output.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}")
    (curve / "report.json").write_text(json.dumps({
        "status": "completed", "seed": training_seed, "checkpoints": [32],
        "config": str(config), "reverse_mapping": False,
    }))
    seeds = [801, 802, 803, 804]
    records = []
    for arm in sorted(m.ARMS):
        correct = 88 if arm == "paired" else 64
        raw = [{"seed": seed, "cue": cue, "actions": [cue] * correct
                + ["right" if cue == "left" else "left"] * (128 - correct)}
               for seed in seeds for cue in ("left", "right")]
        state_hash = None
        if arm != "original":
            state_file = curve / f"{arm}-32.npz"
            state_file.write_bytes(b"fixture, not neural data")
            state_hash = m.sha256(state_file)
        records.append({
            "arm": arm, "raw": raw, "state_sha256": state_hash,
            "overall": m.score([
                {"cue": r["cue"], "actions": dict(Counter(r["actions"]))} for r in raw
            ], False),
        })
    (output / "report.json").write_text(json.dumps({
        "status": "completed", "curve": str(curve), "seeds": seeds,
        "checkpoint": 32, "neutral_warmup": 128, "test_decisions_per_cue": 128,
        "arms": sorted(m.ARMS), "rows": records,
    }))
    return output


def test_provenance_audit_requires_two_independent_complete_sources(tmp_path, monkeypatch):
    m = module(monkeypatch)
    a, b = artifact(m, tmp_path, 501), artifact(m, tmp_path, 601)
    assert not m.audit_reports([a])["passed"]
    assert m.audit_reports([a, b])["passed"]
    with pytest.raises(ValueError, match="Duplicate"):
        m.audit_reports([a, a])
    report = json.loads((b / "report.json").read_text())
    report["status"] = "running"
    (b / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="completed status"):
        m.audit_reports([a, b])


def test_provenance_audit_rejects_unregistered_weights(tmp_path, monkeypatch):
    m = module(monkeypatch)
    path = artifact(m, tmp_path, 501)
    report = json.loads((path / "report.json").read_text())
    next(row for row in report["rows"] if row["arm"] == "paired")["state_sha256"] = "wrong"
    (path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="registered final neural state"):
        m.audit_reports([path])


@pytest.mark.parametrize('key,value', [
    ('correct_reward', 0.05), ('incorrect_reward', -0.05), ('reward_delay_decisions', 8),
])
def test_different_assay_feedback_cannot_be_pooled(tmp_path, monkeypatch, key, value):
    m = module(monkeypatch)
    a, b = artifact(m, tmp_path, 501), artifact(m, tmp_path, 601)
    report = tmp_path / "curve-601" / "report.json"
    source = json.loads(report.read_text())
    source[key] = value
    report.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="different circuits or retention protocols"):
        m.audit_reports([a, b])


def test_explicit_historical_assay_rewards_match_legacy_defaults(tmp_path, monkeypatch):
    m = module(monkeypatch)
    a, b = artifact(m, tmp_path, 501), artifact(m, tmp_path, 601)
    report = tmp_path / "curve-601" / "report.json"
    source = json.loads(report.read_text())
    source.update(correct_reward=1.0, incorrect_reward=0.0, reward_delay_decisions=0)
    report.write_text(json.dumps(source))
    result = m.audit_reports([a, b])
    assert result["passed"]
    assert result["assay_rewards"] == {"correct": 1.0, "incorrect": 0.0}
    assert result['reward_delay_decisions'] == 0


@pytest.mark.parametrize('delay', [-1, True, 1.5, 32, 1000])
def test_audit_rejects_invalid_feedback_delay(tmp_path, monkeypatch, delay):
    m = module(monkeypatch)
    a = artifact(m, tmp_path, 501)
    path = tmp_path / 'curve-501' / 'report.json'
    source = json.loads(path.read_text())
    source['reward_delay_decisions'] = delay
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='Feedback delay'):
        m.audit_reports([a])


@pytest.mark.parametrize("reward", [0.0, -0.05, float("nan"), float("inf")])
def test_audit_rejects_invalid_correct_reward(tmp_path, monkeypatch, reward):
    m = module(monkeypatch)
    a = artifact(m, tmp_path, 501)
    report = tmp_path / "curve-501" / "report.json"
    source = json.loads(report.read_text())
    source["correct_reward"] = reward
    report.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="assay reward strengths"):
        m.audit_reports([a])
