"""Fabricated tiny games validate evidence accounting, not fly performance."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from test_game_series_audit import panels as series_panels
from test_game_series_audit import write


def panels(tmp_path, monkeypatch):
    _, series, snapshots = series_panels(tmp_path, monkeypatch)
    m = importlib.import_module("summarize_game_retention_panel")
    fixed = m.asdict(m.load_config(series[0] / "fixed.json"))
    sources = []
    for index, seed in enumerate((401, 402)):
        checkpoint = series[index] / "trained"
        provenance = {"run": str(series[index]), "source_launch_seed": seed, "rom_sha1": "fake",
                      "brain_sha256": m.sha256(checkpoint / "brain.npz")}
        sources.append((checkpoint, fixed, provenance))
    paths = []
    for seed in (1601, 1602):
        path = tmp_path / f"panel-{seed}"
        paths.append(path)
        config = path / "fixed.json"
        write(config, fixed)
        report = {
            "status": "completed", "intro_intervention": True,
            "one_shared_original_per_seed": True, "seeds": [seed], "steps_per_arm": 8,
            "config": str(config), "config_sha256": m.sha256(config),
            "sources": [s[2] for s in sources], "rows": [],
        }
        for arm, base in (("original", series[0]), ("retained_401", series[0]),
                          ("retained_402", series[1])):
            phase = "original" if arm == "original" else "retained"
            game = base / f"{phase}-{seed}"
            stored = m.read(game / "config.json")
            stored["config"] = fixed
            write(game / "config.json", stored)
            (game / "brain.npz").write_bytes(b"fake frozen checkpoint")
            snapshots[str(game / "latest-checkpoint.json")][1]["directory"] = str(game)
            report["rows"].append({
                "arm": arm, "seed": seed, **m.measure(game),
                "final_checkpoint": str(game), "final_brain_sha256": m.sha256(game / "brain.npz"),
                "start_state_sha256": m.sha256(game / "start.state"),
            })
        write(path / "report.json", report)
    monkeypatch.setattr(m, "read_checkpoint", lambda path: snapshots[str(path)])
    monkeypatch.setattr(m, "completed_game_source",
                        lambda path: sources[[s[2]["run"] for s in sources].index(str(path))])
    return m, paths, snapshots, sources


def test_combined_panel_counts_each_original_once(tmp_path, monkeypatch):
    m, paths, _, _ = panels(tmp_path, monkeypatch)
    result = m.audit(paths)
    assert result["seeds"] == [1601, 1602] and result["shared_original_controls"] == 2
    assert len(result["comparisons"]) == 4
    assert result["aggregate"]["original"]["starters"] == 0
    for arm in ("retained_401", "retained_402"):
        assert result["aggregate"][arm]["trials"] == 2
        assert result["aggregate"][arm]["battle_wins"] == 2  # Not four from double-counting rivals.
    assert not result["new_trials"] and not result["statistical_significance_claimed"]


@pytest.mark.parametrize("change", ["partial", "missing", "duplicate", "wrong_start", "source"])
def test_combined_audit_rejects_missing_or_changed_evidence(tmp_path, monkeypatch, change):
    m, paths, _, sources = panels(tmp_path, monkeypatch)
    report = m.read(paths[1] / "report.json")
    if change == "partial":
        with pytest.raises(ValueError, match="partial panel"):
            m.audit(paths[:1])
        return
    if change == "missing":
        report["rows"].pop()
    elif change == "duplicate":
        report["rows"].append(report["rows"][0])
    elif change == "wrong_start":
        game = Path(report["rows"][-1]["run"])
        (game / "start.state").write_bytes(b"different initial game")
        report["rows"][-1]["start_state_sha256"] = m.sha256(game / "start.state")
    elif change == "source":
        sources[1][2]["brain_sha256"] = "changed since testing"
    write(paths[1] / "report.json", report)
    messages = {"missing": "arm is missing", "duplicate": "Duplicate",
                "wrong_start": "different initial game states", "source": "source changed"}
    with pytest.raises(ValueError, match=messages[change]):
        m.audit(paths)


def test_combined_audit_checks_actual_frozen_weights(tmp_path, monkeypatch):
    m, paths, snapshots, _ = panels(tmp_path, monkeypatch)
    game = Path(m.read(paths[0] / "report.json")["rows"][-1]["run"])
    snapshots[str(game / "latest-checkpoint.json")][0]["weights"] = np.array([99.0])
    with pytest.raises(AssertionError):
        m.audit(paths)
