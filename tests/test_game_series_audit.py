"""Tiny fabricated records test audit integrity, not fly behavior."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("summarize_game_series")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def panels(tmp_path, monkeypatch):
    m = script(monkeypatch)
    snapshots, paths = {}, []
    for source_seed, training in ((401, [1401, 1402]), (402, [1403, 1404])):
        path = tmp_path / str(source_seed)
        paths.append(path)
        config, checkpoint = path / "fixed.json", path / "trained"
        write(config, {})
        checkpoint.mkdir()
        (checkpoint / "brain.npz").write_bytes(b"protocol fixture")
        trained_weights = np.array([2.0 if source_seed == 401 else 3.0])
        snapshots[str(checkpoint)] = ({"weights": trained_weights}, {
            "rom_sha1": "fake", "experiment": {"mode": "learn", "sample": 8, "config": {}},
        })
        report = {
            "status": "completed", "recovery": {"source": "protocol-fixture"},
            "config": str(config), "config_sha256": m.sha256(config),
            "eval_seeds": [1601, 1602], "evaluation_steps_per_arm": 8,
            "initial_actual_game_source": {"source_launch_seed": source_seed},
            "training_seeds": training, "steps_per_attempt": 8,
            "evaluation_source": str(checkpoint),
            "evaluation_brain_sha256": m.sha256(checkpoint / "brain.npz"),
            "rows": [{"phase": "training", "seed": seed, "samples": 8} for seed in training],
        }
        for seed in (1601, 1602):
            for arm in ("original", "retained"):
                game = path / f"{arm}-{seed}"
                retained = arm == "retained"
                telemetry = {"map": 0, "x": 3, "y": 5, "party": int(retained),
                             "levels": 6 if retained else 0, "battle": 0}
                summary = {
                    "samples": 8, "new_samples": 8, "reason": "step_limit",
                    "tiles": 12 if retained else 10, "maps": [0, 37],
                    "reward": 3 if retained else 0,
                    "reward_counts": {"battle_win": 1, "rival_win": 1} if retained else {},
                    "button_counts": {"up": 8}, "first_house_exit": 1, "final_state": telemetry,
                }
                write(game / "summary.json", summary)
                write(game / "config.json", {
                    "config": {}, "rom_sha1": "fake", "options": {
                        "mode": "frozen", "intro": True, "seed": seed, "steps": 8,
                        "weights": str(checkpoint) if retained else None,
                    },
                })
                (game / "start.state").write_bytes(b"same explicitly reset game")
                rows = [{"run_id": str(game), "sample": i, "action": "up", "action_source": "fly",
                         "telemetry": {**telemetry, "battle": int(retained and 2 <= i <= 4)},
                         "reward_events": [
                             {"category": "battle_win", "encounter": 1},
                             {"category": "rival_win", "encounter": 1},
                         ] if retained and i == 4 else [],
                         "learning": {"changed_this_reward": 0}, "compute_ms": source_seed}
                        for i in range(1, 9)]
                (game / "trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
                snapshots[str(game / "latest-checkpoint.json")] = ({
                    "base": np.array([1.0]),
                    "weights": trained_weights if retained else np.array([1.0]),
                }, {"experiment": {"sample": 8, "mode": "frozen"},
                    "rewards": {"counts": summary["reward_counts"], "active": None}})
                report["rows"].append({"phase": arm, "seed": seed, **m.measure(game)})
        write(path / "report.json", report)
    monkeypatch.setattr(m, "read_checkpoint", lambda path: snapshots[str(path)])
    return m, paths, snapshots


def test_full_panels_count_shared_originals_once_and_not_duplicate_rival(tmp_path, monkeypatch):
    m, paths, _ = panels(tmp_path, monkeypatch)
    result = m.audit(paths)
    assert result["independent_training_sources"] == 2
    assert len(result["comparisons"]) == 4 and len(result["shared_original_controls"]) == 2
    assert all(len(r["runs"]) == 2 for r in result["shared_original_controls"].values())
    assert result["new_trials"] is False and not result["statistical_significance_claimed"]
    for row in result["comparisons"]:
        assert row["tile_difference"] == 2
        assert row["retained"]["battle_wins"] == 1  # Not battle+rival=2.
        assert row["retained"]["rival_wins"] == 1


@pytest.mark.parametrize("change", ["missing_arm", "duplicate_arm", "partial", "source_hash"])
def test_panel_rejects_missing_or_changed_evidence(tmp_path, monkeypatch, change):
    m, paths, _ = panels(tmp_path, monkeypatch)
    report = m.read(paths[0] / "report.json")
    if change == "missing_arm":
        report["rows"].pop()
    elif change == "duplicate_arm":
        report["rows"].append(report["rows"][-1])
    elif change == "partial":
        report["status"] = "running"
    else:
        report["evaluation_brain_sha256"] = "changed"
    write(paths[0] / "report.json", report)
    with pytest.raises(ValueError):
        m.audit(paths)


def test_shared_control_checks_full_trajectory_not_just_summary(tmp_path, monkeypatch):
    m, paths, _ = panels(tmp_path, monkeypatch)
    trajectory = paths[1] / "original-1601" / "trajectory.jsonl"
    rows = [json.loads(line) for line in trajectory.read_text().splitlines()]
    rows[0]["input_window"] = {"frame_hashes": ["different causal input"]}
    trajectory.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="not exact replications"):
        m.audit(paths)


def test_frozen_retention_verifies_actual_weights_unchanged(tmp_path, monkeypatch):
    m, paths, snapshots = panels(tmp_path, monkeypatch)
    final = str(paths[0] / "retained-1601" / "latest-checkpoint.json")
    snapshots[final][0]["weights"] = np.array([99.0])
    with pytest.raises(AssertionError):
        m.audit(paths)
