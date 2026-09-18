"""Recovery protocol fakes are software checks, not evidence of neural learning."""

import copy
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("resume_game_series")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_report():
    return {
        "status": "running", "training_seeds": [7, 8], "eval_seeds": [9, 10],
        "initial_actual_game_source": {"source_launch_seed": 6}, "rows": [],
        "steps_per_attempt": 32, "evaluation_steps_per_arm": 48,
    }


def test_series_recovery_rejects_schedule_changes_and_nonprefix(monkeypatch):
    m = script(monkeypatch)
    valid = source_report()
    m.validate_schedule(valid)
    for updates in (
        {"status": "completed"}, {"recovery": {"source": "old"}},
        {"eval_seeds": [7, 9]}, {"eval_seeds": [6, 9]},
        {"training_seeds": [7, 7]},
        {"rows": [{"phase": "training", "seed": 8, "samples": 32}]},
        {"rows": [{"phase": "training", "seed": 7, "samples": 24}]},
        {"rows": [{"phase": "original", "seed": 9, "samples": 32}]},
    ):
        with pytest.raises(ValueError):
            m.validate_schedule({**valid, **updates})


def test_interrupted_attempt_requires_same_weights_seed_model_and_budget(monkeypatch, tmp_path):
    m = script(monkeypatch)
    report, fixed = source_report(), asdict(m.ExperimentConfig())
    expected = tmp_path / "previous"
    game = {
        "options": {"mode": "learn", "intro": True, "seed": 7, "steps": 32,
                    "weights": str(expected)}, "config": {}, "rom_sha1": "test-rom",
    }
    saved = {"experiment": {"mode": "learn", "sample": 24, "config": {}},
             "rom_sha1": "test-rom"}
    m.validate_interrupted(report, game, saved, fixed, expected)
    for updates in ({"weights": str(tmp_path / "wrong")}, {"seed": 8}, {"steps": 64},
                    {"intro": False}, {"resume": "another"}, {"mode": "frozen"}):
        wrong = copy.deepcopy(game)
        wrong["options"].update(updates)
        with pytest.raises(ValueError, match="registered series"):
            m.validate_interrupted(report, wrong, saved, fixed, expected)
    wrong = copy.deepcopy(saved)
    wrong["experiment"]["config"] = {"frames": 48}
    with pytest.raises(ValueError, match="registered series"):
        m.validate_interrupted(report, game, wrong, fixed, expected)


def test_recovery_restores_full_attempt_then_carries_weights_and_freezes_evaluation(
    monkeypatch, tmp_path,
):
    m = script(monkeypatch)
    source, output = tmp_path / "source", tmp_path / "output"
    output.mkdir()
    initial, interrupted = tmp_path / "initial", tmp_path / "interrupted"
    initial_checkpoint, checkpoint = initial / "checkpoint", interrupted / "checkpoint"
    for path in (initial_checkpoint, checkpoint):
        path.mkdir(parents=True)
        (path / "brain.npz").write_bytes(b"fixture, not a neural brain")
    write(initial / "config.json", {})
    config = source / "config.json"
    write(config, {})
    report = source_report()
    report.update(config=str(config), config_sha256=m.sha256(config))
    report["initial_actual_game_source"].update({
        "run": str(initial), "checkpoint": str(initial_checkpoint),
        "brain_sha256": m.sha256(initial_checkpoint / "brain.npz"),
        "config_sha256": m.sha256(initial / "config.json"), "completed_samples": 32,
        "rom_sha1": "test-rom",
    })
    write(source / "report.json", report)
    write(interrupted / "config.json", {
        "options": {"mode": "learn", "intro": True, "seed": 7, "steps": 32,
                    "weights": str(initial_checkpoint)}, "config": {}, "rom_sha1": "test-rom",
    })
    (interrupted / "trajectory.jsonl").write_text(
        "\n".join(json.dumps({"sample": i}) for i in range(1, 29))
    )
    snapshots = {}

    def state(path, sample):
        return {"directory": str(path), "rom_sha1": "test-rom",
                "experiment": {"mode": "learn", "sample": sample, "config": {}}}

    snapshots[str(initial_checkpoint)] = ({"weights": np.array([2.0])},
                                          state(initial_checkpoint, 32))
    snapshots[str(interrupted / "latest-checkpoint.json")] = (
        {"weights": np.array([3.0])}, state(checkpoint, 24)
    )
    calls = []

    def train(options):
        calls.append(options)
        run = output / f"run-{len(calls)}"
        run.mkdir()
        final = run / "checkpoint"
        final.mkdir()
        (final / "brain.npz").write_bytes(b"fixture final neural weights")
        total = 32 if options.mode == "learn" else options.steps
        start = 25 if options.resume else 1
        (run / "trajectory.jsonl").write_text(
            "\n".join(json.dumps({"sample": i}) for i in range(start, total + 1))
        )
        write(run / "summary.json", {"reason": "step_limit", "samples": total})
        weight = 4.0 if options.resume else (
            float(snapshots[str(options.weights)][0]["weights"][0]) if options.weights else 1.0
        )
        if options.mode == "learn":
            weight += 1
        entry = ({"weights": np.array([weight]), "base": np.array([1.0])}, state(final, total))
        snapshots[str(run / "latest-checkpoint.json")] = entry
        snapshots[str(final)] = entry
        return run

    monkeypatch.setattr(m, "read_checkpoint", lambda path: snapshots[str(path)])
    monkeypatch.setattr(m, "train", train)
    monkeypatch.setattr(m, "run_directory", lambda name: output)
    monkeypatch.setattr(m, "resolve_rom", lambda *args: Path("ignored.gb"))
    monkeypatch.setattr(m, "measure_complete", lambda path, summary, rows: {
        "samples": len(rows), "run": str(path),
    })
    monkeypatch.setattr(m, "measure", lambda path: {
        "samples": json.loads((path / "summary.json").read_text())["samples"],
        "run": str(path), "weight_updates_this_evaluation": 0,
    })
    monkeypatch.setattr(sys, "argv", ["resume_game_series.py", "--source", str(source),
                                     "--interrupted-run", str(interrupted)])
    m.main()
    assert len(calls) == 6
    assert calls[0].resume == checkpoint and calls[0].steps == 8 and not calls[0].intro
    assert calls[0].weights is None and calls[0].config is None
    assert calls[1].intro and calls[1].steps == 32 and calls[1].resume is None
    assert calls[1].weights == output / "run-1" / "checkpoint"
    assert [o.seed for o in calls] == [7, 8, 9, 9, 10, 10]
    assert all(o.mode == "frozen" and o.intro and o.steps == 48 for o in calls[2:])
    final_checkpoint = output / "run-2" / "checkpoint"
    assert [o.weights for o in calls[2:]] == [None, final_checkpoint, final_checkpoint, None]
    completed = json.loads((output / "report.json").read_text())
    assert completed["status"] == "completed"
    assert completed["recovery"]["overlap_decisions_exact"] == 4
    assert completed["rows"][0]["samples"] == 32
    assert json.loads((source / "report.json").read_text()) == report
