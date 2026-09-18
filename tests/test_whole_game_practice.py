"""Training schedule fixtures; no emulation or claimed neural learning."""

import importlib
import json
import sys
from pathlib import Path

import pytest


def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("train_game_series")


def test_deferred_evaluation_keeps_final_brain_without_claiming_completed_study(
    tmp_path, monkeypatch,
):
    m = script(monkeypatch)
    checkpoint = tmp_path / "pinned"
    checkpoint.mkdir()
    (checkpoint / "brain.npz").write_bytes(b"fixture, not neural weights")
    saved = {"directory": str(checkpoint), "experiment": {"mode": "learn", "sample": 64}}
    monkeypatch.setattr(m, "read_checkpoint", lambda path: ({}, saved))
    report = {"defer_evaluation": True, "steps_per_attempt": 64, "status": "running"}
    assert m.defer_evaluation(report, tmp_path, tmp_path / "latest-checkpoint.json")
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["status"] == "training_completed" and written["evaluation_pending"]
    assert written["evaluation_source"] == str(checkpoint)
    assert written["evaluation_brain_sha256"] == m.sha256(checkpoint / "brain.npz")
    assert not m.defer_evaluation({}, tmp_path, checkpoint)  # Legacy schedule unchanged.
    saved["experiment"]["sample"] = 32
    with pytest.raises(ValueError, match="final completed training"):
        m.defer_evaluation(report, tmp_path, checkpoint)


@pytest.mark.parametrize("training", [[4101], [4101, 4102]])
def test_whole_game_training_carries_each_final_brain_and_never_uses_stage_resets(
    tmp_path, monkeypatch, training,
):
    m = script(monkeypatch)
    output, initial = tmp_path / "output", tmp_path / "initial"
    output.mkdir()
    initial.mkdir()
    snapshots, calls = {}, []
    monkeypatch.setattr(m, "completed_game_source", lambda path: (
        initial, {}, {"source_launch_seed": 401, "checkpoint": str(initial)},
    ))
    monkeypatch.setattr(m, "run_directory", lambda name: output)
    monkeypatch.setattr(m, "resolve_rom", lambda *args: Path("unused.gb"))
    monkeypatch.setattr(m, "read_checkpoint", lambda path: ({}, snapshots[str(path)]))
    monkeypatch.setattr(m, "measure", lambda path: {"run": str(path), "samples": 64})

    def train(options):
        calls.append(options)
        path = output / f"run-{len(calls)}"
        path.mkdir()
        (path / "summary.json").write_text('{"reason":"step_limit"}')
        (path / "brain.npz").write_bytes(b"fixture final synapses")
        snapshots[str(path / "latest-checkpoint.json")] = {
            "directory": str(path), "experiment": {"mode": "learn", "sample": 64},
        }
        return path

    monkeypatch.setattr(m, "train", train)
    monkeypatch.setattr(sys, "argv", ["train_game_series.py", "--initial-game-run", str(initial),
                                     "--training-seeds", *map(str, training), "--eval-seeds",
                                     "3801", "3802", "--steps", "64", "--defer-evaluation"])
    m.main()
    assert len(calls) == len(training)
    assert calls[0].weights == initial
    if len(training) > 1:
        assert calls[1].weights == output / "run-1" / "latest-checkpoint.json"
    assert all(o.mode == "learn" and o.intro and not o.load_state and not o.resume
               and o.config is None for o in calls)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "training_completed" and report["evaluation_pending"]
    assert [r["phase"] for r in report["rows"]] == ["training"] * len(training)
    assert report["eval_seeds"] == [3801, 3802]  # Reserved, not claimed as evaluated.
