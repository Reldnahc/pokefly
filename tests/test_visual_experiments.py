"""Protocol tests use tiny fakes, not synthetic evidence of neural learning."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def script(monkeypatch, name):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module(name)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_same_start_reversal_restores_acquired_paired_state_for_both_arms(tmp_path, monkeypatch):
    module = script(monkeypatch, "probe_visual_curve")
    source, output = tmp_path / "source", tmp_path / "output"
    output.mkdir()
    config = tmp_path / "config.json"
    write(config, {})
    records = [{"cue": cue, "actions": {cue: 2}, "n": 2} for cue in ("left", "right")]
    write(
        source / "report.json",
        {
            "seed": 5,
            "checkpoints": [32],
            "reverse_mapping": False,
            "pretest": records,
            "status": "completed",
            "rows": [
                {"arm": arm, "training_decisions": 32, "retention": records}
                for arm in ("paired", "unpaired_within_cue")
            ],
        },
    )
    for arm, weight in (("paired", 2), ("unpaired_within_cue", 9)):
        np.savez(source / f"{arm}-32.npz", weights=np.array([weight]))
        write(source / f"{arm}-32.json", {"plasticity": {}})
        write(
            source / f"{arm}-training.json",
            [
                {"cue": ("left", "right")[(i // 16) % 2], "reward": float(i % 2), "action": "left"}
                for i in range(32)
            ],
        )
    restored = []

    class FakeBrain:
        def __init__(self, **kwargs):
            self.weights = np.array([0])

        def snapshot(self):
            return {"weights": self.weights.copy()}, {"plasticity": {}}

        def restore(self, arrays, state, **kwargs):
            restored.append(int(arrays["weights"][0]))
            self.weights = arrays["weights"].copy()

        def observe(self, frame):
            return None

        def choose(self, observation):
            return "left", {}

        def reinforce(self, reward):
            self.weights += int(reward)

    monkeypatch.setattr(module, "InternalBrain", FakeBrain)
    monkeypatch.setattr(module, "test_choices", lambda *a, **kw: records)
    monkeypatch.setattr(module, "run_directory", lambda name: output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_visual_curve.py",
            "--config",
            str(config),
            "--seed",
            "5",
            "--continue-from",
            str(source),
            "--allow-reversal",
            "--same-start-reversal",
            "--reverse",
            "--checkpoints",
            "64",
        ],
    )
    module.main()
    assert restored[0] == restored[2] == 2  # Both arms, not control's old weight 9.
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["same_acquired_start_for_both_reversal_arms"]
    assert (
        report["pre_reversal_scores"]["paired"]
        == report["pre_reversal_scores"]["unpaired_within_cue"]
    )
    paired = json.loads((output / "paired-training.json").read_text())
    shuffled = json.loads((output / "unpaired_within_cue-training.json").read_text())
    assert paired[:32] == shuffled[:32]
    for cue in ("left", "right"):
        assert sorted(r["reward"] for r in paired[32:] if r["cue"] == cue) == sorted(
            r["reward"] for r in shuffled[32:] if r["cue"] == cue
        )


def test_game_continuation_restores_both_final_modes_without_intro(tmp_path, monkeypatch):
    module = script(monkeypatch, "evaluate_visual_gameplay")
    source, output = tmp_path / "source", tmp_path / "output"
    output.mkdir()
    config = source / "fixed-config.json"
    write(config, {})
    rows = []
    for mode in ("frozen", "learn"):
        run = source / mode
        write(run / "summary.json", {"reason": "step_limit", "samples": 6000})
        rows.append({"run": str(run), "mode": mode, "first_starter": 4000})
    write(
        source / "report.json",
        {
            "seed": 401,
            "status": "completed",
            "config": str(config),
            "config_sha256": module.sha256(config),
            "rows": rows,
        },
    )
    calls = []

    def checkpoint(path):
        mode = path.parent.name
        return {}, {
            "directory": str(path.parent / "checkpoint"),
            "experiment": {"mode": mode, "sample": 6000},
        }

    def train(options):
        calls.append(options)
        run = output / options.mode
        write(run / "summary.json", {"reason": "step_limit"})
        (run / "trajectory.jsonl").write_text("")
        return run

    monkeypatch.setattr(module, "read_checkpoint", checkpoint)
    monkeypatch.setattr(module, "run_directory", lambda name: output)
    monkeypatch.setattr(module, "resolve_rom", lambda *args: Path("ignored.gb"))
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(
        module,
        "measure",
        lambda path: {
            "samples": 18000,
            "weight_updates_this_evaluation": 0,
            "first_starter": None,
            "first_party_count": None,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_visual_gameplay.py",
            "--seed",
            "401",
            "--steps",
            "18000",
            "--continue-from",
            str(source),
        ],
    )
    module.main()
    assert [o.mode for o in calls] == ["frozen", "learn"]
    assert all(not o.intro and o.config is None and o.weights is None for o in calls)
    assert all(o.resume == source / o.mode / "checkpoint" for o in calls)
    report = json.loads((output / "report.json").read_text())
    assert all(row["total_samples"] == 24000 for row in report["rows"])
    assert all(row["first_starter"] == 4000 for row in report["rows"])


def test_same_start_reversal_requires_explicit_continuation(monkeypatch):
    module = script(monkeypatch, "probe_visual_curve")
    monkeypatch.setattr(sys, "argv", ["probe_visual_curve.py", "--same-start-reversal"])
    with pytest.raises(SystemExit):
        module.main()


def test_game_continuation_cannot_silently_change_its_visual_model(monkeypatch):
    module = script(monkeypatch, "evaluate_visual_gameplay")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_visual_gameplay.py",
            "--seed",
            "401",
            "--config",
            "configs/visual-release-wide-v3.json",
            "--continue-from",
            "previous",
        ],
    )
    with pytest.raises(SystemExit):
        module.main()


def test_learning_rate_candidate_cannot_change_rewards_or_firing(monkeypatch):
    from dataclasses import asdict

    module = script(monkeypatch, "evaluate_visual_learning_rate")
    original = asdict(module.load_config(Path("configs/visual-rate-v1.json")))
    candidate = asdict(module.load_config(Path("configs/visual-fast-learning-v2.json")))
    assert module.validate_one_factor(original, candidate) == (0.02, 0.2)
    candidate["rewards"]["new_tile"] *= 2
    with pytest.raises(ValueError, match="ONLY"):
        module.validate_one_factor(original, candidate)
    candidate = asdict(module.load_config(Path("configs/visual-fast-learning-v2.json")))
    candidate["brain"]["noise_amplitude"] *= 2
    with pytest.raises(ValueError, match="ONLY"):
        module.validate_one_factor(original, candidate)


@pytest.mark.parametrize(
    "baseline,candidate_name",
    [
        ("visual-rate-v1", "visual-projected-learning-v4"),
        ("visual-release-wide-v3", "visual-wide-projected-v4"),
    ],
)
def test_projected_visual_profile_changes_only_internal_credit_rule(baseline, candidate_name):
    from dataclasses import asdict

    from pokefly.experiment import load_config

    original = asdict(load_config(Path(f"configs/{baseline}.json")))
    candidate = asdict(load_config(Path(f"configs/{candidate_name}.json")))
    assert candidate["brain"]["plasticity"]["rule"] == "sensorimotor-perturb-projected-v4"
    candidate["brain"]["plasticity"]["rule"] = original["brain"]["plasticity"]["rule"]
    assert candidate == original


def test_reversal_retention_cannot_silently_use_naive_or_different_start_controls(monkeypatch):
    module = script(monkeypatch, "probe_visual_retention")
    source = {"pre_reversal_scores": {}}
    with pytest.raises(ValueError, match="acquired neural state"):
        module.validate_reversal_controls(source, ["original"], None, None)
    with pytest.raises(ValueError, match="same acquired start"):
        module.validate_reversal_controls(source, ["unpaired_within_cue"], None, None)
    module.validate_reversal_controls(source, ["paired"], None, None)
    source["same_acquired_start_for_both_reversal_arms"] = True
    module.validate_reversal_controls(
        source, ["original", "unpaired_within_cue"], Path("acquired.npz"), None
    )


def test_series_initial_source_must_be_a_final_actual_game_checkpoint(tmp_path, monkeypatch):
    module = script(monkeypatch, "train_game_series")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "brain.npz").write_bytes(b"fake checkpoint data for protocol test")
    write(tmp_path / "config.json", {"options": {"mode": "learn", "seed": 401}})
    write(tmp_path / "summary.json", {"reason": "step_limit", "samples": 6000})
    saved = {
        "directory": str(checkpoint),
        "rom_sha1": "not-a-real-rom",
        "experiment": {"mode": "learn", "sample": 6000, "config": {"brain": {}}},
    }
    monkeypatch.setattr(module, "read_checkpoint", lambda path: ({}, saved))
    pinned, config, provenance = module.completed_game_source(tmp_path)
    assert pinned == checkpoint
    assert config == saved["experiment"]["config"]
    assert provenance["completed_samples"] == 6000
    saved["experiment"]["sample"] = 5500
    with pytest.raises(ValueError, match="final completed"):
        module.completed_game_source(tmp_path)
    write(tmp_path / "config.json", {"options": {"mode": "frozen", "seed": 401}})
    with pytest.raises(ValueError, match="actual-game learning"):
        module.completed_game_source(tmp_path)


def test_retained_game_intro_pins_final_weights_and_freezes_all_arms(tmp_path, monkeypatch):
    module = script(monkeypatch, "evaluate_saved_gameplay")
    source, output, checkpoint = tmp_path / "source", tmp_path / "out", tmp_path / "saved"
    output.mkdir()
    checkpoint.mkdir()
    (checkpoint / "brain.npz").write_bytes(b"fake protocol fixture, not neural evidence")
    write(source / "config.json", {"options": {"mode": "learn", "seed": 402}})
    write(source / "summary.json", {"reason": "step_limit", "samples": 24000})
    saved = {
        "directory": str(checkpoint),
        "experiment": {"mode": "learn", "sample": 24000, "config": {"brain": {}}},
    }
    calls = []

    def train(options):
        calls.append(options)
        path = output / f"arm-{len(calls)}"
        write(path / "summary.json", {"reason": "step_limit"})
        return path

    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(module, "measure", lambda path: {"weight_updates_this_evaluation": 0})
    monkeypatch.setattr(module, "read_checkpoint", lambda path: ({}, saved))
    monkeypatch.setattr(module, "run_directory", lambda name: output)
    monkeypatch.setattr(module, "resolve_rom", lambda *a: Path("ignored.gb"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_saved_gameplay.py",
            "--source-run",
            str(source),
            "--intro",
            "--seeds",
            "1501",
            "1502",
            "--steps",
            "6000",
        ],
    )
    module.main()
    assert [o.weights for o in calls] == [None, checkpoint, checkpoint, None]
    assert [o.seed for o in calls] == [1501, 1501, 1502, 1502]
    assert all(o.mode == "frozen" and o.intro and o.load_state is None for o in calls)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed" and report["intro_intervention"]
    assert report["source_completed_sample"] == 24000
    saved["experiment"]["sample"] = 23500
    with pytest.raises(ValueError, match="final checkpoint"):
        module.main()
