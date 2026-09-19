import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return (importlib.import_module("probe_visual_curve"),
            importlib.import_module("resume_visual_acquisition"))


@pytest.mark.parametrize('reward,delay', [(1.0, 0), (0.05, 0), (1.0, 3), (0.05, 20)])
@pytest.mark.parametrize('interrupted_arm', ['paired', 'unpaired_within_cue'])
def test_interrupted_control_resumes_same_history_and_neural_arrays(
    tmp_path, monkeypatch, reward, delay, interrupted_arm,
):
    original, recovery = modules(monkeypatch)
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}")

    class FakeBrain:
        def __init__(self, **kwargs):
            self.weights, self.counter = np.array([0.0]), 0

        def snapshot(self):
            return {"weights": self.weights.copy(), "counter": np.array(self.counter)}, {
                "plasticity": {}
            }

        def restore(self, arrays, state, *, weights_only=False):
            self.weights = arrays["weights"].copy()
            self.counter = 0 if weights_only else int(arrays["counter"])

        def observe(self, image):
            self.counter += 1

        def choose(self, observed):
            return ("up", "left", "right", "down")[(self.counter + int(self.weights[0])) % 4], {}

        def reinforce(self, reward):
            self.weights += reward

    def warmup(brain, seed):
        brain.counter = seed % 7

    def test_choices(brain, **kwargs):
        brain.counter += 13  # Probes must not leak this into continuing training.
        return [{"cue": cue, "actions": {cue: 2}, "n": 2} for cue in ("left", "right")]

    for module, directory in ((original, source), (recovery, output)):
        monkeypatch.setattr(module, "InternalBrain", FakeBrain)
        monkeypatch.setattr(module, "warmup", warmup)
        monkeypatch.setattr(module, "test_choices", test_choices)
        monkeypatch.setattr(module, "run_directory", lambda name, value=directory: value)
    monkeypatch.setattr(sys, "argv", ["probe_visual_curve.py", "--config", str(config),
                                     "--seed", "5", "--checkpoints", "32", "64",
                                     '--correct-reward', str(reward),
                                     '--reward-delay-decisions', str(delay)])
    original.main()
    expected = {arm: json.loads((source / f'{arm}-training.json').read_text())
                for arm in ('paired', 'unpaired_within_cue')}
    report = json.loads((source / "report.json").read_text())
    report["status"] = "running"
    report["rows"] = [r for r in report["rows"]
                      if (r['arm'] == interrupted_arm and r['training_decisions'] == 32)
                      or (interrupted_arm == 'unpaired_within_cue' and r['arm'] == 'paired')]
    (source / "report.json").write_text(json.dumps(report))
    (source / f'{interrupted_arm}-training.json').write_text(
        json.dumps(expected[interrupted_arm][:32]),
    )
    monkeypatch.setattr(sys, "argv", ["resume_visual_acquisition.py", "--source", str(source)])
    recovery.main()
    for arm in ("paired", "unpaired_within_cue"):
        assert json.loads((output / f'{arm}-training.json').read_text()) == expected[arm]
        with np.load(source / f"{arm}-64.npz") as a, np.load(output / f"{arm}-64.npz") as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    final = json.loads((output / "report.json").read_text())
    assert final["status"] == "completed"
    assert final['correct_reward'] == reward
    assert final['reward_delay_decisions'] == delay
    assert final["recovery"]["copied_stages_are_not_new_trials"]
    assert len(final["rows"]) == 4
    assert json.loads((source / "report.json").read_text()) == report


def test_resume_rejects_reversal_completed_sources_and_missing_stages(monkeypatch):
    _, module = modules(monkeypatch)
    valid = {"status": "running", "checkpoints": [32, 64], "rows": []}
    module.validate_source(valid)
    for changes in ({"status": "completed"}, {"previous_report": "other"},
                    {"reverse_mapping": True},
                    {"rows": [{"arm": "paired", "training_decisions": 64}]},
                    {"rows": [{"arm": "unpaired_within_cue", "training_decisions": 32}]}):
        with pytest.raises(ValueError):
            module.validate_source({**valid, **changes})


def test_shuffled_schedule_preserves_rewards_per_interval_and_cue(monkeypatch):
    _, module = modules(monkeypatch)
    history = [{"cue": ("left", "right")[(i // 16) % 2], "reward": float(i % 3 == 0)}
               for i in range(64)]
    schedule = module.shuffled_schedule(history, [32, 64], ("left", "right"), 5)
    for start in (0, 32):
        for cue in ("left", "right"):
            indices = [i for i in range(start, start + 32) if history[i]["cue"] == cue]
            assert sorted(schedule[indices]) == sorted(history[i]["reward"] for i in indices)
    with pytest.raises(ValueError, match="Complete paired history"):
        module.shuffled_schedule(history[:32], [32, 64], ("left", "right"), 5)
