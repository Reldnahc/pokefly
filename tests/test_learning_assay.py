import importlib.util
from pathlib import Path


def load_assay():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_learning_choices.py"
    spec = importlib.util.spec_from_file_location("learning_assay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_more_left_right_activity_without_cue_discrimination_fails_screen():
    assay = load_assay()
    pre = {"target_rate": 0.1, "conditional_accuracy": 0.5}
    after = {"target_rate": 0.4, "conditional_accuracy": 0.5}
    retention = [{"cue": "left", "n": 10, "actions": {"left": 4, "right": 4, "wait": 2}}]
    paired = {
        "pre_score": pre,
        "stages": [{"score": after, "retention": retention}, {"score": after}],
    }
    unpaired = {"stages": [{"score": pre}, {"score": pre}]}
    result = assay.compare_arms(paired, unpaired, False)
    assert result["target_rate_acquisition_minus_pre"] > 0.05
    assert result["conditional_accuracy_acquisition_minus_pre"] == 0
    assert not result["screen_pass"]


def test_raw_brightness_to_up_down_scoring_and_counterbalancing():
    assay = load_assay()
    rows = [
        {"cue": "black", "n": 10, "actions": {"up+a": 6, "down": 2, "wait": 2}},
        {"cue": "white", "n": 10, "actions": {"down+b": 6, "up": 2, "wait": 2}},
    ]
    options = {"buttons": ("up", "down"), "cues": ("black", "white")}
    assert assay.score(rows, False, **options)["conditional_accuracy"] == 0.75
    assert assay.score(rows, True, **options)["conditional_accuracy"] == 0.25
    assert assay.score(rows, False, **options)["target_rate"] == 0.6
