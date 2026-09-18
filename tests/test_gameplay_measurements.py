import importlib.util
import json
from pathlib import Path


def test_uninitialized_nickname_party_is_not_a_completed_starter(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/evaluate_saved_gameplay.py"
    spec = importlib.util.spec_from_file_location("measure_gameplay", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = {
        "new_samples": 3,
        "tiles": 1,
        "maps": [40],
        "reward": 0,
        "reward_counts": {},
        "button_counts": {},
        "final_state": {},
        "first_house_exit": None,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    rows = [
        {
            "sample": i + 1,
            "telemetry": {"map": 40, "battle": 0, "party": 1, "levels": level},
            "learning": {"changed_this_reward": 0},
        }
        for i, level in enumerate([0, 0, 5])
    ]
    log = tmp_path / "trajectory.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows))
    measured = module.measure(tmp_path)
    assert measured["first_party_count"] == 1 and measured["first_starter"] == 3
    log.write_text("\n".join(json.dumps(r) for r in rows[:2]))
    assert module.measure(tmp_path)["first_starter"] is None
