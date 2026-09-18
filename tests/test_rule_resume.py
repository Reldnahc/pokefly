import copy
import importlib
from dataclasses import asdict
from pathlib import Path

import pytest

from pokefly.experiment import ExperimentConfig


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("continue_rule_gameplay")


def test_exact_stitch_counts_each_decision_once_and_excludes_wall_time(module):
    source = [{"sample": i, "action": "up", "run_id": "old", "compute_ms": 1}
              for i in range(1, 5)]
    resumed = [{"sample": i, "action": "up", "run_id": "new", "compute_ms": 8}
               for i in range(3, 7)]
    rows, overlap = module.stitch(source, resumed, 2, 6)
    assert [r["sample"] for r in rows] == list(range(1, 7))
    assert overlap == 2
    resumed[0]["action"] = "down"
    with pytest.raises(ValueError, match="Exact replay"):
        module.stitch(source, resumed, 2, 6)


def test_stitch_rejects_missing_or_duplicate_records(module):
    rows = [{"sample": i} for i in range(1, 5)]
    for prefix, suffix in [(rows[:1], rows[2:]), (rows[:2], rows[3:]),
                           (rows[:2], rows[2:] + rows[3:])]:
        with pytest.raises(ValueError, match="incomplete or duplicated"):
            module.stitch(prefix, suffix, 2, 4)


def test_resume_cannot_change_model_seed_budget_or_skip_control_check(module):
    report = {"status": "running", "rows": [], "seed": 401, "steps": 6000,
              "prefix_verification": {"all_fields_exact": True}}
    game = {"options": {"mode": "learn", "intro": True, "seed": 401, "steps": 6000},
            "config": {}, "rom_sha1": "same"}
    saved = {"experiment": {"mode": "learn", "config": {}, "sample": 1000},
             "rom_sha1": "same"}
    fixed = asdict(ExperimentConfig())
    module.validate_source(report, game, saved, fixed)
    for key, value in [("seed", 999), ("steps", 5000), ("intro", False), ("weights", "other")]:
        bad = copy.deepcopy(game)
        bad["options"][key] = value
        with pytest.raises(ValueError, match="changed"):
            module.validate_source(report, bad, saved, fixed)
    with pytest.raises(ValueError, match="changed"):
        module.validate_source(report, game, saved, {**fixed, "frames": 48})
    report["prefix_verification"] = {}
    with pytest.raises(ValueError, match="verification"):
        module.validate_source(report, game, saved, fixed)
