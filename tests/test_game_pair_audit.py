import importlib
from pathlib import Path

import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_game_pair")


def test_paid_victory_is_distinguished_from_finished_encounter(monkeypatch):
    m = module(monkeypatch)
    rows = [{"reward_events": [{"category": "new_tile"},
                               {"category": "battle_win", "encounter": 1},
                               {"category": "rival_win", "encounter": 1}]}]
    ledger = {"counts": {"new_tile": 1, "battle_win": 1, "rival_win": 1},
              "active": {"id": 1, "paid": True}}
    result = m.encounter_outcomes(rows, ledger)
    assert result["awarded_outcomes"]["battle_win"] == 1
    assert result["completed_outcomes"]["battle_win"] == 0
    assert result["completed_outcomes"]["rival_win"] == 0
    assert result["paid_but_unfinished_encounter"] == {"id": 1, "paid": True}
    ledger["active"] = None
    result = m.encounter_outcomes(rows, ledger)
    assert result["completed_outcomes"]["battle_win"] == 1  # Not two from rival bonus.
    assert result["paid_but_unfinished_encounter"] is None


def test_pending_capture_does_not_remove_a_previous_win(monkeypatch):
    m = module(monkeypatch)
    rows = [{"reward_events": [{"category": "battle_win", "encounter": 1},
                               {"category": "capture", "encounter": 2}]}]
    ledger = {"counts": {"battle_win": 1, "capture": 1}, "active": {"id": 2, "paid": True}}
    result = m.encounter_outcomes(rows, ledger)
    assert result["completed_outcomes"]["battle_win"] == 1
    assert result["awarded_outcomes"]["capture"] == 1
    assert result["completed_outcomes"]["capture"] == 0
    ledger["active"] = {"id": 3}  # A new unfinished encounter has not earned anything.
    assert m.encounter_outcomes(rows, ledger)["completed_outcomes"]["capture"] == 1


@pytest.mark.parametrize("change", ["missing_event", "paid_without_outcome", "duplicate_outcome"])
def test_encounter_audit_rejects_inconsistent_reward_history(monkeypatch, change):
    m = module(monkeypatch)
    rows = [{"reward_events": [{"category": "battle_win", "encounter": 1}]}]
    ledger = {"counts": {"battle_win": 1}, "active": None}
    if change == "missing_event":
        rows = [{"reward_events": []}]
    else:
        ledger["active"] = {"id": 2 if change == "paid_without_outcome" else 1, "paid": True}
        if change == "duplicate_outcome":
            rows[0]["reward_events"].append({"category": "battle_win", "encounter": 1})
            ledger["counts"]["battle_win"] = 2
    with pytest.raises(ValueError):
        m.encounter_outcomes(rows, ledger)


def test_pair_audit_rejects_incomplete_or_single_seed_reports(tmp_path, monkeypatch):
    m = module(monkeypatch)
    (tmp_path / "report.json").write_text('{"status":"running","rows":[]}')
    with pytest.raises(ValueError, match="Completed fresh-game"):
        m.audit([tmp_path])
    with pytest.raises(ValueError, match="At least two"):
        m.audit([])
