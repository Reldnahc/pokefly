import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pokefly.rewards import RewardConfig


@pytest.mark.parametrize("timing,expected", [("encounter-end-v1", 32), ("last-faint-v3", 0)])
@pytest.mark.parametrize("finished", [False, True])
def test_latency_audit_records_early_rewards_without_claiming_unfinished_battles(
    monkeypatch, timing, expected, finished,
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    m = importlib.import_module("audit_outcome_latency")
    memory = np.zeros(65536, np.uint8)
    for address, value in {0xD057: 2, 0xD031: 25, 0xD163: 1, 0xD16D: 20,
                           0xD016: 20, 0xD89C: 1}.items():
        memory[address] = value
    clock = [0]
    rewards = m.TimedRewards(RewardConfig(timing=timing), lambda: clock[0])
    events = [(1, "start"), (10, "faint"), (26, "trainer_win")]
    if finished:
        events.append((42, "end"))
    for sample, event in events:
        rewards.sample, clock[0] = sample, sample * 24
        rewards.event(event, memory)
    result = m.summarize_outcomes(rewards, 12)
    if not finished and timing == "encounter-end-v1":
        assert result == []
        return
    assert len(result) == 1
    assert result[0]["encounter_finished"] == finished
    assert [e["category"] for e in result[0]["events"]] == ["battle_win", "rival_win"]
    faint = next(d for d in result[0]["delays"] if d["from_hook"] == "faint")
    assert faint["decisions"] == expected and faint["neural_seconds"] == expected * .24
    assert all(d["decisions"] >= 0 for d in result[0]["delays"])


def test_move_observer_uses_confirmed_callback_and_never_changes_reward(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    m = importlib.import_module("audit_outcome_latency")
    memory, frame = np.zeros(65536, np.uint8), [0]
    rewards = m.TimedRewards(RewardConfig(), lambda: frame[0])
    game = SimpleNamespace(pyboy=SimpleNamespace(memory=memory))
    observer = m.MoveConfirmationHooks(game, rewards)
    observer.record()
    assert not rewards.history  # Outside an encounter, do not invent one.
    rewards.active = {"id": 1}
    for sample, move, slot in ((7, 33, 0), (8, 45, 1), (25, 33, 0)):
        rewards.sample, frame[0] = sample, sample * 24
        memory[0xCCDC], memory[0xCC2E] = move, slot
        observer.record()
    rewards.outcomes = [{"encounter": 1, "events": [], "delivered_at": {
        "hook": "faint", "sample": 10, "frame": 240,
    }}]
    assert rewards.pending == [] and rewards.total == 0 and rewards.counts == {}
    measured = m.summarize_outcomes(rewards, 12, .6)[0]["last_confirmed_move"]
    assert measured["move_id"] == 45 and measured["slot"] == 1
    assert measured["decisions_to_reward"] == 2 and measured["game_frames_to_reward"] == 48
    assert measured["confirmed_choices_before_outcome"] == 2  # Excludes the future choice.
    assert measured["isolated_trace_decay_fraction"] == pytest.approx(math.exp(-.48 / .6))


def test_move_history_does_not_cross_encounters(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    m = importlib.import_module("audit_outcome_latency")
    rewards = SimpleNamespace(history={
        1: [{"hook": "move_confirmed", "frame": 24, "sample": 1}],
        2: [{"hook": "faint", "frame": 240, "sample": 10}],
    }, outcomes=[{"encounter": 2, "events": [], "delivered_at": {
        "hook": "faint", "frame": 240, "sample": 10,
    }}])
    assert m.summarize_outcomes(rewards, 12, .6)[0]["last_confirmed_move"] is None


@pytest.mark.parametrize("timing", ["encounter-end-v1", "last-faint-v3"])
@pytest.mark.parametrize("won", [True, False])
def test_end_observer_keeps_unrewarded_encounters_and_never_changes_game_or_reward(
    monkeypatch, timing, won,
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    m = importlib.import_module("audit_outcome_latency")
    memory = np.zeros(65536, np.uint8)
    hp = 20 if won else 0
    for address, value in {0xD057: 2, 0xD031: 25, 0xD163: 1, 0xD16D: hp,
                           0xD016: hp, 0xD89C: 1, 0xCF0B: int(not won),
                           0xCFE7: 0 if won else 15}.items():
        memory[address] = value
    original_memory = memory.copy()
    config = RewardConfig(timing=timing)
    observer = m.TimedRewards(config, lambda: 100)
    control = m.GeneralRewards(config)
    events = ["start", *(["faint", "trainer_win"] if won else []), "end"]
    for name in events:
        observer.event(name, memory)
        control.event(name, memory)
        assert observer.state() == control.state()
        np.testing.assert_array_equal(memory, original_memory)
    end = observer.history[1][-1]
    assert end["hook"] == "end"
    assert end["battle_result_raw"] == int(not won)
    assert end["party_alive"] == won
    assert end["player_hp"] == hp
    assert end["enemy_hp"] == (0 if won else 15)
    assert end["trainer_victory_observed"] == won
    assert end["enemy_faint_observed"] == won
    assert end["captured_species"] == 0
    assert bool(observer.outcomes) == won


@pytest.mark.parametrize("valid", [True, False])
def test_move_observer_validates_rom_block_before_hooking(monkeypatch, valid):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    m = importlib.import_module("audit_outcome_latency")
    calls = []

    class Memory:
        def __getitem__(self, key):
            assert key == (15, slice(0x538D, 0x539D))
            return m.MoveConfirmationHooks.signature if valid else bytes(16)

    game = SimpleNamespace(pyboy=SimpleNamespace(
        memory=Memory(), hook_register=lambda *args: calls.append(("register", args[:2])),
        hook_deregister=lambda *args: calls.append(("deregister", args)),
    ))
    if valid:
        with m.MoveConfirmationHooks(game, None):
            assert calls == [("register", (15, 0x539B))]
        assert calls[-1] == ("deregister", (15, 0x539B))
    else:
        with pytest.raises(ValueError, match="ROM signature mismatch"):
            with m.MoveConfirmationHooks(game, None):
                pytest.fail("An invalid ROM must never be hooked")
        assert calls == []
