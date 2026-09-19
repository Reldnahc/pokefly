import importlib
from pathlib import Path

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
