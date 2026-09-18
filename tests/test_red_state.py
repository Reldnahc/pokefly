from collections import defaultdict
from dataclasses import replace

import pytest

from pokefly.red_state import (
    CURRENT_MAP,
    EVENT_FLAGS,
    STATUS_FLAGS_6,
    X_COORD,
    Y_COORD,
    ProgressReward,
    RedState,
)


def test_reads_supported_state() -> None:
    memory: defaultdict[int, int] = defaultdict(int)
    memory.update({CURRENT_MAP: 7, X_COORD: 3, Y_COORD: 9, EVENT_FLAGS: 0b10101, STATUS_FLAGS_6: 1})
    state = RedState.read(memory)
    assert state.started and state.position == (7, 3, 9) and state.event_count == 3


def test_intro_scratch_ram_does_not_earn_rewards() -> None:
    memory = defaultdict(int, {CURRENT_MAP: 38, EVENT_FLAGS: 255})
    state = RedState.read(memory)
    tracker = ProgressReward()
    assert state.event_count == 0
    assert tracker.observe(state) == (0.0, {})
    assert not tracker.tiles


def test_events_and_tiles_cannot_be_farmed_by_toggling() -> None:
    memory = defaultdict(int, {CURRENT_MAP: 38, STATUS_FLAGS_6: 1})
    initial = RedState.read(memory)
    tracker = ProgressReward()
    assert tracker.observe(initial) == (0.0, {})
    memory[X_COORD] = 1
    memory[EVENT_FLAGS] = 1
    changed = RedState.read(memory)
    reward, parts = tracker.observe(changed)
    assert reward == pytest.approx(1.19)
    assert parts["tile"] == 0.2 and parts["event"] == 1
    tracker.observe(initial)
    assert tracker.observe(changed) == (-0.01, {"time": -0.01})


def test_loading_state_uses_existing_progress_as_baseline() -> None:
    memory = defaultdict(int, {CURRENT_MAP: 38, STATUS_FLAGS_6: 1, EVENT_FLAGS: 255})
    tracker = ProgressReward()
    state = RedState.read(memory)
    assert tracker.observe(state) == (0.0, {})
    assert tracker.observe(state) == (-0.01, {"time": -0.01})
    tracker.observe(replace(state, battle=1, x=10))
    assert len(tracker.tiles) == 1
