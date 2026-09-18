import threading
import time
from pathlib import Path

import pytest

from pokefly.cli import _parser
from pokefly.experiment import TrainOptions
from pokefly.pacing import RuntimePacing, SimulationRate


def test_train_defaults_to_unlimited_but_explicit_limits_remain():
    parser = _parser()
    assert parser.parse_args(["train"]).steps == 0
    assert TrainOptions(rom=Path("unused.gb")).steps == 0
    assert parser.parse_args(["train", "--steps", "20"]).steps == 20
    launcher = Path("scripts/start.ps1").read_text()
    assert "[int]$Steps = 0" in launcher
    assert "[double]$Hz = 5" in launcher


@pytest.mark.parametrize("value", [-1, 3001, float("nan"), float("inf"), True, "5", None])
def test_pacing_rejects_invalid_rate_without_changing_state(value):
    pacing = RuntimePacing(5)
    with pytest.raises(ValueError):
        pacing.set_hz(value)
    assert pacing.state() == {"target_hz": 5, "revision": 0}


def test_pacing_revision_is_monotonic_and_duplicates_are_idempotent():
    pacing = RuntimePacing(5)
    assert pacing.set_hz(250) == {"target_hz": 250, "revision": 1}
    assert pacing.set_hz(250) == {"target_hz": 250, "revision": 1}
    assert pacing.set_hz(0) == {"target_hz": 0, "revision": 2}
    pacing.wait(time.perf_counter())


def test_pacing_change_wakes_an_existing_slow_wait():
    pacing = RuntimePacing(0.01)
    entered = threading.Event()

    def worker():
        entered.set()
        pacing.wait(time.perf_counter())

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        assert entered.wait(timeout=1)
        pacing.set_hz(0)
        thread.join(timeout=1)
        assert not thread.is_alive()
    finally:
        pacing.set_hz(0)
        thread.join(timeout=1)


def test_stop_interrupts_a_low_rate_wait():
    pacing = RuntimePacing(0.01)
    stopped = threading.Event()
    thread = threading.Thread(
        target=lambda: pacing.wait(time.perf_counter(), stopped=stopped.is_set), daemon=True
    )
    thread.start()
    try:
        stopped.set()
        thread.join(timeout=1)
        assert not thread.is_alive()
    finally:
        pacing.set_hz(0)
        thread.join(timeout=1)


def test_speed_is_measured_from_elapsed_time_not_a_target_or_resumed_frame_total():
    now = [10.0]
    rate = SimulationRate(24000, 24, clock=lambda: now[0])
    now[0] += 0.4
    assert rate.record(24024) == {"actual_game_speed": 1, "decisions_per_second": 2.5}
    now[0] += 0.4
    assert rate.record(24048)["actual_game_speed"] == 1
    now[0] += 2  # A long checkpoint/delay must reduce measured speed.
    assert rate.record(24072)["actual_game_speed"] < 1
    for _ in range(100):
        now[0] += 0.04
        record = rate.record(rate.points[-1][1] + 24)
    assert record == {"actual_game_speed": 10, "decisions_per_second": 25}
    assert len(rate.points) <= 53
