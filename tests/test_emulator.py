import numpy as np
import pytest

from pokefly.actions import count_buttons, pressed_buttons
from pokefly.emulator import RedEmulator, button_phases


class FakePyBoy:
    def __init__(self):
        self.events = []

    def button_press(self, button):
        self.events.append(("press", button))

    def button_release(self, button):
        self.events.append(("release", button))

    def tick(self, frames, render):
        self.events.append(("tick", frames))
        return True


def test_repeated_buttons_have_a_release_frame() -> None:
    game = RedEmulator.__new__(RedEmulator)
    game.pyboy = FakePyBoy()
    assert game.act("a", frames=24)
    assert game.act("a", frames=24)
    assert game.pyboy.events == [
        ("press", "a"),
        ("tick", 23),
        ("release", "a"),
        ("tick", 1),
        ("press", "a"),
        ("tick", 23),
        ("release", "a"),
        ("tick", 1),
    ]


def test_direction_and_function_press_and_release_together():
    game = RedEmulator.__new__(RedEmulator)
    game.pyboy = FakePyBoy()
    assert game.act("up+a", frames=24)
    assert game.pyboy.events == [
        ("press", "up"),
        ("press", "a"),
        ("tick", 23),
        ("release", "up"),
        ("release", "a"),
        ("tick", 1),
    ]


@pytest.mark.parametrize("bad", ["up+down", "up+left", "a+b", "a+up", "up+up", "wait+a", "x", ""])
def test_invalid_combinations_queue_nothing(bad):
    game = RedEmulator.__new__(RedEmulator)
    game.pyboy = FakePyBoy()
    with pytest.raises(ValueError):
        game.act(bad)
    assert game.pyboy.events == []


@pytest.mark.parametrize("failure", ["press", "tick"])
def test_emulator_failure_releases_every_queued_button(failure):
    game = RedEmulator.__new__(RedEmulator)
    game.pyboy = FakePyBoy()

    def fail(*args):
        raise RuntimeError("Test failure")

    if failure == "tick":
        game.pyboy.tick = fail
        expected = [("press", "up"), ("press", "a"), ("release", "up"), ("release", "a")]
    else:
        original = game.pyboy.button_press
        game.pyboy.button_press = lambda b: fail() if b == "a" else original(b)
        expected = [("press", "up"), ("release", "up")]
    with pytest.raises(RuntimeError, match="Test failure"):
        game.act("up+a")
    assert game.pyboy.events == expected


def test_button_counts_are_not_command_counts():
    counts = count_buttons({"up+a": 3, "up": 2, "b": 1, "wait": 5})
    assert counts == {"up": 5, "down": 0, "left": 0, "right": 0, "a": 3, "b": 1, "start": 0}
    assert pressed_buttons("wait") == ()


def test_sampled_pulse_preserves_press_release_and_frame_budget():
    game = RedEmulator.__new__(RedEmulator)
    game.pyboy = FakePyBoy()
    game.screen = lambda: np.zeros((144, 160, 4), np.uint8)
    observations = []
    assert game.act_sampled("up+a", 24, 12, lambda frame, t: observations.append(t))
    assert observations == list(range(2, 25, 2))
    assert game.pyboy.events[:2] == [("press", "up"), ("press", "a")]
    assert game.pyboy.events[-3:] == [("release", "up"), ("release", "a"), ("tick", 1)]
    assert sum(v for k, v in game.pyboy.events if k == "tick") == 24


def test_sampled_callback_failure_releases_buttons():
    game = RedEmulator.__new__(RedEmulator)
    game.pyboy = FakePyBoy()
    game.screen = lambda: np.zeros((144, 160, 4), np.uint8)

    def fail(frame, offset):
        raise RuntimeError("neural failure")

    with pytest.raises(RuntimeError, match="neural failure"):
        game.act_sampled("down+b", 24, 12, fail)
    assert game.pyboy.events[-2:] == [("release", "down"), ("release", "b")]


@pytest.mark.parametrize("frames,samples", [(1, 1), (24, 0), (24, 25)])
def test_invalid_sampling_does_not_press(frames, samples):
    game = RedEmulator.__new__(RedEmulator)
    game.pyboy = FakePyBoy()
    with pytest.raises(ValueError):
        game.act_sampled("up", frames, samples, lambda *_: None)
    assert game.pyboy.events == []


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
@pytest.mark.parametrize("function", ["a", "b", "start"])
def test_serial_channels_do_not_mask_each_other(direction, function):
    game = RedEmulator.__new__(RedEmulator)
    game.button_timing, game.pyboy = "serial-v2", FakePyBoy()
    assert game.act(direction + "+" + function, 24)
    assert game.pyboy.events == [
        ("press", direction),
        ("tick", 11),
        ("release", direction),
        ("tick", 1),
        ("press", function),
        ("tick", 11),
        ("release", function),
        ("tick", 1),
    ]


@pytest.mark.parametrize("frames", [4, 5, 23, 24, 120])
def test_serial_sampling_is_independent_of_phase_boundaries(frames):
    for samples in (1, 3, frames):
        game = RedEmulator.__new__(RedEmulator)
        game.button_timing, game.pyboy = "serial-v2", FakePyBoy()
        game.screen = lambda: np.zeros((144, 160, 4), np.uint8)
        observed = []
        assert game.act_sampled("up+a", frames, samples, lambda _, t, seen=observed: seen.append(t))
        assert observed == [(i + 1) * frames // samples for i in range(samples)]
        assert sum(n for kind, n in game.pyboy.events if kind == "tick") == frames
        assert [e for e in game.pyboy.events if e[0] != "tick"] == [
            ("press", "up"),
            ("release", "up"),
            ("press", "a"),
            ("release", "a"),
        ]


@pytest.mark.parametrize("action", ["wait", "a", "up", "start"])
def test_serial_preserves_single_channel_pulse(action):
    assert button_phases(action, 24, "serial-v2") == button_phases(action, 24, "simultaneous-v1")


def test_serial_callback_failure_releases_current_channel():
    game = RedEmulator.__new__(RedEmulator)
    game.button_timing, game.pyboy = "serial-v2", FakePyBoy()
    game.screen = lambda: np.zeros((144, 160, 4), np.uint8)

    def fail(_frame, offset):
        if offset == 14:
            raise RuntimeError("neural failure")

    with pytest.raises(RuntimeError, match="neural failure"):
        game.act_sampled("up+a", 24, 12, fail)
    assert game.pyboy.events[-1] == ("release", "a")


def test_serial_invalid_frames_queue_nothing():
    game = RedEmulator.__new__(RedEmulator)
    game.button_timing, game.pyboy = "serial-v2", FakePyBoy()
    with pytest.raises(ValueError):
        game.act("up+a", 3)
    assert game.pyboy.events == []
