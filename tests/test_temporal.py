from types import SimpleNamespace

import numpy as np
import pytest

from pokefly.experiment import ExperimentConfig
from pokefly.pixel_brain import PixelObservation
from pokefly.temporal import TemporalVision


class Controller:
    brain_steps = 12
    brain = SimpleNamespace(n=3, visual=np.array([0]))

    def __init__(self):
        self.inputs = []

    def integrate_frame(self, frame, counts):
        self.inputs.append(int(frame[0, 0, 0]))
        counts[0] += 1
        return np.array([frame[0, 0, 0] / 255], np.float32)

    def observation(self, counts, drive):
        return PixelObservation(drive, counts, {}, self.brain_steps)

    def observe(self, frame):
        counts = np.zeros(3, np.int32)
        for _ in range(self.brain_steps):
            drive = self.integrate_frame(frame, counts)
        return self.observation(counts, drive)


class Game:
    def __init__(self):
        self.elapsed = 0

    def screen(self):
        return np.full((144, 160, 3), self.elapsed, np.uint8)

    def act(self, action, frames):
        self.elapsed += frames
        return True

    def act_sampled(self, action, frames, samples, observe):
        start = self.elapsed
        for i in range(samples):
            offset = (i + 1) * frames // samples
            self.elapsed = start + offset
            observe(self.screen(), offset)
        return True


def test_stream_observes_raw_intermediate_frames_and_restore_does_not_reobserve():
    c, game = Controller(), Game()
    t = TemporalVision(c, "stream-v1")
    t.observe(game.screen())
    assert c.inputs == [0] * 12
    assert t.advance(game, "up", 24)
    assert c.inputs[12:] == list(range(2, 25, 2))
    assert t.window["unique_frames"] == 12
    before = len(c.inputs)
    restored = TemporalVision(c, "stream-v1", arrays=t.arrays(), state=t.state())
    observation = restored.observe(game.screen())
    assert len(c.inputs) == before
    np.testing.assert_array_equal(observation.counts, t.pending.counts)
    np.testing.assert_array_equal(observation.drive, t.pending.drive)
    assert restored.state() == t.state()


def test_endpoint_has_same_neural_budget_but_holds_final_frame():
    c, game = Controller(), Game()
    t = TemporalVision(c, "endpoint-v1")
    t.observe(game.screen())
    t.advance(game, "up", 24)
    assert c.inputs == [0] * 12 + [24] * 12
    assert t.window["unique_frames"] == 1


def test_legacy_timing_does_not_add_neural_steps_or_checkpoint_arrays():
    c, game = Controller(), Game()
    t = TemporalVision(c)
    t.observe(game.screen())
    t.advance(game, "up", 24)
    assert c.inputs == [0] * 12
    assert not t.arrays() and t.state() is None
    t.observe(game.screen())
    assert c.inputs == [0] * 12 + [24] * 12


def test_modes_and_corrupt_pending_observation_fail_closed():
    with pytest.raises(ValueError, match="timing"):
        ExperimentConfig(visual_timing="unknown")
    with pytest.raises(ValueError, match="game frame"):
        ExperimentConfig(visual_timing="stream-v1", frames=3)
    c = Controller()
    t = TemporalVision(c, "stream-v1")
    t.observe(Game().screen())
    arrays = t.arrays()
    arrays["temporal_counts"] = np.array([-1, 0, 0])
    with pytest.raises(ValueError, match="observation"):
        TemporalVision(c, "stream-v1", arrays=arrays, state=t.state())
