import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assay_stimuli import motion_grating, observe_stimulus  # noqa: E402


def test_opposite_motion_shares_first_frame_and_moves_without_luminance_cue():
    first = motion_grating("left", 0)
    np.testing.assert_array_equal(first, motion_grating("right", 0))
    for step in range(16):
        for direction, sign in (("left", -1), ("right", 1)):
            frame = motion_grating(direction, step)
            assert frame.dtype == np.uint8 and frame.shape == (144, 160, 3)
            assert frame.mean() == 127.5
            np.testing.assert_array_equal(frame, np.roll(first, sign * 2 * step, axis=1))
    np.testing.assert_array_equal(first, motion_grating("left", 16))
    with pytest.raises(ValueError):
        motion_grating("up", 0)


def test_motion_is_raw_pixels_at_every_step_not_a_synthetic_neural_drive():
    class Controller:
        brain = type("Brain", (), {"n": 3})()
        brain_steps = 4

        def __init__(self):
            self.frames = []

        def integrate_frame(self, frame, counts):
            self.frames.append(frame.copy())
            counts[1] += 1
            return "retina result"

        def observation(self, counts, drive):
            return counts, drive

    c = Controller()
    counts, drive = observe_stimulus(c, "right", 3, "motion-grating-v1")
    np.testing.assert_array_equal(counts, [0, 4, 0])
    assert drive == "retina result"
    for i, frame in enumerate(c.frames, 12):
        np.testing.assert_array_equal(frame, motion_grating("right", i))
