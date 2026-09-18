"""Synthetic ROM-free diagnostic images; never used by the gameplay controller."""

import numpy as np

from pokefly.pixel_brain import test_patterns


def motion_grating(direction, neural_step):
    """32-pixel period, 50% duty, 2 pixels/step; opposite motion, equal luminance."""
    if direction not in ("left", "right") or neural_step < 0:
        raise ValueError("A nonnegative step and horizontal direction are required")
    phase = 2 * neural_step * (1 if direction == "left" else -1)
    row = ((((np.arange(160) + phase) // 16) % 2) * 255).astype(np.uint8)
    return np.broadcast_to(row[None, :, None], (144, 160, 3)).copy()


def observe_stimulus(controller, cue, decision, stimulus):
    if stimulus == "static-half-v1":
        return controller.observe(test_patterns()[cue])
    if stimulus != "motion-grating-v1":
        raise ValueError("Unknown diagnostic stimulus")
    counts = np.zeros(controller.brain.n, np.int32)
    for offset in range(controller.brain_steps):
        frame = motion_grating(cue, decision * controller.brain_steps + offset)
        drive = controller.integrate_frame(frame, counts)
    return controller.observation(counts, drive)
