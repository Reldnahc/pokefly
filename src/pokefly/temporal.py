"""Versioned raw-image timing, with no rewards, RAM features, or button policy.

snapshot-v1 preserves the original observe -> choose -> whole pulse ordering.
stream-v1 chooses from the preceding neural window, then integrates one fresh
image per neural step while executing the next pulse. endpoint-v1 is a matched
timing control: it integrates the final image repeatedly after that same pulse.
Both new modes have one explicitly recorded initial static neural window.
Reward remains a decision-boundary update after the entire pulse/input window.
"""

from __future__ import annotations

import hashlib

import numpy as np

MODES = ("snapshot-v1", "stream-v1", "endpoint-v1")


def frame_hash(frame):
    return hashlib.sha256(np.ascontiguousarray(frame[:, :, :3]).tobytes()).hexdigest()


class TemporalVision:
    def __init__(self, controller, mode="snapshot-v1", *, arrays=None, state=None):
        if mode not in MODES:
            raise ValueError("Unknown visual timing")
        self.controller, self.mode = controller, mode
        self.pending, self.window = None, None
        if mode == "snapshot-v1":
            if state is not None:
                raise ValueError("Snapshot mode cannot restore a temporal window")
        elif state is not None:
            if state["mode"] != mode:
                raise ValueError("Temporal checkpoint mode mismatch")
            counts = np.asarray(arrays["temporal_counts"])
            drive = np.asarray(arrays["temporal_drive"])
            if (
                counts.shape != (controller.brain.n,)
                or counts.dtype.kind not in "iu"
                or (counts < 0).any()
                or (counts > controller.brain_steps).any()
                or drive.shape != (len(controller.brain.visual),)
                or not np.isfinite(drive).all()
                or ((drive < 0) | (drive > 1)).any()
            ):
                raise ValueError("Invalid temporal observation checkpoint")
            self.pending = controller.observation(counts.copy(), drive.copy())
            self.window = dict(state["window"])

    def observe(self, frame):
        if self.mode == "snapshot-v1":
            return self.controller.observe(frame)
        if self.pending is None:
            self.pending = self.controller.observe(frame)
            self.window = {
                "kind": "initial-static-window",
                "samples": self.controller.brain_steps,
                "unique_frames": 1,
                "frame_hashes": [frame_hash(frame)] * self.controller.brain_steps,
                "offsets": [0] * self.controller.brain_steps,
            }
        return self.pending

    def advance(self, game, action, frames):
        if self.mode == "snapshot-v1":
            return game.act(action, frames)
        c = self.controller
        hashes, offsets = [], []
        counts, drive = np.zeros(c.brain.n, np.int32), None

        def sample(frame, offset):
            nonlocal drive
            drive = c.integrate_frame(frame, counts)
            hashes.append(frame_hash(frame))
            offsets.append(offset)

        if self.mode == "stream-v1":
            if not game.act_sampled(action, frames, c.brain_steps, sample):
                return False
            self.pending = c.observation(counts, drive)
        else:
            if not game.act(action, frames):
                return False
            frame = game.screen()
            self.pending = c.observe(frame)
            hashes = [frame_hash(frame)] * c.brain_steps
            offsets = [frames] * c.brain_steps
        self.window = {
            "kind": self.mode,
            "samples": c.brain_steps,
            "unique_frames": len(set(hashes)),
            "frame_hashes": hashes,
            "offsets": offsets,
        }
        return True

    def arrays(self):
        if self.pending is None:
            return {}
        return {
            "temporal_counts": self.pending.counts,
            "temporal_drive": self.pending.drive,
        }

    def state(self):
        if self.pending is None:
            return None
        return {"mode": self.mode, "window": self.window}
