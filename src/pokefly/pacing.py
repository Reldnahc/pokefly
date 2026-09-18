"""Wall-clock controls only: never change neural/game timesteps or reward rules."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from numbers import Real

GAME_FPS = 60.0  # Approximate normal Game Boy speed, for display/pacing only.
MAX_HZ = 3000.0  # 100x even with the minimum two game frames per decision.


class RuntimePacing:
    def __init__(self, hz: float):
        self._condition = threading.Condition()
        self._hz = self.validate(hz)
        self._revision = 0

    @staticmethod
    def validate(hz) -> float:
        if isinstance(hz, bool) or not isinstance(hz, Real):
            raise ValueError("Decision rate must be a number")
        if not math.isfinite(hz) or not 0 <= hz <= MAX_HZ:
            raise ValueError(f"hz must be in [0,{MAX_HZ:g}] (0 = unthrottled)")
        return float(hz)

    def state(self) -> dict:
        with self._condition:
            return {"target_hz": self._hz, "revision": self._revision}

    def set_hz(self, hz: float) -> dict:
        hz = self.validate(hz)
        with self._condition:
            if hz != self._hz:
                self._hz = hz
                self._revision += 1
                self._condition.notify_all()
            return {"target_hz": self._hz, "revision": self._revision}

    def wait(self, started: float, *, stopped=lambda: False) -> None:
        # Recompute the deadline on edits, including while sleeping. Short waits
        # let Ctrl+C finish the current transaction even at a very low rate.
        with self._condition:
            while self._hz and not stopped():
                remaining = 1 / self._hz - (time.perf_counter() - started)
                if remaining <= 0:
                    return
                self._condition.wait(timeout=min(remaining, 0.1))


class SimulationRate:
    """Recent completed game frames / elapsed wall time, not GPU-time estimates."""

    def __init__(self, frames: int, frames_per_decision: int, *, clock=time.perf_counter):
        self.clock = clock
        self.frames_per_decision = frames_per_decision
        self.points = deque([(clock(), frames)])

    def record(self, frames: int) -> dict:
        now = self.clock()
        self.points.append((now, frames))
        while len(self.points) > 2 and self.points[1][0] <= now - 2:
            self.points.popleft()
        then, previous = self.points[0]
        elapsed = now - then
        fps = (frames - previous) / elapsed if elapsed > 0 else 0.0
        return {
            "actual_game_speed": round(fps / GAME_FPS, 4),
            "decisions_per_second": round(fps / self.frames_per_decision, 4),
        }
