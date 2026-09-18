"""Fixed neural motor decoder. No game state, rewards, or fitted policy enters here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOTOR_TYPES = {
    "up": ("DNg100", None),
    "down": ("MDN", None),
    "left": ("DNa02", "L"),
    "right": ("DNa02", "R"),
    "a": ("CB0701", None),  # MN9, checked against MaleCNS body IDs below.
    "b": ("DNp01", None),
    "start": ("pIP10", None),
}
MOTOR_GROUPS = dict(
    zip(
        MOTOR_TYPES,
        (
            "forward_Up",
            "backward_Down",
            "steer_Left",
            "steer_Right",
            "proboscis_A",
            "escape_B",
            "song_Start",
        ),
        strict=True,
    )
)
MN9_IDS = {10331, 16949}
MOTOR_MAPPING_VERSION = "fixed-movement-v1-mn9"


def populations(brain, body_ids: np.ndarray) -> dict[str, np.ndarray]:
    result = {key: brain.cells([kind], side=side) for key, (kind, side) in MOTOR_TYPES.items()}
    if any(not len(indices) for indices in result.values()):
        raise ValueError("A required motor population is missing; no substitute is allowed")
    if set(map(int, body_ids[result["a"]])) != MN9_IDS:
        raise ValueError("MN9/CB0701 body IDs differ from the verified motor registry")
    return result


@dataclass(frozen=True)
class MotorConfig:
    trace_seconds: float = 0.15
    threshold_hz: float = 0.5
    start_cooldown_seconds: float = 1.0
    arbitration: str = "parallel-v2"
    direction_trace_seconds: float = 1.0

    def __post_init__(self):
        if self.arbitration not in ("exclusive-v1", "parallel-v2", "sustained-v3"):
            raise ValueError("Unknown fixed motor arbitration")
        if not np.isfinite(
            [self.trace_seconds, self.threshold_hz, self.start_cooldown_seconds,
             self.direction_trace_seconds]
        ).all():
            raise ValueError("Motor parameters must be finite")
        if self.trace_seconds <= 0 or self.threshold_hz <= 0 or self.start_cooldown_seconds < 0:
            raise ValueError("Invalid fixed motor parameters")
        if self.direction_trace_seconds <= 0:
            raise ValueError("Directional trace duration must be positive")


class MotorDecoder:
    """Fixed rate readout: one direction plus one function button; silence -> Wait.

    The channels compete only within themselves; A/B/Start cannot suppress Up.
    Legacy exclusive-v1 is retained for old checkpoints and matched controls.
    Exact ties rotate deterministically within each channel. Start has a cooldown.
    These settings NEVER adapt in response to rewards.
    This is a neural command readout, not a simulated physical fly body.
    """

    def __init__(self, groups: dict[str, np.ndarray], config: MotorConfig | None = None):
        if tuple(groups) != tuple(MOTOR_TYPES) or any(not len(v) for v in groups.values()):
            raise ValueError("All seven ordered motor populations are required")
        self.groups, self.config = groups, config or MotorConfig()
        self.rates = np.zeros(7, np.float64)
        self.time = 0.0
        self.start_ready = 0.0
        self.tie_cursor = 0
        self.direction_cursor = 0
        self.function_cursor = 0

    def _winner(self, scores, cursor):
        maximum = scores.max()
        if maximum < self.config.threshold_hz:
            return None, cursor
        tied = np.flatnonzero(np.isclose(scores, maximum, rtol=0, atol=1e-10))
        index = min(map(int, tied), key=lambda i: (i - cursor) % len(scores))
        return index, (index + 1) % len(scores)

    def choose(self, counts: np.ndarray, seconds: float) -> tuple[str, dict[str, float]]:
        if (
            not np.isfinite(seconds)
            or seconds <= 0
            or not np.isfinite(counts).all()
            or (counts < 0).any()
        ):
            raise ValueError("Nonnegative spike counts and positive time required")
        observed = np.array([counts[idx].mean() / seconds for idx in self.groups.values()])
        decay = np.exp(-seconds / self.config.trace_seconds)
        if self.config.arbitration == "sustained-v3":
            # Fixed movement-bout interpretation, identical for every direction.
            # Trace magnitude still comes only from actual motor spikes. No
            # position, collision, reward or action-frequency target is used.
            decay = np.array(
                [np.exp(-seconds / self.config.direction_trace_seconds)] * 4 + [decay] * 3
            )
        self.rates = decay * self.rates + (1 - decay) * observed
        self.time += seconds
        scores = self.rates.copy()
        if self.time < self.start_ready:
            scores[6] = 0
        # Legacy/function channels require current spikes. The optional sustained
        # direction readout permits a decaying trace, never a forced button timer.
        if self.config.arbitration == "sustained-v3":
            scores[4:][observed[4:] == 0] = 0
        else:
            scores[observed == 0] = 0
        if self.config.arbitration == "exclusive-v1":
            index, self.tie_cursor = self._winner(scores, self.tie_cursor)
            selected = [] if index is None else [index]
        else:
            direction, self.direction_cursor = self._winner(scores[:4], self.direction_cursor)
            function, self.function_cursor = self._winner(scores[4:], self.function_cursor)
            selected = ([direction] if direction is not None else []) + (
                [function + 4] if function is not None else []
            )
        if 6 in selected:
            self.start_ready = self.time + self.config.start_cooldown_seconds
        action = "+".join(tuple(self.groups)[i] for i in selected) or "wait"
        return action, dict(zip(self.groups, map(float, self.rates), strict=True))

    def state(self) -> dict:
        state = {
            "rates": self.rates.tolist(),
            "time": self.time,
            "start_ready": self.start_ready,
            "tie_cursor": self.tie_cursor,
        }
        if self.config.arbitration != "exclusive-v1":
            state.update(
                arbitration=self.config.arbitration,
                direction_cursor=self.direction_cursor,
                function_cursor=self.function_cursor,
            )
        return state

    def restore(self, state: dict) -> None:
        if state.get("arbitration", "exclusive-v1") != self.config.arbitration:
            raise ValueError("Motor checkpoint arbitration mismatch")
        rates = np.asarray(state["rates"], dtype=np.float64)
        if rates.shape != (7,) or not np.isfinite(rates).all() or (rates < 0).any():
            raise ValueError("Invalid motor checkpoint")
        time, ready = float(state["time"]), float(state["start_ready"])
        cursor = int(state["tie_cursor"])
        direction, function = 0, 0
        if self.config.arbitration != "exclusive-v1":
            direction, function = int(state["direction_cursor"]), int(state["function_cursor"])
        if (
            not np.isfinite([time, ready]).all()
            or min(time, ready) < 0
            or not 0 <= cursor < 7
            or not 0 <= direction < 4
            or not 0 <= function < 3
        ):
            raise ValueError("Invalid motor checkpoint timing")
        self.rates, self.time, self.start_ready = rates.copy(), time, ready
        self.tie_cursor, self.direction_cursor, self.function_cursor = cursor, direction, function
