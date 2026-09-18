"""Read-only telemetry for the exact English Red release validated by rom.py.

Addresses: https://github.com/pret/pokered/blob/symbols/pokered.sym
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Memory(Protocol):
    def __getitem__(self, address: int) -> int: ...


CURRENT_MAP = 0xD35E
Y_COORD = 0xD361
X_COORD = 0xD362
STATUS_FLAGS_6 = 0xD732
PARTY_COUNT = 0xD163
PARTY_LEVEL = 0xD18C
PARTY_STRIDE = 44
BADGES = 0xD356
IN_BATTLE = 0xD057
EVENT_FLAGS = 0xD747
EVENT_FLAG_BYTES = 320


@dataclass(frozen=True, slots=True)
class RedState:
    started: bool
    map_id: int
    x: int
    y: int
    battle: int
    party_count: int
    level_sum: int
    badges: int
    events: bytes

    @property
    def position(self) -> tuple[int, int, int]:
        return self.map_id, self.x, self.y

    @property
    def event_count(self) -> int:
        return sum(byte.bit_count() for byte in self.events)

    def telemetry(self) -> dict[str, int | bool]:
        return {
            "started": self.started,
            "map": self.map_id,
            "x": self.x,
            "y": self.y,
            "battle": self.battle,
            "party": self.party_count,
            "levels": self.level_sum,
            "badges": self.badges.bit_count(),
            "events": self.event_count,
        }

    @classmethod
    def read(cls, memory: Memory) -> RedState:
        count = int(memory[PARTY_COUNT])
        # The intro uses game-shaped scratch memory before real play starts.
        started = bool(memory[STATUS_FLAGS_6] & 1) and 0 <= count <= 6
        return cls(
            started=started,
            map_id=int(memory[CURRENT_MAP]),
            x=int(memory[X_COORD]),
            y=int(memory[Y_COORD]),
            battle=int(memory[IN_BATTLE]) if started else 0,
            party_count=count if started else 0,
            level_sum=sum(int(memory[PARTY_LEVEL + PARTY_STRIDE * i]) for i in range(count))
            if started
            else 0,
            badges=int(memory[BADGES]) if started else 0,
            events=bytes(memory[EVENT_FLAGS + offset] for offset in range(EVENT_FLAG_BYTES))
            if started
            else bytes(EVENT_FLAG_BYTES),
        )


class ProgressReward:
    """Novelty and milestones, paid once per run; RAM never reaches the policy."""

    def __init__(self) -> None:
        self.tiles: set[tuple[int, int, int]] = set()
        self.maps: set[int] = set()
        self.events = bytearray(EVENT_FLAG_BYTES)
        self.badges = 0
        self.max_party = 0
        self.max_levels = 0
        self.initialized = False

    def observe(self, state: RedState) -> tuple[float, dict[str, float]]:
        if not state.started:
            return 0.0, {}
        parts: dict[str, float] = {"time": -0.01}
        if state.battle == 0:
            if state.map_id not in self.maps:
                parts["map"] = 2.0
            if state.position not in self.tiles:
                parts["tile"] = 0.2
            self.maps.add(state.map_id)
            self.tiles.add(state.position)
        new_events = sum(
            (b & ~old).bit_count() for b, old in zip(state.events, self.events, strict=True)
        )
        parts["event"] = float(new_events)
        parts["badge"] = 10.0 * (state.badges & ~self.badges).bit_count()
        parts["party"] = 2.0 * max(0, state.party_count - self.max_party)
        parts["level"] = 0.1 * max(0, state.level_sum - self.max_levels)
        self.events = bytearray(old | b for old, b in zip(self.events, state.events, strict=True))
        self.badges |= state.badges
        self.max_party = max(self.max_party, state.party_count)
        self.max_levels = max(self.max_levels, state.level_sum)
        if not self.initialized:
            self.initialized = True
            return 0.0, {}  # Establish a baseline, including when loading a save state.
        return sum(parts.values()), {key: value for key, value in parts.items() if value}
