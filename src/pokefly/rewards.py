"""General outcome rewards; this module never supplies a button or sensory feature.

Exact-ROM CPU execution hooks distinguish outcomes which coarse RAM polling
cannot (including a capture field cleared before the next sampled frame).
Hooks alter only the emulator's in-memory debug opcodes, never the ROM file.
Addresses: pret/pokered symbols 3f618d59edf43918f48f5e558c34e04cb2fc5619.
Source: pret/pokered a1a22aaf84d1675bcdbaeb194592379d586d838e.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from pokefly.red_state import PARTY_STRIDE, RedState

SYMBOLS_REVISION = "3f618d59edf43918f48f5e558c34e04cb2fc5619"
SOURCE_REVISION = "a1a22aaf84d1675bcdbaeb194592379d586d838e"
# name, bank, instruction address, expected original instruction bytes
HOOKS = (
    ("overworld", 0, 0x03FF, bytes.fromhex("cdaf20")),
    ("start", 15, 0x411E, bytes.fromhex("afea58d0")),
    ("faint", 15, 0x4567, bytes.fromhex("cd434d")),
    ("trainer_win", 15, 0x4696, bytes.fromhex("cd4346")),
    ("ball_done", 3, 0x5928, bytes.fromhex("fa5ad0a7")),
    ("end", 4, 0x77AA, bytes.fromhex("fa2bd1fe04")),
)


@dataclass(frozen=True)
class RewardConfig:
    """Provisional engineering weights, not user-approved biological constants.

    Positive categories only; no added punishment/shaping. Every genuine completed
    encounter may pay again, but never twice in one encounter. Novelty is persistent.
    Catch and ordinary win have equal weight, and are mutually exclusive.
    """

    new_tile: float = 0.05
    new_area: float = 1.0
    battle_win: float = 1.0
    capture: float = 1.0
    gym_win: float = 2.0
    rival_win: float = 2.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Reward magnitudes must be finite and nonnegative")


class GeneralRewards:
    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.tiles: set[tuple[int, int, int]] = set()
        self.maps: set[int] = set()
        self.initialized = False
        self.active: dict | None = None
        self.encounters = 0
        self.pending: list[dict] = []
        self.counts: Counter = Counter()
        self.total = 0.0

    def baseline(self, state: RedState) -> None:
        if state.started:
            self.initialized = True
            if not state.battle:
                self.maps.add(state.map_id)
                self.tiles.add(state.position)

    def _emit(self, category: str, **details) -> None:
        amount = float(getattr(self.config, category))
        self.pending.append({"category": category, "reward": amount, **details})
        self.counts[category] += 1
        self.total += amount

    def visit(self, state: RedState) -> None:
        if not state.started or state.battle:
            return
        if not self.initialized:
            self.baseline(state)
            return
        if state.map_id not in self.maps:
            self._emit("new_area", map=state.map_id)
        if state.position not in self.tiles:
            self._emit("new_tile", position=list(state.position))
        self.maps.add(state.map_id)
        self.tiles.add(state.position)

    def start(self, memory) -> None:
        self.encounters += 1
        self.active = {
            "id": self.encounters,
            "battle": int(memory[0xD057]),
            "type": int(memory[0xD05A]),
            "trainer": int(memory[0xD031]),
            "trainer_number": int(memory[0xD05D]),
            "gym": int(memory[0xD05C]),
            "link": int(memory[0xD12B]),
            "fainted": False,
            "trainer_won": False,
            "captured": 0,
        }

    def event(self, name: str, memory) -> None:
        if name == "overworld":
            # Read at the game's own overworld loop, not mid-warp scratch coordinates.
            self.visit(RedState.read(memory))
            return
        if name == "start":
            self.start(memory)
            return
        encounter = self.active
        if encounter is None:
            return  # Missing history is not evidence of an outcome.
        if name == "faint":
            encounter["fainted"] = True
        elif name == "trainer_win":
            encounter["trainer_won"] = True
        elif name == "ball_done":
            # .done follows AddPartyMon/SendNewMonToBox for a successful catch.
            # Failed throws also pass here, but their captured species is zero.
            if int(memory[0xD05A]) != 1:
                encounter["captured"] = int(memory[0xD11C])
        elif name == "end":
            self.active = None  # Consume exactly once even if the hook fires again.
            if encounter["type"] == 1 or encounter["link"] or encounter["battle"] not in (1, 2):
                return  # Old-man demonstration, link battle, or invalid encounter.
            details = {"encounter": encounter["id"], "trainer": encounter["trainer"]}
            if encounter["captured"] and encounter["battle"] == 1:
                self._emit("capture", species=encounter["captured"], **details)
                return
            count = min(6, max(0, int(memory[0xD163])))
            alive = any(
                memory[0xD16C + PARTY_STRIDE * i] or memory[0xD16D + PARTY_STRIDE * i]
                for i in range(count)
            )
            confirmed = (encounter["battle"] == 1 and encounter["fainted"]) or (
                encounter["battle"] == 2 and encounter["trainer_won"]
            )
            if int(memory[0xCF0B]) == 0 and alive and confirmed:
                self._emit("battle_win", **details)
                trainer = encounter["trainer"]
                if encounter["battle"] == 2:
                    if trainer in range(0x22, 0x29) or (trainer == 0x1D and encounter["gym"]):
                        self._emit("gym_win", **details)
                    if trainer in (0x19, 0x2A, 0x2B):
                        self._emit("rival_win", **details)

    def drain(self) -> tuple[float, list[dict]]:
        events, self.pending = self.pending, []
        return sum(item["reward"] for item in events), events

    def state(self) -> dict:
        return {
            "tiles": sorted(self.tiles),
            "maps": sorted(self.maps),
            "initialized": self.initialized,
            "active": self.active,
            "encounters": self.encounters,
            "pending": self.pending,
            "counts": dict(self.counts),
            "total": self.total,
        }

    def restore(self, state: dict) -> None:
        self.tiles = {tuple(map(int, tile)) for tile in state["tiles"]}
        self.maps = set(map(int, state["maps"]))
        self.initialized, self.active = bool(state["initialized"]), state["active"]
        self.encounters = int(state["encounters"])
        self.pending = list(state["pending"])
        self.counts = Counter(state["counts"])
        self.total = float(state["total"])


class RewardHooks:
    def __init__(self, game, rewards: GeneralRewards):
        self.game, self.rewards = game, rewards
        self.registered: list[tuple[int, int]] = []

    def attach(self) -> None:
        if self.registered:
            raise RuntimeError("Reward hooks are already installed")
        # Fail closed if symbols/ROM instructions ever disagree.
        for name, bank, address, signature in HOOKS:
            actual = bytes(self.game.pyboy.memory[bank, address : address + len(signature)])
            if actual != signature:
                raise ValueError(f"Reward hook ROM signature mismatch: {name}")
        try:
            for name, bank, address, _ in HOOKS:
                self.game.pyboy.hook_register(bank, address, self._callback, name)
                self.registered.append((bank, address))
        except Exception:
            self.detach()
            raise

    def _callback(self, name):
        self.rewards.event(name, self.game.pyboy.memory)

    def detach(self) -> None:
        for bank, address in self.registered:
            self.game.pyboy.hook_deregister(bank, address)
        self.registered.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *_args):
        self.detach()
