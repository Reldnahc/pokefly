"""Forced-button transport diagnostic on COPIES of a recorded naming screen.

Not autonomous play or neural training. Compare identical requested commands,
frame budgets and exact initial emulator state; never edit source checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.emulator import RedEmulator
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    _, saved = read_checkpoint(args.checkpoint)
    source = Path(saved["directory"]) / "game.state"
    output = run_directory("forced-button-delivery-probe")
    report = {"scope": __doc__, "source": str(source), "sha256": sha256(source), "rows": []}
    for timing in ("simultaneous-v1", "serial-v2"):
        with RedEmulator(resolve_rom(None, Path.cwd()), button_timing=timing) as game:
            for direction in ("up", "down", "left", "right"):
                game.load(source, advance=False)
                before = game.state().telemetry()
                if before["party"] != 1 or before["levels"] != 0:
                    p.error("Diagnostic expects a recorded, unfinished first nickname entry")
                history = []
                for _ in range(16):
                    game.act(direction + "+start", 24)
                    history.append(game.state().telemetry())
                game.screenshot(output / f"{timing}-{direction}.png")
                row = {
                    "timing": timing,
                    "command": direction + "+start",
                    "before": before,
                    "first_initialized_party_decision": next(
                        (i + 1 for i, s in enumerate(history) if s["levels"] > 0), None
                    ),
                    "history": history,
                }
                report["rows"].append(row)
                print(timing, direction, row["first_initialized_party_decision"], flush=True)
    assert sha256(source) == report["sha256"]
    write_json(output / "report.json", report)
    print("Transport diagnostic (not autonomous progress):", output, flush=True)


if __name__ == "__main__":
    main()
