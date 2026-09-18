"""Diagnose raw-renderer resume independently of the fly; replay is NOT gameplay.

Only recorded actions are replayed. A temporary warm-render frame is discarded
by reloading the exact saved state; it is never supplied to neurons or rewards.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from pokefly.emulator import RedEmulator
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json
from pokefly.temporal import frame_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--boundary", type=int, default=50)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / "trajectory.jsonl").read_text().splitlines()]
    config = json.loads((args.run / "config.json").read_text())["config"]
    options = dict(button_timing=config["button_timing"])
    rom = resolve_rom(None, Path.cwd())
    output = run_directory("emulator-resume-probe")
    report = {"scope": __doc__, "source": str(args.run), "boundary": args.boundary, "arms": []}
    reference = []
    with RedEmulator(rom, **options) as game:
        game.load(args.run / "start.state", advance=False)
        for row in rows[:args.boundary]:
            game.act_sampled(row["action"], config["frames"], 12, lambda *args: None)
        buffer = io.BytesIO()
        game.pyboy.save_state(buffer)
        state = buffer.getvalue()
        game.act_sampled(rows[args.boundary]["action"], config["frames"], 12,
                         lambda frame, offset: reference.append(frame.copy()))
    recorded_hashes = rows[args.boundary + 1]["input_window"]["frame_hashes"]
    report["replay_matches_recorded_frames"] = list(map(frame_hash, reference)) == recorded_hashes
    for warm_frames in (0, 1, 2):
        with RedEmulator(rom, **options) as game:
            game.pyboy.load_state(io.BytesIO(state))
            if warm_frames:
                game.tick(warm_frames)
                game.pyboy.load_state(io.BytesIO(state))
            buffer = io.BytesIO()
            game.pyboy.save_state(buffer)
            actual = []
            game.act_sampled(rows[args.boundary]["action"], config["frames"], 12,
                             lambda frame, offset, actual=actual: actual.append(frame.copy()))
            differences = [np.any(a[:, :, :3] != b[:, :, :3], axis=2)
                           for a, b in zip(reference, actual, strict=True)]
            report["arms"].append({
                "discarded_warm_render_frames": warm_frames,
                "serialized_state_identical_before_replay": buffer.getvalue() == state,
                "different_pixels_per_sample": [int(d.sum()) for d in differences],
                "different_scanlines": [
                    np.flatnonzero(d.any(axis=1)).tolist() for d in differences
                ],
            })
    report["saved_state_sha256"] = hashlib.sha256(state).hexdigest()
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
