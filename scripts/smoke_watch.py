"""Opt-in end-to-end test using the local ROM and real fly data; no browser needed.

Run from the project root with the project Python. It starts its own short-lived
watch process, verifies synchronized pixels/drive/actions, and preserves its run.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from pokefly.emulator import button_phases
from pokefly.rom import resolve_rom, validate_rom
from pokefly.runtime import configure_runtime
from pokefly.vision import RetinaMap


def packet(stream):
    while line := stream.readline():
        if line.startswith(b"data: "):
            return json.loads(line[6:])
    raise RuntimeError("Event stream ended before a sample arrived")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--autonomous", action="store_true", help="test the internal train display")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.config and not args.autonomous:
        parser.error("--config requires --autonomous")
    data = configure_runtime()
    rom = resolve_rom(None, Path.cwd())
    original_hash = validate_rom(rom)
    retina = RetinaMap.load(data / "retina.npz", data / "brain.npz")
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "pokefly",
            "train" if args.autonomous else "watch",
            "--device",
            args.device,
            *([] if args.autonomous else ["--manual"]),
            *(["--config", str(args.config)] if args.config else []),
            "--intro",
            "--port",
            "0",
            "--steps",
            "50",
            "--hz",
            "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = queue.Queue()
    transcript = []

    def read_output():
        for line in process.stdout:
            transcript.append(line.rstrip())
            lines.put(line.strip())
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        while True:
            line = lines.get(timeout=45)
            if line is None:
                raise RuntimeError("Watch failed before starting: " + "\n".join(transcript))
            if line.startswith("Dashboard: "):
                url = line.removeprefix("Dashboard: ")
                break
        with urllib.request.urlopen(url + "/static.json", timeout=5) as response:
            metadata = json.load(response)
        assert metadata["internal_learning"] is args.autonomous
        assert metadata["manual"] is not args.autonomous
        assert metadata["retina"]["photoreceptors"] == len(retina.indices)
        with urllib.request.urlopen(url + "/events", timeout=5) as stream:
            first = packet(stream)
            image_bytes = base64.b64decode(first["screen"].split(",", 1)[1])
            frame = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            assert frame.shape == (144, 160, 3)
            expected_drive = np.rint(retina.encode(frame) * 255).astype(np.uint8)
            shown_drive = np.frombuffer(base64.b64decode(first["drive_u8"]), np.uint8)
            np.testing.assert_array_equal(expected_drive, shown_drive)
            if metadata.get("visual_timing", "snapshot-v1") != "snapshot-v1":
                from pokefly.temporal import frame_hash

                assert first["input_window"]["frame_hashes"][-1] == frame_hash(frame)
            if metadata.get("graded_indices"):
                graded = np.frombuffer(base64.b64decode(first["graded_u8"]), np.uint8)
                assert len(graded) == len(metadata["graded_indices"])
                assert first["groups"]["photoreceptors"] == 0
                assert first["groups"]["lamina_L1_L5"] == 0
                assert "NOT spikes" in first["graded_units"]
                assert all(0 <= v <= 1 for v in first["graded_groups"].values())
                assert not set(first["spikes"]) & set(metadata["graded_indices"])
            request = urllib.request.Request(
                url + "/control",
                data=b'{"action":"right"}',
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Origin": url,
                    "X-Pokefly-Token": metadata["control_token"],
                },
            )
            if args.autonomous:
                try:
                    urllib.request.urlopen(request, timeout=5)
                except urllib.error.HTTPError as exc:
                    assert exc.code == 403  # Even a valid token cannot intervene in training.
                else:
                    raise AssertionError("Autonomous controller accepted a human button")
            else:
                with urllib.request.urlopen(request, timeout=5) as response:
                    assert response.status == 202
            delivered = None
            speed_changes = []
            for _ in range(20):
                current = packet(stream)
                if (args.autonomous and current["action"] != "wait") or (
                    not args.autonomous and current["action"] == "right"
                ):
                    delivered = current
                    break
            assert delivered is not None, "No delivered pulse was observed"
            assert delivered["action_source"] == ("fly" if args.autonomous else "human")
            assert delivered["buttons_released"] is True
            phases = button_phases(
                delivered["action"],
                metadata["emulator_frames_per_sample"],
                metadata.get("button_timing", "simultaneous-v1"),
            )
            assert delivered["pulse_frames"] == sum(n for keys, n in phases if keys)
            if metadata.get("button_timing") == "serial-v2":
                assert delivered["button_phases"] == [
                    {"buttons": list(keys), "frames": n} for keys, n in phases
                ]
            if args.autonomous:
                from pokefly.actions import pressed_buttons

                assert delivered["buttons"] == list(pressed_buttons(delivered["action"]))
                from pokefly.experiment import load_config

                expected_motor = load_config(args.config).brain.motor.arbitration
                assert metadata["motor_arbitration"] == expected_motor
                assert delivered["learning"]["plastic_edges"] == metadata["plastic_edges"]
                assert "a" in delivered["motor_rates_hz"]
                if metadata.get("visual_timing", "snapshot-v1") == "snapshot-v1":
                    assert "pre-action" in metadata["screen_timing"]
                else:
                    assert "input window" in metadata["screen_timing"]
                    assert delivered["input_window"]["samples"] == 12
                    assert delivered["input_window"]["frame_hashes"][-1]
                assert metadata["speed_control"] is True
                for hz in (2.5, 0):
                    speed_request = urllib.request.Request(
                        url + "/speed",
                        data=json.dumps({"hz": hz}).encode(),
                        method="POST",
                        headers={
                            "Content-Type": "application/json",
                            "Origin": url,
                            "X-Pokefly-Token": metadata["control_token"],
                        },
                    )
                    with urllib.request.urlopen(speed_request, timeout=5) as response:
                        assert response.status == 200
                        acknowledged = json.load(response)["pacing"]
                    for _ in range(20):
                        current = packet(stream)
                        if current["pacing"]["revision"] >= acknowledged["revision"]:
                            assert current["pacing"]["target_hz"] == hz
                            assert current["pacing"]["actual_game_speed"] > 0
                            speed_changes.append(hz)
                            break
                    else:
                        raise AssertionError("Live speed edit never reached the experiment")
        process.wait(timeout=30)
        reader.join(timeout=2)
        assert process.returncode == 0, "\n".join(transcript)
        output = Path(
            next(line.removeprefix("Run: ") for line in transcript if line.startswith("Run: "))
        )
        summary = json.loads((output / "summary.json").read_text())
        if not args.autonomous:
            assert summary["actions"]["right"] == 1
        assert summary["samples"] == 50
        records = [
            json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()
        ]
        logged = next(row for row in records if row["sample"] == delivered["sample"])
        assert all(
            logged[key] == delivered[key]
            for key in ("action", "action_source", "groups", "telemetry", "buttons_released")
        )
        if args.autonomous:
            assert logged["buttons"] == delivered["buttons"]
            assert all(row["buttons"] == list(pressed_buttons(row["action"])) for row in records)
            assert speed_changes == [2.5, 0]
            assert {2.5, 0} <= {row["pacing"]["target_hz"] for row in records}
        assert validate_rom(rom) == original_hash
        result = {
            "visual_timing": metadata.get("visual_timing", "snapshot-v1"),
            "button_timing": metadata.get("button_timing", "simultaneous-v1"),
            "config": str(args.config) if args.config else "baseline",
            "graded_activity_separate_from_spikes": bool(metadata.get("graded_indices")),
            "sensory_isolation": metadata["sensory_isolation"],
            "pixel_display_matches_neural_drive": True,
            "action_source": "fly" if args.autonomous else "human",
            "pulse_logged_and_released": True,
            "multi_button_log_matches_commands": args.autonomous,
            "human_controls_disabled": args.autonomous,
            "live_speed_changes_hz": speed_changes,
            "rom_unchanged": True,
            "samples": summary["samples"],
            "run": str(output),
        }
        (output / "smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        if process.poll() is None:
            process.terminate()  # Only this test's own child process, never an existing run.
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
