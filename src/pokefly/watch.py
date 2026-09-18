"""Live pixel-input observatory; never substitutes a policy for the fly."""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

from pokefly.dashboard import BrainView, Dashboard, png_data
from pokefly.emulator import RedEmulator
from pokefly.pixel_brain import PixelBrain
from pokefly.rom import validate_rom
from pokefly.runner import run_directory, write_json


@dataclass
class WatchOptions:
    rom: Path
    device: str = "auto"
    seed: int = 64
    frames: int = 12
    brain_steps: int = 10
    hz: float = 5
    steps: int = 0
    port: int = 8777
    manual: bool = False
    intro: bool = False
    load_state: Path | None = None


def watch(options: WatchOptions) -> Path:
    if options.frames < 2 or options.brain_steps < 1 or options.steps < 0:
        raise ValueError("frames >= 2, brain-steps >= 1 and steps >= 0 required")
    if not 1 <= options.hz <= 30 or not 0 <= options.port <= 65535:
        raise ValueError("hz must be in [1,30] and port in [0,65535]")
    validate_rom(options.rom)
    if options.load_state and not options.load_state.is_file():
        raise FileNotFoundError(options.load_state)
    print("Loading the pixel-driven fly (frozen weights; no autonomous policy)...", flush=True)
    controller = PixelBrain(
        device=options.device, seed=options.seed, brain_steps=options.brain_steps
    )
    view = BrainView(controller)
    output = run_directory("vision-watch")
    view.static.update(
        {
            "run_id": output.name,
            "neural_seconds_per_sample": options.brain_steps * controller.brain.dt,
            "emulator_frames_per_sample": options.frames,
            "target_hz": options.hz,
            "device": controller.brain.device,
        }
    )
    actions = Counter()
    recent_actions = deque(maxlen=12)
    completed, reason = 0, "step_limit"
    with RedEmulator(options.rom) as game:
        bootstrap_actions = 0
        if options.load_state:
            game.load(options.load_state)
        elif options.intro:
            print("Running the requested scripted intro (not fly behavior).", flush=True)
            bootstrap_actions = game.bootstrap()
        else:
            game.tick(120)
        game.save(output / "start.state")
        game.screenshot(output / "start.png")
        write_json(
            output / "config.json",
            {
                **asdict(options),
                "rom_sha1": game.rom_sha1,
                "device_resolved": controller.brain.device,
                "retina": controller.retina.summary(),
                "bootstrap_actions": bootstrap_actions,
                "brain_weights_frozen": True,
                "internal_learning": False,
                "action_source": "manual-or-wait" if options.manual else "wait-only",
                "ram_use": "telemetry only; never passed to PixelBrain",
            },
        )
        with Dashboard(view.static, port=options.port, manual=options.manual) as dashboard:
            print(f"Dashboard: {dashboard.url}", flush=True)
            print(f"Run: {output}\nCtrl+C saves and stops. This mode is NOT learning.", flush=True)
            with (output / "trajectory.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                try:
                    while options.steps == 0 or completed < options.steps:
                        started = time.perf_counter()
                        action = dashboard.next_action()
                        if not game.act(action, options.frames):
                            reason = "emulator_stopped"
                            break
                        # The SAME post-action frame is displayed and supplied to the brain.
                        frame = game.screen()
                        observation = controller.observe(frame)
                        completed += 1
                        actions[action] += 1
                        if action != "wait":
                            recent_actions.append({"sample": completed, "action": action})
                        telemetry = game.state().telemetry()  # Human/logging only.
                        record = {
                            "run_id": output.name,
                            "sample": completed,
                            "monotonic_seconds": round(time.monotonic(), 6),
                            "emulator_frames": completed * options.frames,
                            "brain_seconds": completed * options.brain_steps * controller.brain.dt,
                            "action": action,
                            "action_source": "human" if action != "wait" else "idle",
                            "pulse_frames": options.frames - 1 if action != "wait" else 0,
                            "buttons_released": True,
                            "spikes_total": observation.spikes,
                            "groups": observation.groups,
                            "telemetry": telemetry,
                            "compute_ms": round((time.perf_counter() - started) * 1000, 1),
                        }
                        log.write(json.dumps(record) + "\n")
                        dashboard.publish(
                            {
                                **record,
                                **view.activity(observation),
                                "recent_actions": list(recent_actions),
                                "screen": png_data(frame),
                            }
                        )
                        if completed % 100 == 0:
                            game.save(output / "latest.state")
                            game.screenshot(output / "latest.png")
                        delay = 1 / options.hz - (time.perf_counter() - started)
                        if delay > 0:
                            time.sleep(delay)
                except KeyboardInterrupt:
                    reason = "interrupted"
                except Exception:
                    reason = "error"
                    raise
                finally:
                    game.save(output / "latest.state")
                    game.screenshot(output / "latest.png")
                    write_json(
                        output / "summary.json",
                        {
                            "reason": reason,
                            "samples": completed,
                            "actions": dict(actions),
                            "internal_learning": False,
                            "final_state": game.state().telemetry(),
                        },
                    )
    return output
