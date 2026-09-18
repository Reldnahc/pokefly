from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pokefly.brain import FlyController
from pokefly.emulator import ACTIONS, RedEmulator
from pokefly.policy import LinearReadout
from pokefly.red_state import ProgressReward


def run_directory(kind: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path.cwd() / "runs" / f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


@dataclass
class RunOptions:
    rom: Path
    steps: int = 1000
    headless: bool = False
    speed: int = 0
    device: str = "auto"
    seed: int = 64
    frames: int = 24
    brain_steps: int = 12
    epsilon: float = 0.2
    learn: bool = True
    intro: bool = True
    load_state: Path | None = None
    checkpoint: Path | None = None


def play(options: RunOptions) -> Path:
    if options.steps < 1 or options.frames < 2 or options.brain_steps < 1:
        raise ValueError("steps/brain-steps must be positive and frames must be at least 2")
    if options.speed < 0:
        raise ValueError("speed must be non-negative (0 means unlimited)")
    if not 0 <= options.epsilon <= 1:
        raise ValueError("epsilon must be between zero and one")
    # Validate local files before loading the model or creating output.
    from pokefly.rom import validate_rom

    validate_rom(options.rom)
    for path in (options.load_state, options.checkpoint):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    print("Loading the fly connectome...", flush=True)
    controller = FlyController(
        device=options.device, seed=options.seed, brain_steps=options.brain_steps
    )
    policy = LinearReadout(
        controller.feature_count,
        seed=options.seed,
        epsilon=options.epsilon,
        signature=controller.signature,
    )
    if options.checkpoint:
        policy.load(options.checkpoint)
    initial_weights = policy.weights.copy()
    output = run_directory("fly")
    print(
        f"{controller.brain.n:,} neurons / {len(controller.brain.weights):,} connections / "
        f"{len(controller.trace.idx):,} outputs on {controller.brain.device}",
        flush=True,
    )
    print(f"Run: {output}", flush=True)
    reward_tracker = ProgressReward()
    actions: Counter[str] = Counter()
    exploratory_count = 0
    total_reward = 0.0
    completed = 0
    started_at = time.perf_counter()
    reason = "step_limit"
    with RedEmulator(options.rom, headless=options.headless, speed=options.speed) as game:
        bootstrap_actions = 0
        if options.load_state:
            game.load(options.load_state)
        elif options.intro:
            bootstrap_actions = game.bootstrap()
        else:
            game.tick(120)
        write_json(
            output / "config.json",
            {
                **asdict(options),
                "device_resolved": controller.brain.device,
                "rom_sha1": game.rom_sha1,
                "encoder_signature": controller.signature,
                "bootstrap_actions": bootstrap_actions,
                "brain_weights_frozen": True,
            },
        )
        game.save(output / "start.state")
        game.screenshot(output / "start.png")
        reward_tracker.observe(game.state())
        features = controller.observe(game.screen())
        action_index, exploratory = policy.choose(features)
        with (output / "trajectory.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            try:
                for step in range(1, options.steps + 1):
                    action = ACTIONS[action_index]
                    before = game.state().telemetry()
                    # These are the neural observations that selected THIS action.
                    neural = {
                        "spikes": controller.last_spikes,
                        "descending_spikes": controller.last_dn_spikes,
                        "drives": controller.last_drives.tolist(),
                    }
                    running = game.act(action, options.frames)
                    state = game.state()
                    reward, reward_parts = reward_tracker.observe(state)
                    next_features = controller.observe(game.screen())
                    next_action, next_exploratory = policy.choose(next_features)
                    error = None
                    if options.learn:
                        error = policy.learn(
                            features,
                            action_index,
                            reward,
                            next_features,
                            next_action,
                            terminal=not running,
                        )
                    actions[action] += 1
                    exploratory_count += int(exploratory)
                    total_reward += reward
                    completed = step
                    log.write(
                        json.dumps(
                            {
                                "step": step,
                                "action": action,
                                "exploratory": exploratory,
                                "before": before,
                                "after": state.telemetry(),
                                "reward": reward,
                                "reward_parts": reward_parts,
                                "td_error": error,
                                **neural,
                            }
                        )
                        + "\n"
                    )
                    if step % 100 == 0 or step == options.steps:
                        print(
                            f"step={step} map={state.map_id} pos=({state.x},{state.y}) "
                            f"tiles={len(reward_tracker.tiles)} reward={total_reward:.2f} "
                            f"action={action}",
                            flush=True,
                        )
                        game.screenshot(output / "latest.png")
                        game.save(output / "latest.state")
                        policy.save(output / "readout.npz")
                    features, action_index, exploratory = (
                        next_features,
                        next_action,
                        next_exploratory,
                    )
                    if not running:
                        reason = "window_closed"
                        break
            except KeyboardInterrupt:
                reason = "interrupted"
                print("Stopping and saving the current experiment...", flush=True)
            except Exception:
                reason = "error"
                raise
            finally:
                # Each run owns its output folder; the supplied ROM/save is never overwritten.
                game.save(output / "latest.state")
                game.screenshot(output / "latest.png")
                policy.save(output / "readout.npz")
                elapsed = time.perf_counter() - started_at
                summary = {
                    "reason": reason,
                    "steps": completed,
                    "wall_seconds": round(elapsed, 3),
                    "steps_per_second": round(completed / max(elapsed, 0.001), 2),
                    "total_reward": round(total_reward, 4),
                    "unique_tiles": len(reward_tracker.tiles),
                    "maps": sorted(reward_tracker.maps),
                    "action_counts": dict(actions),
                    "exploratory_actions": exploratory_count,
                    "readout_actions": completed - exploratory_count,
                    "readout_updates": policy.updates,
                    "weight_change_l2": float(np.linalg.norm(policy.weights - initial_weights)),
                    "final_state": game.state().telemetry(),
                }
                write_json(output / "summary.json", summary)
                print(json.dumps(summary, indent=2), flush=True)
    return output
