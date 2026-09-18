"""Autonomous internal-learning experiment; RAM is isolated in the outcome observer."""

from __future__ import annotations

import json
import signal
import time
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pokefly.actions import count_buttons, pressed_buttons
from pokefly.checkpoint import read_checkpoint, save_checkpoint, versions
from pokefly.dashboard import BrainView, Dashboard, png_data
from pokefly.dynamics import DynamicsConfig
from pokefly.emulator import BUTTON_TIMINGS, RedEmulator, button_phases
from pokefly.internal_brain import BrainConfig, InternalBrain
from pokefly.motors import MOTOR_MAPPING_VERSION, MotorConfig
from pokefly.pacing import RuntimePacing, SimulationRate
from pokefly.plasticity import PlasticityConfig
from pokefly.rewards import GeneralRewards, RewardConfig, RewardHooks
from pokefly.rom import validate_rom
from pokefly.runner import run_directory, write_json
from pokefly.temporal import MODES, TemporalVision


@dataclass(frozen=True)
class ExperimentConfig:
    brain: BrainConfig = field(default_factory=BrainConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    frames: int = 24
    visual_timing: str = "snapshot-v1"
    button_timing: str = "simultaneous-v1"

    def __post_init__(self):
        if not 2 <= self.frames <= 120:
            raise ValueError("frames must be in [2,120]")
        if self.visual_timing not in MODES:
            raise ValueError("Unknown visual timing")
        if self.button_timing not in BUTTON_TIMINGS:
            raise ValueError("Unknown button timing")
        if self.button_timing == "serial-v2" and self.frames < 4:
            raise ValueError("Serial button timing requires at least four frames")
        if self.visual_timing == "stream-v1" and self.brain.brain_steps > self.frames:
            raise ValueError("Stream mode needs at least one game frame per neural step")

    @classmethod
    def from_dict(cls, data: dict, *, checkpoint: bool = False):
        data = dict(data)
        neural = dict(data.pop("brain", {}))
        neural["plasticity"] = PlasticityConfig(**neural.get("plasticity", {}))
        motor = dict(neural.get("motor", {}))
        if checkpoint:
            # Missing version means the historical single-winner decoder, not a migration.
            motor.setdefault("arbitration", "exclusive-v1")
        neural["motor"] = MotorConfig(**motor)
        neural["dynamics"] = DynamicsConfig(**neural.get("dynamics", {}))
        return cls(
            brain=BrainConfig(**neural), rewards=RewardConfig(**data.pop("rewards", {})), **data
        )


@dataclass
class TrainOptions:
    rom: Path
    device: str = "auto"
    seed: int = 64
    steps: int = 0
    mode: str = "learn"
    config: Path | None = None
    load_state: Path | None = None
    intro: bool = False
    resume: Path | None = None
    weights: Path | None = None
    dashboard: bool = True
    port: int = 8777
    hz: float = 5.0
    checkpoint_every: int = 500


class StopAtBoundary:
    """Defer Ctrl+C until a complete neural/action/reward transaction is saved."""

    def __init__(self):
        self.requested = False

    def __enter__(self):
        self.previous = signal.signal(signal.SIGINT, self.request)
        return self

    def request(self, *_args):
        self.requested = True

    def __exit__(self, *_args):
        signal.signal(signal.SIGINT, self.previous)


def load_config(path: Path | None) -> ExperimentConfig:
    if not path:
        return ExperimentConfig()
    try:
        return ExperimentConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid experiment configuration: {exc}") from exc


def train(options: TrainOptions, *, config_override: ExperimentConfig | None = None) -> Path:
    if options.steps < 0 or options.checkpoint_every < 0:
        raise ValueError("steps and checkpoint-every cannot be negative")
    if options.mode not in ("learn", "frozen", "no-reward"):
        raise ValueError("mode must be learn, frozen or no-reward")
    pacing = RuntimePacing(options.hz)
    if not 0 <= options.port <= 65535:
        raise ValueError("port must be in [0,65535]")
    if sum(bool(x) for x in (options.intro, options.load_state, options.resume)) > 1:
        raise ValueError("Choose just one of intro, load-state and resume")
    if options.resume and (options.weights or options.config or config_override):
        raise ValueError("Resume restores its own configuration and weights")
    if options.weights and options.config:
        raise ValueError("A retention run uses the checkpoint's fixed configuration")
    rom_sha1 = validate_rom(options.rom)
    loaded = (
        read_checkpoint(options.resume or options.weights)
        if (options.resume or options.weights)
        else None
    )
    config = config_override or load_config(options.config)
    if loaded:
        arrays, saved = loaded
        if saved["rom_sha1"] != rom_sha1:
            raise ValueError("Checkpoint ROM mismatch")
        config = ExperimentConfig.from_dict(saved["experiment"]["config"], checkpoint=True)
        if options.resume:
            if saved["versions"] != versions():
                raise ValueError("Exact resume requires the same dependency versions")
            if saved["experiment"]["mode"] != options.mode:
                raise ValueError(
                    "Resume mode differs; use --weights for a separate retention trial"
                )
    print(f"Loading internally plastic fly; mode={options.mode} (experimental)...", flush=True)
    controller = InternalBrain(device=options.device, seed=options.seed, config=config.brain)
    if loaded:
        controller.restore(arrays, saved["neural"], weights_only=not bool(options.resume))
    output = run_directory(f"internal-{options.mode}")
    rewards = GeneralRewards(config.rewards)
    completed, local_steps, reason = 0, 0, "step_limit"
    actions, spike_totals = Counter(), Counter()
    recent = deque(maxlen=12)
    visited_this_trial: set[tuple[int, int, int]] = set()
    first_house_exit = None
    checkpoint = None
    if options.resume:
        rewards.restore(saved["rewards"])
        progress = saved["experiment"]
        completed = int(progress["sample"])
        actions.update(progress["actions"])
        spike_totals.update(progress["spike_totals"])
        visited_this_trial = {tuple(tile) for tile in progress["visited_this_trial"]}
        first_house_exit = progress["first_house_exit"]
    with RedEmulator(options.rom, button_timing=config.button_timing) as game:
        bootstrap_actions = 0
        if options.resume:
            game.load(
                Path(saved["directory"]) / "game.state", advance=False, prime_renderer=True
            )
            frame = arrays["next_frame"].copy()
        else:
            if options.load_state:
                game.load(options.load_state)
            elif options.intro:
                print("Explicit scripted intro: intervention, not fly behavior.", flush=True)
                bootstrap_actions = game.bootstrap()
            else:
                game.tick(120)
            frame = game.screen()
            rewards.baseline(game.state())
            # A mid-battle starting state cannot retrospectively invent a win/catch.
            if game.state().battle:
                rewards.start(game.pyboy.memory)
        game.save(output / "start.state")
        game.screenshot(output / "start.png")
        start_state = game.state()
        starts_in_house = start_state.started and start_state.map_id in (37, 38)
        if options.resume:
            starts_in_house = saved["experiment"]["starts_in_house"]
        temporal = TemporalVision(
            controller,
            config.visual_timing,
            arrays=arrays if options.resume else None,
            state=saved["experiment"].get("temporal") if options.resume else None,
        )
        if config.visual_timing != "snapshot-v1":
            if options.resume and saved["experiment"].get("temporal") is None:
                raise ValueError("Temporal resume requires the saved pending neural window")
            temporal.observe(frame)  # One initial static window; restored on exact resume.
        write_json(
            output / "config.json",
            {
                "options": asdict(options),
                "config": asdict(config),
                "rom_sha1": rom_sha1,
                "neural": controller.identity(),
                "versions": versions(),
                "motor_mapping": MOTOR_MAPPING_VERSION,
                "motor_arbitration": config.brain.motor.arbitration,
                "sensory_isolation": controller.sensory_isolation,
                "bootstrap_actions": bootstrap_actions,
                "ram_use": "reward observer and measurements only; no action/sensory features",
                "learning_claim": "experimental rule, not demonstrated gameplay improvement",
            },
        )
        view = BrainView(controller) if options.dashboard else None
        if view:
            view.static.update(
                {
                    "run_id": output.name,
                    "controller": "internal",
                    "mode": options.mode,
                    "internal_learning": options.mode == "learn",
                    "autonomous": True,
                    "neural_seconds_per_sample": config.brain.brain_steps * controller.brain.dt,
                    "emulator_frames_per_sample": config.frames,
                    "target_hz": options.hz,
                    "device": controller.brain.device,
                    "screen_timing": (
                        "pre-action frame that produced the displayed neural decision"
                        if config.visual_timing == "snapshot-v1"
                        else "last image of the preceding neural input window; displayed "
                        "activity integrates that window and selects the following pulse"
                    ),
                    "visual_timing": config.visual_timing,
                    "plastic_edges": len(controller.plasticity.base),
                    "motor_arbitration": config.brain.motor.arbitration,
                    "button_timing": config.button_timing,
                }
            )
        display = (
            Dashboard(view.static, port=options.port, pacing=pacing) if view else nullcontext(None)
        )
        with RewardHooks(game, rewards) as hooks, display as dashboard, StopAtBoundary() as stop:
            if dashboard:
                print(f"Dashboard: {dashboard.url}", flush=True)
            print(f"Run: {output}\nCtrl+C checkpoints and stops.", flush=True)

            def snapshot():
                return save_checkpoint(
                    output,
                    controller,
                    game,
                    hooks,
                    rewards,
                    frame=frame,
                    temporal=temporal,
                    experiment={
                        "sample": completed,
                        "mode": options.mode,
                        "config": asdict(config),
                        "actions": dict(actions),
                        "spike_totals": dict(spike_totals),
                        "visited_this_trial": sorted(visited_this_trial),
                        "first_house_exit": first_house_exit,
                        "starts_in_house": starts_in_house,
                        **({"temporal": temporal.state()} if temporal.state() else {}),
                    },
                )

            with (output / "trajectory.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                at_boundary = True
                rate = SimulationRate(completed * config.frames, config.frames)
                try:
                    while options.steps == 0 or local_steps < options.steps:
                        if stop.requested:
                            reason = "interrupted"
                            break
                        at_boundary = False
                        clock_start = time.perf_counter()
                        input_frame = frame
                        observation = temporal.observe(input_frame)  # PIXELS ONLY
                        input_window = temporal.window
                        action, rates = controller.choose(observation)  # NEURONS ONLY
                        buttons = list(pressed_buttons(action))
                        if not temporal.advance(game, action, config.frames):
                            reason = "emulator_stopped"
                            break
                        measured_reward, events = rewards.drain()  # RAM stays on this side.
                        delivered_reward = 0.0 if options.mode == "no-reward" else measured_reward
                        learning = controller.reinforce(
                            delivered_reward, enabled=options.mode != "frozen"
                        )
                        frame = game.screen()
                        completed += 1
                        local_steps += 1
                        actions[action] += 1
                        spike_totals.update(observation.groups)
                        if action != "wait":
                            recent.append(
                                {"sample": completed, "action": action, "buttons": buttons}
                            )
                        state = game.state()  # EVALUATION/LOGGING ONLY
                        if state.started and not state.battle:
                            visited_this_trial.add(state.position)
                            if starts_in_house and state.map_id == 0 and first_house_exit is None:
                                first_house_exit = completed  # NO corresponding reward term.
                        record = {
                            "run_id": output.name,
                            "sample": completed,
                            "brain_seconds": controller.brain.steps * controller.brain.dt,
                            "emulator_frames": completed * config.frames,
                            "action": action,
                            "buttons": buttons,
                            "action_source": "fly",
                            "motor_rates_hz": rates,
                            "pulse_frames": config.frames - 1 if action != "wait" else 0,
                            "buttons_released": True,
                            "spikes_total": observation.spikes,
                            "groups": observation.groups,
                            "graded_mean": observation.continuous["graded_mean"]
                            if observation.continuous
                            else None,
                            "telemetry": state.telemetry(),
                            "reward": measured_reward,
                            "delivered_reward": delivered_reward,
                            "reward_events": events,
                            "learning": learning,
                            "compute_ms": round((time.perf_counter() - clock_start) * 1000, 1),
                            "pacing": {
                                **pacing.state(),
                                **rate.record(completed * config.frames),
                            },
                            **(
                                {
                                    "visual_timing": config.visual_timing,
                                    "input_window": input_window,
                                }
                                if input_window is not None
                                else {}
                            ),
                        }
                        if config.button_timing != "simultaneous-v1":
                            phases = button_phases(action, config.frames, config.button_timing)
                            record.update(
                                button_timing=config.button_timing,
                                button_phases=[
                                    {"buttons": list(keys), "frames": duration}
                                    for keys, duration in phases
                                ],
                                pulse_frames=sum(duration for keys, duration in phases if keys),
                            )
                        at_boundary = True
                        log.write(json.dumps(record, allow_nan=False) + "\n")
                        if dashboard:
                            dashboard.publish(
                                {
                                    **record,
                                    **view.activity(observation),
                                    "screen": png_data(input_frame),
                                    "recent_actions": list(recent),
                                }
                            )
                        if options.checkpoint_every and completed % options.checkpoint_every == 0:
                            checkpoint = snapshot()
                            print(
                                f"step={completed} tiles={len(visited_this_trial)} "
                                f"reward={rewards.total:.3f} "
                                f"changed_edges={learning['changed_edges']}",
                                flush=True,
                            )
                        pacing.wait(clock_start, stopped=lambda: stop.requested)
                except KeyboardInterrupt:
                    reason = "interrupted"
                except Exception:
                    reason = "error"
                    raise
                finally:
                    if at_boundary:
                        checkpoint = snapshot()
                    game.screenshot(output / "latest.png")
                    summary = {
                        "reason": reason,
                        "mode": options.mode,
                        "seed": options.seed,
                        "samples": completed,
                        "new_samples": local_steps,
                        "pacing": pacing.state(),
                        "actions": dict(actions),
                        "button_counts": count_buttons(actions),
                        "reward": rewards.total,
                        "reward_counts": dict(rewards.counts),
                        "tiles": len(visited_this_trial),
                        "maps": sorted({tile[0] for tile in visited_this_trial}),
                        "starts_in_house": starts_in_house,
                        "first_house_exit": first_house_exit,
                        "house_exit": first_house_exit is not None,
                        "plasticity": controller.plasticity.metrics(),
                        "population_mean_hz": {
                            key: spike_totals[key]
                            / max(1, len(idx))
                            / max(controller.brain.steps * controller.brain.dt, 1e-12)
                            for key, idx in controller.groups.items()
                        },
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "checkpoint_at_consistent_boundary": at_boundary,
                        "final_state": game.state().telemetry(),
                        "learned_gameplay_demonstrated": False,
                        "note": "A run measures behavior, not evidence of causal improvement.",
                        "purpose": "Participation/perception/action/learning checks, not mastery",
                        "dynamics_profile": config.brain.dynamics.profile,
                        "sensory_isolation": controller.sensory_isolation,
                        "motor_arbitration": config.brain.motor.arbitration,
                        "button_timing": config.button_timing,
                        "button_coverage": sum(v > 0 for v in count_buttons(actions).values()),
                        "non_wait_fraction": 1 - actions.get("wait", 0) / max(1, completed),
                    }
                    write_json(output / "summary.json", summary)
                    print(
                        json.dumps(
                            {k: summary[k] for k in ("samples", "tiles", "house_exit", "actions")}
                        ),
                        flush=True,
                    )
    return output


def evaluate(
    *,
    rom: Path,
    load_state: Path,
    trials=3,
    steps=1000,
    seed=64,
    device="auto",
    config: Path | None = None,
    checkpoint: Path | None = None,
) -> Path:
    """Matched neural seeds, same game state, complete outcome-category rewards.

    Three fresh controls per seed; optional retained weights tested frozen against
    the same baseline. No human interventions or reward for the evaluation goal.
    """
    if trials < 2 or steps < 1:
        raise ValueError("Evaluation requires at least two trials and a positive step budget")
    if not load_state.is_file():
        raise FileNotFoundError(load_state)
    config_value = load_config(config)
    if checkpoint:
        _, saved = read_checkpoint(checkpoint)
        if config:
            raise ValueError("Retention evaluation uses the checkpoint configuration")
        config_value = ExperimentConfig.from_dict(saved["experiment"]["config"], checkpoint=True)
    output = run_directory("evaluation")
    records = []
    conditions = [
        ("learning", "learn", None),
        ("frozen", "frozen", None),
        ("absent_reward", "no-reward", None),
    ]
    if checkpoint:
        conditions.append(("retained_frozen", "frozen", checkpoint))
    for trial in range(trials):
        for condition, mode, weights in conditions:
            path = train(
                TrainOptions(
                    rom=rom,
                    device=device,
                    seed=seed + trial,
                    steps=steps,
                    mode=mode,
                    load_state=load_state,
                    weights=weights,
                    dashboard=False,
                    hz=0,
                    checkpoint_every=0,
                ),
                config_override=config_value,
            )
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            records.append({"condition": condition, "run": str(path), **summary})
            write_json(output / "trials.json", records)
            if summary["reason"] != "step_limit":
                raise RuntimeError(f"Evaluation interrupted; partial results saved in {output}")
    aggregate = {}
    for condition, _, _ in conditions:
        rows = [row for row in records if row["condition"] == condition]
        aggregate[condition] = {
            "trials": len(rows),
            "house_exits": sum(row["house_exit"] for row in rows),
            "mean_tiles": sum(row["tiles"] for row in rows) / len(rows),
            "mean_reward": sum(row["reward"] for row in rows) / len(rows),
            "battle_wins": sum(row["reward_counts"].get("battle_win", 0) for row in rows),
            "captures": sum(row["reward_counts"].get("capture", 0) for row in rows),
        }
    write_json(
        output / "report.json",
        {
            "conditions": aggregate,
            "seeds": list(range(seed, seed + trials)),
            "steps_per_trial": steps,
            "starting_state": str(load_state.resolve()),
            "learned_gameplay_demonstrated": False,
            "interpretation": "Small matched-control experiment; inspect effects and saturation. "
            "Identical/no-progress outcomes are negative results, not learning success.",
        },
    )
    print(f"Evaluation report: {output / 'report.json'}", flush=True)
    return output
