"""Exploratory intrinsic-excitability calibration, not a gameplay controller.

Every non-sensory spiking cell gets the same rate-homeostasis equation. No motor
identity, game image, reward, or action enters calibration. Calibrated offsets
are frozen before evaluation. This is an engineering hypothesis, not measured
cell-specific physiology. Compare https://elifesciences.org/articles/45717 .
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from pokefly.actions import count_buttons
from pokefly.checkpoint import read_checkpoint
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--calibration-steps", type=int, default=10000)
    parser.add_argument("--target-hz", type=float, default=1.0)
    parser.add_argument("--noise-amplitude", type=float, default=0.22)
    parser.add_argument("--noise-hz", type=float, default=1.2)
    parser.add_argument("--synaptic-gain", type=float, default=3.0)
    parser.add_argument("--spike-temperature", type=float, default=0.0)
    parser.add_argument("--motor-adaptation-increment", type=float, default=0.0)
    parser.add_argument("--motor-adaptation-seconds", type=float, default=3.0)
    parser.add_argument(
        "--visual-model", choices=("legacy-v1", "calibrated-rate-v1"), default="legacy-v1"
    )
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument(
        "--reset-every",
        type=int,
        default=0,
        help="Optional neutral fitting resets, in neural steps; 0 preserves v1",
    )
    parser.add_argument(
        "--reset-probe-decisions",
        type=int,
        default=0,
        help="Frozen neutral-image decisions per independent reset seed",
    )
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.calibration_steps < 2 or not np.isfinite(args.target_hz) or args.target_hz <= 0:
        parser.error("Positive finite rate and at least two neural steps required")
    if args.reset_every < 0 or args.reset_probe_decisions < 0:
        parser.error("Reset intervals and probe lengths cannot be negative")
    if not args.calibration_only and args.source_run is None:
        parser.error("--source-run is required for the optional game-image probes")
    if args.export and args.export.exists():
        parser.error("Export target already exists; calibration files are immutable")
    output = run_directory("intrinsic-probe")
    cfg = load_config(Path("configs/sensory-isolated-v1.json")).brain
    cfg = replace(
        cfg, noise_amplitude=args.noise_amplitude, noise_hz=args.noise_hz,
        synaptic_gain=args.synaptic_gain,
    )
    cfg = replace(
        cfg,
        dynamics=replace(
            cfg.dynamics,
            spike_temperature=args.spike_temperature,
            motor_adaptation_increment=args.motor_adaptation_increment,
            motor_adaptation_seconds=args.motor_adaptation_seconds,
            visual_model=args.visual_model,
        ),
    )
    c = InternalBrain(device=args.device, config=cfg)
    b, h, xp = c.brain, c.hybrid, c.brain.xp
    bias = xp.zeros((b.n, 1), xp.float32)
    mask = np.ones(b.n, bool)
    mask[h.graded_host] = False
    mask[np.char.find(b.superclass.astype(str), "sensory") >= 0] = False
    allowed = xp.asarray(np.flatnonzero(mask))
    original_current = h.current

    def current(release):
        return original_current(release) + bias / b.gain

    h.current = current
    gray = np.full((144, 160, 3), 128, np.uint8)
    drive = c.retina.encode(gray)
    c.reset_dynamics(707)
    counts = np.zeros(b.n, np.int32)
    for step in range(args.calibration_steps):
        if args.reset_every and step and step % args.reset_every == 0:
            c.reset_dynamics(707 + step // args.reset_every)
        fired = b.step(eye_drive=drive)
        bias[allowed, 0] += np.float32(0.001 * args.target_hz * b.dt)
        bias[b.fired, 0] -= np.float32(0.001)
        bias[:] = xp.clip(bias, -0.14, 0.2)
        bias[xp.asarray(np.flatnonzero(~mask)), 0] = 0
        if step >= args.calibration_steps // 2:
            counts[fired] += 1
    calibrated = bias.copy()
    protocol = {
        "rule": (
            "uniform-neutral-rate-homeostasis-reset-v2"
            if args.reset_every
            else "uniform-neutral-rate-homeostasis-v1"
        ),
        "seed": 707,
        "neutral_gray": 128,
        "steps": args.calibration_steps,
        "target_hz": args.target_hz,
        "step_size": 0.001,
        "bounds": [-0.14, 0.2],
        "config": c.identity(),
    }
    if args.reset_every:
        protocol["reset_every_neural_steps"] = args.reset_every
        protocol["reset_seed_schedule"] = "707 + reset index"
    host_bias = bias if xp is np else bias.get()
    artifact = dict(
        bias=host_bias, mask=mask, body_ids=c.body_ids, protocol=np.array(json.dumps(protocol))
    )
    np.savez_compressed(output / "calibration.npz", **artifact)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open("xb") as destination:
            np.savez_compressed(destination, **artifact)
    report = {
        "protocol": "Uniform neural homeostasis on neutral gray, seed 707; no rewards, "
        "game frames or motor labels used to calibrate. Offsets frozen during probes.",
        "target_hz": args.target_hz,
        "calibration_steps": args.calibration_steps,
        "calibration_hz": {
            key: float(counts[idx].mean() / (args.calibration_steps / 2 * b.dt))
            for key, idx in c.groups.items()
        },
        "motor_bias": {key: host_bias[idx, 0].tolist() for key, idx in c.motors.items()},
        "rows": [],
    }
    if args.reset_probe_decisions:
        report["frozen_reset_probes"] = []
        for seed in (301, 302, 303):
            c.reset_dynamics(seed)
            probe_counts = np.zeros(b.n, np.int32)
            for _ in range(args.reset_probe_decisions):
                probe_counts += c.observe(gray).counts
            report["frozen_reset_probes"].append(
                {
                    "seed": seed,
                    "decisions": args.reset_probe_decisions,
                    "population_hz": {
                        key: float(
                            probe_counts[idx].mean()
                            / (args.reset_probe_decisions * c.brain_steps * b.dt)
                        )
                        for key, idx in c.groups.items()
                    },
                }
            )
    write_json(output / "report.json", report)
    if args.calibration_only:
        print("Calibration:", args.export or output / "calibration.npz", flush=True)
        return
    arrays, _ = read_checkpoint(args.source_run / "latest-checkpoint.json")
    frames = {**test_patterns(), "town": arrays["next_frame"]}
    for variant in ("original", "calibrated"):
        bias[:] = calibrated if variant == "calibrated" else 0
        for seed in (301, 302, 303):
            for name in ("black", "left", "right", "town"):
                c.reset_dynamics(seed)
                actions, counts = [], np.zeros(b.n, np.int32)
                for _ in range(256):
                    observation = c.observe(frames[name])
                    counts += observation.counts
                    actions.append(c.choose(observation)[0])
                row = {
                    "variant": variant,
                    "seed": seed,
                    "stimulus": name,
                    "actions": actions,
                    "buttons": count_buttons(Counter(actions)),
                    "population_hz": {
                        key: float(counts[idx].mean() / (256 * c.brain_steps * b.dt))
                        for key, idx in c.groups.items()
                    },
                }
                report["rows"].append(row)
                write_json(output / "report.json", report)
                print(variant, seed, name, row["buttons"], flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
