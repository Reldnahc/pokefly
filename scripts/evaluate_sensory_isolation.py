"""Matched isolation experiment: neural equivalence, visual checks, frozen gameplay.

This does not fit a decoder or choose thresholds from game performance. The sole
experimental difference is the user-approved incoming sensory-current gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from pokefly.circuit_eval import _patterns
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.internal_brain import InternalBrain
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


def host(value):
    return value if isinstance(value, np.ndarray) else value.get()


def check_promoted_gate(config, frame, device, seed):
    """The permanent setting must exactly reproduce the prior temporary intervention."""
    baseline = replace(config, dynamics=replace(config.dynamics, isolate_nonvisual_sensory=False))
    old = InternalBrain(device=device, seed=seed, config=baseline)
    new = InternalBrain(device=device, seed=seed, config=config)
    b = old.brain
    selected = np.char.find(b.superclass.astype(str), "sensory") >= 0
    selected[b.visual] = False
    indices = np.flatnonzero(selected)
    assert len(indices) == new.sensory_isolation["neurons"]
    np.testing.assert_array_equal(indices, new.hybrid.isolated_host)
    assert not np.isin(indices, np.concatenate(tuple(new.motors.values()))).any()
    original = old.hybrid.current

    def temporary_current(release):
        result = original(release)
        result[b.xp.asarray(indices), 0] = 0
        return result

    old.hybrid.current = temporary_current
    drive = old.retina.encode(frame)
    for _ in range(384):
        np.testing.assert_array_equal(b.step(eye_drive=drive), new.brain.step(eye_drive=drive))
    np.testing.assert_array_equal(host(b.v), host(new.brain.v))
    for key, value in old.hybrid.arrays().items():
        np.testing.assert_array_equal(host(value), host(new.hybrid.arrays()[key]))
    np.testing.assert_array_equal(b.weights, new.brain.weights)
    return {
        "neural_steps": 384,
        "all_spikes_voltages_release_adaptation_exact": True,
        "weights_unchanged": True,
        "motor_populations_excluded": True,
        "sensory_isolation": new.sensory_isolation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-state", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.trials < 2 or args.steps < 1 or not args.load_state.is_file():
        parser.error("Require a real starting state, >=2 trials and positive steps")
    rom, data = resolve_rom(None, Path.cwd()), configure_runtime()
    protected = [rom, data / "brain.npz", data / "weights.npz"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    config = load_config(Path("configs/sensory-isolated-v1.json"))
    output = run_directory("sensory-isolation-evaluation")
    with Image.open(args.load_state.with_suffix(".png")) as image:
        frame = np.asarray(image.convert("RGB"))
    report = {
        "scope": "Only nonvisual sensory incoming-current isolation differs; same fixed "
        "motor decoder, raw-pixel retina, noise, rewards and frozen initial weights.",
        "starting_state": str(args.load_state.resolve()),
        "state_sha256": hashlib.sha256(args.load_state.read_bytes()).hexdigest(),
        "steps_per_game_trial": args.steps,
        "promotion_equivalence": check_promoted_gate(config.brain, frame, args.device, args.seed),
        "visual_checks": [],
        "gameplay": [],
    }
    for isolated in (False, True):
        condition = "isolated" if isolated else "unisolated"
        value = replace(
            config,
            brain=replace(
                config.brain,
                dynamics=replace(config.brain.dynamics, isolate_nonvisual_sensory=isolated),
            ),
        )
        controller = InternalBrain(device=args.device, seed=args.seed, config=value.brain)
        for seed in range(args.seed, args.seed + 2):
            report["visual_checks"].append(
                {
                    "condition": condition,
                    "seed": seed,
                    "identity": controller.identity(),
                    **_patterns(controller, seed=seed, steps=384),
                }
            )
            write_json(output / "report.json", report)
        del controller
        for seed in range(args.seed, args.seed + args.trials):
            path = train(
                TrainOptions(
                    rom=rom,
                    load_state=args.load_state,
                    device=args.device,
                    seed=seed,
                    steps=args.steps,
                    mode="frozen",
                    dashboard=False,
                    hz=0,
                    checkpoint_every=0,
                ),
                config_override=value,
            )
            summary = json.loads((path / "summary.json").read_text())
            rows = [
                json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()
            ]
            report["gameplay"].append(
                {
                    "condition": condition,
                    "run": str(path),
                    **summary,
                    "forward_spikes": sum(row["groups"]["forward_Up"] for row in rows),
                }
            )
            write_json(output / "report.json", report)
            if summary["reason"] != "step_limit":
                raise RuntimeError("Partial experiment preserved; run did not finish")
    report["original_files_unchanged"] = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in hashes.items()
    )
    assert report["original_files_unchanged"]
    report["learned_gameplay_demonstrated"] = False
    report["interpretation"] = (
        "This is a controlled model simplification, not a validated intact fly. "
        "More spontaneous Up, tiles or actions does not establish meaningful vision or learning."
    )
    write_json(output / "report.json", report)
    print(f"Sensory isolation evaluation: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
