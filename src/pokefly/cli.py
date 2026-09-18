from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from pokefly.rom import resolve_rom
from pokefly.runtime import configure_runtime


def _doctor(probe: bool, device: str) -> int:
    data = configure_runtime()
    print(f"Python: {sys.version.split()[0]}")
    for package in ("pyboy", "flybrain", "numpy", "pillow"):
        print(f"{package}: {importlib.metadata.version(package)}")
    from flybrain import cuda_available, has_data

    print(f"CUDA detected: {cuda_available()}")
    print(f"Brain data: {data} ({'present' if has_data(data) else 'missing'})")
    if not has_data(data):
        print("Run 'pokefly download' to fetch the model.")
        return 1
    if probe:
        print("Legacy eight-feature probe; use 'vision-probe' to test raw-pixel input.")
        import numpy as np

        from pokefly.brain import FlyController

        controller = FlyController(device=device)
        results = []
        for pixel in (255, 0):
            controller.brain.reset(64)
            controller.trace.reset()
            controller.previous = None
            frame = np.full((144, 160, 4), pixel, dtype=np.uint8)
            for _ in range(8):
                features = controller.observe(frame)
            results.append(features[:-1])
        difference = float(np.linalg.norm(results[1] - results[0]))
        print(
            f"Brain: {controller.brain.n:,} neurons, {len(controller.brain.weights):,} connections"
        )
        print(f"Device: {controller.brain.device}; descending neurons: {len(controller.trace.idx)}")
        print(f"Same-seed dark/bright input difference at descending readout: {difference:.6f}")
        if difference <= 1e-6 or not np.isfinite(difference):
            raise RuntimeError("The neural readout failed the input-sensitivity smoke test")
    return 0


def _inspect(rom: Path, frames: int) -> int:
    from pokefly.emulator import RedEmulator
    from pokefly.runner import run_directory

    if frames < 1:
        raise ValueError("frames must be positive")
    with RedEmulator(rom) as game:
        if not game.tick(frames):
            raise RuntimeError("Emulator stopped during boot")
        output = run_directory("inspect")
        game.screenshot(output / "screen.png")
        print(f"ROM: {rom.name}")
        print(f"Cartridge: {game.pyboy.cartridge_title}")
        print(f"SHA-1: {game.rom_sha1}")
        print(f"Screen: {game.screen().shape}")
        print(json.dumps(game.state().telemetry()))
        print(f"Screenshot: {output / 'screen.png'}")
    return 0


def _prepare(rom: Path, *, manual: bool, load_state: Path | None) -> int:
    from pokefly.emulator import RedEmulator
    from pokefly.runner import run_directory

    output = run_directory("manual" if manual else "prepare")
    with RedEmulator(rom, headless=not manual, speed=1 if manual else 0) as game:
        if load_state:
            game.load(load_state)
        if manual:
            print("Arrows move; A=A, S=B, Enter=Start. Close the window or Ctrl+C to save.")
            try:
                while game.tick(1):
                    pass
            except KeyboardInterrupt:
                pass
        else:
            print(f"Scripted intro actions: {game.bootstrap()}")
        game.save(output / "start.state")
        game.screenshot(output / "start.png")
        print(json.dumps(game.state().telemetry()))
    print(f"Starting state: {output / 'start.state'}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pokefly")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser(
        "doctor", help="verify dependencies, data, and optional neural response"
    )
    doctor.add_argument(
        "--brain", action="store_true", help="run the legacy eight-feature neural input probe"
    )
    doctor.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    sub.add_parser("download", help="fetch and checksum the public connectome into fly-data/")
    sub.add_parser("vision-prepare", help="build the fixed 2-D pixel-to-photoreceptor map")
    probe = sub.add_parser("vision-probe", help="test pixel propagation with matched neural noise")
    probe.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    probe.add_argument("--seed", type=int, default=64)
    probe.add_argument("--steps", type=int, default=96, help="20ms steps per stimulus")
    probe.add_argument("--trials", type=int, default=2, help="number of paired-noise seeds")
    internal_probe = sub.add_parser(
        "internal-probe", help="test visual/motor/internal-weight capacity"
    )
    internal_probe.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    internal_probe.add_argument("--seed", type=int, default=64)
    internal_probe.add_argument("--steps", type=int, default=384, help="20ms steps per stimulus")
    internal_probe.add_argument("--trials", type=int, default=2)
    for name, description in (
        (
            "circuit-evaluate",
            "compare baseline/adaptive/hybrid dynamics and seven neural motor controls",
        ),
        (
            "association-evaluate",
            "ROM-free visual conditioning, retention, shuffled/frozen and reversal checks",
        ),
    ):
        command = sub.add_parser(name, help=description)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument("--config", type=Path, default=Path("configs/hybrid-v1.json"))
        command.add_argument("--seed", type=int, default=64)
        command.add_argument("--trials", type=int, default=2)
        command.add_argument("--steps", type=int, default=384 if name == "circuit-evaluate" else 96)
        if name == "association-evaluate":
            command.add_argument("--pairings", type=int, default=12)
    watch = sub.add_parser(
        "watch", help="pixel-input dashboard; observation/manual diagnostic only"
    )
    watch.add_argument("--rom")
    watch.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    watch.add_argument("--seed", type=int, default=64)
    watch.add_argument("--frames", type=int, default=12, help="Game Boy frames per sample")
    watch.add_argument("--brain-steps", type=int, default=10, help="20ms neural steps per sample")
    watch.add_argument("--hz", type=float, default=5, help="maximum samples/sec, 1 to 30")
    watch.add_argument("--steps", type=int, default=0, help="sample limit; 0 means until Ctrl+C")
    watch.add_argument("--port", type=int, default=8777, help="localhost port; 0 picks a free port")
    watch.add_argument("--manual", action="store_true", help="enable labeled human button pulses")
    watch.add_argument("--intro", action="store_true", help="explicitly script intro to bedroom")
    watch.add_argument("--load-state", type=Path)
    train = sub.add_parser("train", help="raw pixels + fixed motor mapping + INTERNAL plasticity")
    train.add_argument("--rom")
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--seed", type=int, default=64)
    train.add_argument(
        "--steps", type=int, default=0, help="additional decisions; default 0 = until Ctrl+C"
    )
    train.add_argument("--mode", choices=("learn", "frozen", "no-reward"), default="learn")
    train.add_argument(
        "--config", type=Path, help="JSON experiment parameters (provisional defaults)"
    )
    train.add_argument("--load-state", type=Path)
    train.add_argument("--intro", action="store_true", help="explicit scripted intro intervention")
    train.add_argument("--resume", type=Path, help="checkpoint directory or latest-checkpoint.json")
    train.add_argument("--weights", type=Path, help="retained weights only, fresh neural dynamics")
    train.add_argument("--no-dashboard", dest="dashboard", action="store_false")
    train.add_argument("--port", type=int, default=8777)
    train.add_argument("--hz", type=float, default=5, help="decision rate cap; 0 = unthrottled")
    train.add_argument("--checkpoint-every", type=int, default=500, help="0 saves only on exit")
    evaluate = sub.add_parser("evaluate", help="matched learning/frozen/no-reward trials")
    evaluate.add_argument("--rom")
    evaluate.add_argument("--load-state", type=Path, required=True)
    evaluate.add_argument("--trials", type=int, default=3)
    evaluate.add_argument("--steps", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=64)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--checkpoint", type=Path, help="also evaluate learned weights frozen")
    inspect = sub.add_parser("inspect", help="identify and boot the supplied ROM")
    inspect.add_argument("--rom")
    inspect.add_argument("--frames", type=int, default=120)
    for name, help_text in (
        ("prepare", "script the intro and save a bedroom starting state"),
        ("manual", "play with the keyboard and save a starting state on exit"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--rom")
        command.add_argument("--load-state", type=Path)
    play = sub.add_parser("play", help="legacy eight-feature/external-readout learning baseline")
    play.add_argument("--rom", help="defaults to the sole .gb in the project root or roms/")
    play.add_argument("--headless", action="store_true")
    play.add_argument("--steps", type=int, default=1000)
    play.add_argument("--speed", type=int, default=0, help="emulator speed; 0 means unlimited")
    play.add_argument("--frames", type=int, default=24, help="Game Boy frames per action")
    play.add_argument("--brain-steps", type=int, default=12, help="20ms neural steps per action")
    play.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    play.add_argument("--seed", type=int, default=64)
    play.add_argument(
        "--epsilon", type=float, default=0.2, help="probability of exploratory action"
    )
    play.add_argument("--no-learn", dest="learn", action="store_false")
    play.add_argument("--no-intro", dest="intro", action="store_false")
    play.add_argument("--load-state", type=Path)
    play.add_argument("--checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_runtime()
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args.brain, args.device)
        if args.command == "download":
            from flybrain import download

            print(f"Brain data: {download(configure_runtime())}")
            return 0
        if args.command == "vision-prepare":
            from pokefly.vision import build_retina

            print(json.dumps(build_retina(configure_runtime()).summary(), indent=2))
            return 0
        if args.command == "vision-probe":
            from pokefly.pixel_brain import probe_vision

            probe_vision(device=args.device, seed=args.seed, steps=args.steps, trials=args.trials)
            return 0
        if args.command == "internal-probe":
            from pokefly.internal_probe import probe_internal

            probe_internal(device=args.device, seed=args.seed, steps=args.steps, trials=args.trials)
            return 0
        if args.command in ("circuit-evaluate", "association-evaluate"):
            from pokefly.circuit_eval import evaluate_association, evaluate_circuits
            from pokefly.experiment import load_config

            options = {k: v for k, v in vars(args).items() if k not in ("command", "config")}
            function = (
                evaluate_circuits if args.command == "circuit-evaluate" else evaluate_association
            )
            function(config=load_config(args.config).brain, **options)
            return 0
        rom = resolve_rom(args.rom, Path.cwd())
        if args.command == "train":
            from pokefly.experiment import TrainOptions, train

            options = {k: v for k, v in vars(args).items() if k not in ("command", "rom")}
            print(f"Saved internal experiment: {train(TrainOptions(rom=rom, **options))}")
            return 0
        if args.command == "evaluate":
            from pokefly.experiment import evaluate

            options = {k: v for k, v in vars(args).items() if k not in ("command", "rom")}
            evaluate(rom=rom, **options)
            return 0
        if args.command == "watch":
            from pokefly.watch import WatchOptions, watch

            options = {k: v for k, v in vars(args).items() if k not in ("command", "rom")}
            output = watch(WatchOptions(rom=rom, **options))
            print(f"Saved pixel observation: {output}")
            return 0
        if args.command == "inspect":
            return _inspect(rom, args.frames)
        if args.command in ("prepare", "manual"):
            return _prepare(rom, manual=args.command == "manual", load_state=args.load_state)
        if args.command == "play":
            from pokefly.runner import RunOptions, play

            options = {k: v for k, v in vars(args).items() if k not in ("command", "rom")}
            output = play(RunOptions(rom=rom, **options))
            print(f"Saved experiment: {output}")
            return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
