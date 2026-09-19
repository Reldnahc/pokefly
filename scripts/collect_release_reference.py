"""Collect frozen neural release means on generic images, with NO learning/game.

Copies existing intrinsic offsets exactly into a new immutable artifact. It
never loads game states, chooses buttons, uses rewards, or exports synapses.
The resulting reference is an engineering constraint, not measured physiology.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.release_reference import (
    INTRINSIC_KEYS,
    REFERENCE_PLASTICITY,
    REFERENCE_RULE,
    intrinsic_payload_sha256,
    load_release_reference,
    physical_config,
    source_data_hashes,
    verify_intrinsic_copy,
)
from pokefly.runner import run_directory, write_json
from pokefly.runtime import configure_runtime


def collect(controller, *, cycles, warmup, scored_steps, seed):
    """Measure physical transmitter release; neither plasticity nor decoder called."""
    patterns = {"gray": np.full((144, 160, 3), 128, np.uint8), **test_patterns()}
    if min(cycles, scored_steps) < 1 or warmup < 0:
        raise ValueError("Positive cycles/scored steps and nonnegative warmup required")
    controller.reset_dynamics(seed)
    brain, hybrid = controller.brain, controller.hybrid
    total = brain.xp.zeros(brain.n, brain.xp.float64)
    schedule, samples = [], 0
    order_rng = np.random.default_rng(seed + 1)
    for cycle in range(cycles):
        for name in order_rng.permutation(list(patterns)):
            drive = controller.retina.encode(patterns[name])
            for _ in range(warmup):
                brain.step(eye_drive=drive)
            for _ in range(scored_steps):
                brain.step(eye_drive=drive)
                total += hybrid.release
                samples += 1
            schedule.append({"cycle": cycle, "stimulus": str(name)})
        print(f"reference cycle={cycle + 1}/{cycles} scored_neural_steps={samples}", flush=True)
    mean = (total / samples).astype(brain.xp.float32)
    return mean if brain.xp is np else mean.get(), schedule, samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/visual-release-wide-v3.json"))
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=7201)
    p.add_argument("--cycles", type=int, default=4)
    p.add_argument("--warmup", type=int, default=128)
    p.add_argument("--scored-steps", type=int, default=512)
    args = p.parse_args()
    if args.export.exists():
        p.error("Reference exports are immutable; choose a new nonexistent file")
    if min(args.cycles, args.scored_steps) < 1 or args.warmup < 0:
        p.error("Invalid collection budget")
    config = load_config(args.config).brain
    if not config.intrinsic_calibration or config.plasticity.rule == REFERENCE_PLASTICITY:
        p.error("Use an existing unanchored intrinsic model, initialized with original synapses")
    source = Path(config.intrinsic_calibration)
    source_sha = sha256(source)
    with np.load(source, allow_pickle=False) as original:
        if "reference_protocol" in original:
            p.error("Do not chain augmented reference artifacts; copy the original calibration")
        copied = {key: original[key].copy() for key in INTRINSIC_KEYS}
    data = configure_runtime()
    data_hashes = source_data_hashes(data, config)
    output = run_directory("generic-release-reference")
    print("Collection:", output, flush=True)
    report = {"scope": __doc__, "status": "running", "source_config": str(args.config),
              "source_intrinsic_sha256": source_sha, "export": str(args.export),
              "game_frames_used": False, "rewards_used": False, "actions_selected": 0}
    write_json(output / "report.json", report)
    controller = InternalBrain(device=args.device, seed=args.seed, config=config)
    original_weights = controller.brain.weights.copy()
    mean, schedule, samples = collect(
        controller, cycles=args.cycles, warmup=args.warmup,
        scored_steps=args.scored_steps, seed=args.seed,
    )
    np.testing.assert_array_equal(controller.brain.weights, original_weights)
    np.testing.assert_array_equal(controller.plasticity.weights, controller.plasticity.base)
    bias = controller.hybrid.intrinsic_bias
    np.testing.assert_array_equal(bias if controller.brain.xp is np else bias.get(), copied["bias"])
    if sha256(source) != source_sha or source_data_hashes(data, config) != data_hashes:
        raise ValueError("An original calibration/connectome asset changed")
    if (mean.shape != (controller.brain.n,) or not np.isfinite(mean).all()
            or (mean < 0).any() or (mean > 1).any() or not (mean > 0).any()):
        raise ValueError("Collected release must be finite and in physical [0,1] units")
    protocol = {
        "rule": REFERENCE_RULE, "seed": args.seed, "cycles": args.cycles,
        "warmup_per_stimulus": args.warmup, "scored_steps_per_stimulus": args.scored_steps,
        "scored_neural_steps": samples, "neural_dt_seconds": controller.brain.dt,
        "schedule": schedule, "units": "per-step physical transmitter release in [0,1]",
        "physical_config": physical_config(config), "source_intrinsic_sha256": source_sha,
        "intrinsic_payload_sha256": intrinsic_payload_sha256(copied),
        "data_sha256": data_hashes, "game_frames_used": False,
        "rewards_used": False, "actions_selected": 0,
    }
    args.export.parent.mkdir(parents=True, exist_ok=True)
    with args.export.open("xb") as destination:
        np.savez_compressed(destination, **copied, mean_release=mean,
                            reference_protocol=np.array(json.dumps(protocol)))
    loaded, _ = load_release_reference(args.export, controller.body_ids, config, data=data)
    np.testing.assert_array_equal(loaded, mean)
    report["intrinsic_copy"] = verify_intrinsic_copy(source, args.export)
    report.update(status="completed", protocol=protocol, export_sha256=sha256(args.export),
                  intrinsic_biases_unchanged=True, original_synapses_unchanged=True,
                  decoder_never_called=True, mean_min=float(mean.min()), mean_max=float(mean.max()))
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
