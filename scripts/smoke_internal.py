"""Opt-in real-ROM/GPU internal-controller and exact-resume regression test.

No gameplay success claim. Scripted intro is explicitly limited to test setup.
Battle/capture cases are covered by unit fixtures, not claimed as live victories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from pokefly.checkpoint import read_checkpoint
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.internal_brain import InternalBrain
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def rows(path):
    return [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    rom = resolve_rom(None, Path.cwd())
    before_hash = hashlib.sha1(rom.read_bytes()).hexdigest()
    output = run_directory("internal-smoke")
    initial = train(
        TrainOptions(
            rom=rom,
            device=args.device,
            intro=True,
            steps=70,
            dashboard=False,
            hz=0,
            checkpoint_every=50,
            config=args.config,
        )
    )
    checkpoint = next((initial / "checkpoints").glob("step-00000050-*"))
    resumed = train(
        TrainOptions(
            rom=rom,
            device=args.device,
            resume=checkpoint,
            steps=20,
            dashboard=False,
            hz=30,  # A different wall-clock rate must not change the continued experiment.
            checkpoint_every=0,
        )
    )
    expected = rows(initial)[50:]
    actual = rows(resumed)
    # Wall-clock pacing/timing is not part of the deterministic neural experiment.
    ignored = {"run_id", "compute_ms", "pacing"}
    for index, (first, second) in enumerate(zip(expected, actual, strict=True), 51):
        assert {k: v for k, v in first.items() if k not in ignored} == {
            k: v for k, v in second.items() if k not in ignored
        }, f"Resume diverged at sample {index}"
    first_arrays, first_state = read_checkpoint(initial / "latest-checkpoint.json")
    second_arrays, second_state = read_checkpoint(resumed / "latest-checkpoint.json")
    for key, array in first_arrays.items():
        np.testing.assert_array_equal(array, second_arrays[key], err_msg=key)
    assert first_state["rewards"] == second_state["rewards"]
    assert first_state["neural"] == second_state["neural"]
    assert np.count_nonzero(first_arrays["weights"] != first_arrays["base"]) > 0
    controls = {}
    for mode in ("frozen", "no-reward"):
        path = train(
            TrainOptions(
                rom=rom,
                device=args.device,
                intro=True,
                steps=70,
                mode=mode,
                config=args.config,
                dashboard=False,
                hz=0,
                checkpoint_every=0,
            )
        )
        arrays, _ = read_checkpoint(path / "latest-checkpoint.json")
        np.testing.assert_array_equal(arrays["weights"], arrays["base"])
        controls[mode] = str(path)
    controller = InternalBrain(device=args.device, config=load_config(args.config).brain)
    controller.restore(first_arrays, first_state["neural"], weights_only=True)
    np.testing.assert_array_equal(controller.plasticity.weights, first_arrays["weights"])
    b = controller.brain
    inputs = np.arange(0, b.n, 13, dtype=np.int64)
    reference_spikes = np.zeros(b.n, np.float32)
    reference_spikes[inputs] = 1
    matrix = sparse.csc_matrix((b.weights, b.indices, b.indptr), shape=(b.n, b.n))
    reference = matrix @ reference_spikes
    measured = b.synaptic_input(b.xp.asarray(inputs))[:, 0]
    measured = measured if isinstance(measured, np.ndarray) else measured.get()
    np.testing.assert_allclose(measured, reference, atol=2e-6, rtol=2e-5)
    if controller.hybrid:
        graded = np.linspace(0, 1, b.n, dtype=np.float32)
        reference = matrix @ graded
        reference[controller.hybrid.isolated_host] = 0
        measured = controller.hybrid.current(b.xp.asarray(graded))[:, 0]
        measured = measured if isinstance(measured, np.ndarray) else measured.get()
        np.testing.assert_allclose(measured, reference, atol=2e-6, rtol=2e-5)
    # Test that stored plastic weights actually participate in the simulator's current.
    p = controller.plasticity
    edge = int(np.argmax(np.abs(p.weights - p.base)))
    pre = controller.brain.xp.asarray([int(p.pre[edge])])
    learned = controller.brain.synaptic_input(pre)
    p.weights[:] = p.base
    controller.sync_weights()
    unlearned = controller.brain.synaptic_input(pre)
    delta = float((learned - unlearned)[int(p.post[edge]), 0])
    assert abs(delta) > 0, "Internal weight changes did not reach the simulator"
    assert hashlib.sha1(rom.read_bytes()).hexdigest() == before_hash
    result = {
        "config": str(args.config) if args.config else "baseline",
        "sensory_isolation": controller.sensory_isolation,
        "uninterrupted": str(initial),
        "resumed": str(resumed),
        "controls": controls,
        "exact_20_decision_resume": True,
        "resume_matches_across_wall_clock_rates": True,
        "same_final_neural_arrays_and_rewards": True,
        "frozen_and_absent_reward_weights_unchanged": True,
        "retained_internal_weights_affect_synaptic_current": True,
        "propagation_matches_independent_sparse_reference": True,
        "tested_edge_current_delta": delta,
        "rom_unchanged": True,
        "learned_gameplay_demonstrated": False,
    }
    write_json(output / "report.json", result)
    print(f"Internal smoke test passed: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
