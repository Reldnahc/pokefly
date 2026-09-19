"""Real whole-game persistence check, with private test history, not learning evidence."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.history import history_path, launch_with_history, load_history
from pokefly.rom import resolve_rom, validate_rom
from pokefly.runner import run_directory, write_json


def rows(path):
    return [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]


def compare(first, second, *, first_offset=0):
    ignored = {"run_id", "compute_ms", "pacing"}
    for a, b in zip(rows(first)[first_offset:], rows(second), strict=True):
        assert {k: v for k, v in a.items() if k not in ignored} == {
            k: v for k, v in b.items() if k not in ignored
        }
    a, sa = read_checkpoint(first / "latest-checkpoint.json")
    b, sb = read_checkpoint(second / "latest-checkpoint.json")
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    assert sa["neural"] == sb["neural"] and sa["rewards"] == sb["rewards"]


def normal_history_fingerprints():
    directory = Path.cwd() / "runs" / "user-history"
    return {str(path): sha256(path) for path in directory.glob("*.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/visual-release-wide-v3.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    before = normal_history_fingerprints()
    output = run_directory("normal-history-smoke")
    print("Private history smoke:", output, flush=True)
    write_json(output / "report.json", {"status": "running", "scope": __doc__})
    rom = resolve_rom(None, Path.cwd())
    options = TrainOptions(rom=rom, device=args.device, config=args.config, intro=True,
                           steps=70, hz=0, dashboard=False, checkpoint_every=50)
    config, rom_hash = load_config(args.config), validate_rom(rom)
    pointer = history_path(output, config, rom_hash)
    initial = launch_with_history(options, root=output)
    initial_checkpoint = load_history(pointer, config, rom_hash)
    first, _ = read_checkpoint(initial_checkpoint)
    assert np.any(first["weights"] != first["base"])
    resumed = launch_with_history(replace(options, intro=False, steps=20), root=output)
    resumed_checkpoint = load_history(pointer, config, rom_hash)
    saved, _ = read_checkpoint(resumed_checkpoint)
    assert rows(resumed)[0]["sample"] == 71 and rows(resumed)[-1]["sample"] == 90
    uninterrupted = train(replace(options, steps=90, checkpoint_every=0))
    compare(uninterrupted, resumed, first_offset=70)
    source_hash = sha256(resumed_checkpoint / "brain.npz")
    pointer_before_control = pointer.read_bytes()
    frozen = launch_with_history(replace(options, mode="frozen"), root=output)
    frozen_arrays, _ = read_checkpoint(frozen / "latest-checkpoint.json")
    np.testing.assert_array_equal(frozen_arrays["weights"], saved["weights"])
    assert sha256(frozen / "start.state") == sha256(initial / "start.state")
    assert pointer.read_bytes() == pointer_before_control
    repeated = launch_with_history(options, root=output)
    explicit = train(replace(options, config=None, weights=resumed_checkpoint))
    compare(explicit, repeated)
    assert sha256(repeated / "start.state") == sha256(initial / "start.state")
    current = load_history(pointer, config, rom_hash)
    _, repeated_saved = read_checkpoint(repeated / "latest-checkpoint.json")
    assert current == Path(repeated_saved["directory"])
    assert sha256(resumed_checkpoint / "brain.npz") == source_hash
    assert normal_history_fingerprints() == before
    report = {
        "status": "completed", "scope": __doc__, "config": str(args.config),
        "initial": str(initial), "automatic_resume": str(resumed),
        "uninterrupted_control": str(uninterrupted), "frozen_new_game": str(frozen),
        "automatic_retained_new_game": str(repeated), "explicit_retained_control": str(explicit),
        "automatic_resume_exact": True, "automatic_new_game_exactly_matches_explicit_weights": True,
        "whole_new_game_start_matches": True, "control_does_not_advance_history": True,
        "previous_source_unchanged": True, "user_history_untouched": True,
        "synthetic_or_stage_specific_initialization": False,
        "learned_gameplay_demonstrated": False,
    }
    write_json(output / "report.json", report)
    print("Normal-history live smoke passed:", output, flush=True)


if __name__ == "__main__":
    main()
