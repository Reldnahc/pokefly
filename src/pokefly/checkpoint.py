"""Non-pickle, checksummed, complete-generation experiment checkpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path

import numpy as np

from pokefly.runner import write_json

FORMAT_VERSION = 1


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("numpy", "pyboy", "flybrain")}


def save_checkpoint(
    output: Path, controller, game, hooks, rewards, *, frame, experiment, temporal=None
) -> Path:
    directory = output / "checkpoints" / f"step-{experiment['sample']:08d}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    arrays, neural = controller.snapshot()
    np.savez_compressed(
        directory / "brain.npz",
        **arrays,
        **(temporal.arrays() if temporal else {}),
        next_frame=frame,
    )
    # Debug opcodes never leak into an emulator save; checkpoints restore plain ROM execution.
    hooks.detach()
    try:
        game.save(directory / "game.state")
    finally:
        hooks.attach()
    write_json(
        directory / "state.json",
        {
            "format": FORMAT_VERSION,
            "rom_sha1": game.rom_sha1,
            "versions": versions(),
            "neural": neural,
            "rewards": rewards.state(),
            "experiment": experiment,
        },
    )
    manifest = {
        name: sha256(directory / name) for name in ("brain.npz", "game.state", "state.json")
    }
    write_json(directory / "complete.json", manifest)  # Last: incomplete generations cannot load.
    # Only this small pointer is replaced; previous completed checkpoints are retained.
    temporary = output / f"latest-checkpoint-{uuid.uuid4().hex}.tmp"
    write_json(temporary, {"path": str(directory.resolve())})
    os.replace(temporary, output / "latest-checkpoint.json")
    return directory


def read_checkpoint(directory: Path) -> tuple[dict, dict]:
    if directory.is_file() and directory.name == "latest-checkpoint.json":
        directory = Path(json.loads(directory.read_text(encoding="utf-8"))["path"])
    directory = directory.resolve()
    manifest = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if set(manifest) != {"brain.npz", "game.state", "state.json"}:
        raise ValueError("Invalid checkpoint manifest")
    for name, expected in manifest.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Incomplete or modified checkpoint: {name}")
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if state["format"] != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    with np.load(directory / "brain.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    state["directory"] = str(directory)
    return arrays, state
