"""Explicit normal-run history, isolated from research trials and neural control."""

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.rom import validate_rom
from pokefly.runner import write_json


def history_path(root, config, rom_sha1):
    identity = json.dumps({"config": asdict(config), "rom_sha1": rom_sha1}, sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()
    return Path(root) / "runs" / "user-history" / f"{key}.json"


@contextmanager
def history_lock(path):
    """OS-owned lock releases on a crash; no stale PID file or process killing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("This normal-run brain already has an active session") from exc
        # Closing the handle releases either platform's lock, including on errors.
        yield


def checked_saved_config(saved):
    from pokefly.experiment import ExperimentConfig

    return ExperimentConfig.from_dict(saved["experiment"]["config"], checkpoint=True)


def check_history_checkpoint(checkpoint, config, rom_sha1):
    _, saved = read_checkpoint(checkpoint)
    if (checked_saved_config(saved) != config or saved["rom_sha1"] != rom_sha1
            or saved["experiment"]["mode"] != "learn"):
        raise ValueError("Normal-run history model, ROM or learning mode mismatch")
    return Path(saved["directory"])


def load_history(path, config, rom_sha1):
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("format") != 1
            or record.get("config") != asdict(config)
            or record.get("rom_sha1") != rom_sha1):
        raise ValueError("Normal-run history identity mismatch; use an explicit checkpoint")
    if (not isinstance(record.get("checkpoint"), str) or not record["checkpoint"]
            or not isinstance(record.get("manifest_sha256"), str)):
        raise ValueError("Normal-run history is missing its checkpoint identity")
    checkpoint = Path(record["checkpoint"])
    if sha256(checkpoint / "complete.json") != record["manifest_sha256"]:
        raise ValueError("Normal-run history checkpoint changed; refusing to forget prior learning")
    return check_history_checkpoint(checkpoint, config, rom_sha1)


def save_history(path, checkpoint, config, rom_sha1):
    checkpoint = check_history_checkpoint(checkpoint, config, rom_sha1)
    record = {"format": 1, "config": asdict(config), "rom_sha1": rom_sha1,
              "checkpoint": str(checkpoint.resolve()),
              "manifest_sha256": sha256(checkpoint / "complete.json")}
    temporary = path.with_name(f"{path.stem}-{uuid.uuid4().hex}.tmp")
    write_json(temporary, record)
    os.replace(temporary, path)


def launch_with_history(options, *, fresh_brain=False, root=None, runner=None):
    """Normal play retains its own model's brain; research train() stays explicit."""
    from pokefly.experiment import load_config, train

    if options.load_state:
        raise ValueError("Normal history requires a whole new game or exact resume, not load-state")
    if fresh_brain and (options.resume or options.weights):
        raise ValueError("Fresh brain cannot combine with Resume or Weights")
    rom_sha1 = validate_rom(options.rom)
    explicit = options.resume or options.weights
    if explicit:
        if options.config:
            raise ValueError("Saved brains restore their own profile; omit config")
        _, saved = read_checkpoint(explicit)
        if saved["rom_sha1"] != rom_sha1:
            raise ValueError("Checkpoint ROM mismatch")
        config = checked_saved_config(saved)
    else:
        config = load_config(options.config)
    path = history_path(root or Path.cwd(), config, rom_sha1)
    with history_lock(path):
        if not explicit and not fresh_brain and path.exists():
            previous = load_history(path, config, rom_sha1)
            if options.intro or options.mode != "learn":
                options = replace(options, config=None, weights=previous)
            else:
                options = replace(options, config=None, resume=previous)
        if options.resume:
            print(f"Learning history: continuing saved brain AND game: {options.resume}",
                  flush=True)
        elif options.weights:
            print(f"Learning history: retained synapses, new game: {options.weights}", flush=True)
        else:
            print("Learning history: original weights (fresh brain or first normal session).",
                  flush=True)
            print("Older unregistered runs are not auto-imported; adopt one with Resume/Weights.",
                  flush=True)
        print(f"Normal-run history: {path}", flush=True)

        def callback(checkpoint):
            save_history(path, checkpoint, config, rom_sha1)

        if options.mode != "learn":
            print("Control mode: saved learning history will not advance.", flush=True)
            callback = None
        return (runner or train)(options, on_checkpoint=callback)
