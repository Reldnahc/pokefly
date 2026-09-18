"""Exactly resume an interrupted paired/shuffled acquisition, not a new trial.

Only the original two-arm acquisition protocol is supported. Completed stages
are copied with their hashes and never counted as new evidence. Full neural
state and action/reward history resume at each arm's last complete checkpoint.
No weights from this ROM-free assay are permitted in Pokemon.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from evaluate_learning_choices import score, test_choices, warmup

from pokefly.actions import pressed_buttons
from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json

ARMS = ("paired", "unpaired_within_cue")


def validate_source(source):
    if source.get("status") != "running" or source.get("previous_report"):
        raise ValueError("Only an interrupted initial acquisition can resume here")
    if source.get("reverse_mapping") or source.get("same_acquired_start_for_both_reversal_arms"):
        raise ValueError("This recovery must not change or reinterpret a reversal")
    points = source["checkpoints"]
    if points != sorted(set(points)) or not points or points[0] < 32:
        raise ValueError("Invalid checkpoint schedule")
    seen = set()
    for row in source["rows"]:
        key = (row["arm"], row["training_decisions"])
        if key in seen or key[0] not in ARMS or key[1] not in points:
            raise ValueError("Duplicate or unregistered stage")
        seen.add(key)
    for arm in ARMS:
        completed = sorted(step for name, step in seen if name == arm)
        if completed != points[:len(completed)]:
            raise ValueError("Missing intermediate stage")
    if any(name == ARMS[1] for name, _ in seen) and (ARMS[0], points[-1]) not in seen:
        raise ValueError("Shuffled continuation needs the complete paired schedule")


def shuffled_schedule(history, checkpoints, cues, seed):
    values = np.array([row["reward"] for row in history], dtype=float)
    labels = np.array([row["cue"] for row in history])
    if len(history) != checkpoints[-1]:
        raise ValueError("Complete paired history required")
    rng = np.random.default_rng(seed + 712345)
    previous = 0
    for end in checkpoints:
        for cue in cues:
            indices = np.flatnonzero(labels[previous:end] == cue) + previous
            values[indices] = rng.permutation(values[indices])
        previous = end
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.source / "report.json").read_text())
    validate_source(report)
    config = Path(report["config"])
    if report.get("config_sha256") and sha256(config) != report["config_sha256"]:
        raise ValueError("Source configuration changed")
    c = InternalBrain(device="cuda", config=load_config(config).brain)
    initial, initial_state = c.snapshot()
    points, seed = report["checkpoints"], report["seed"]
    cues = tuple(report.get("cue_images", ["left", "right"]))
    buttons = tuple(report.get("rewarded_button_pair", ["left", "right"]))
    patterns = test_patterns()
    output = run_directory("resumed-visual-acquisition")
    shutil.copyfile(config, output / "fixed-config.json")
    report["config"] = str(output / "fixed-config.json")
    report["config_sha256"] = sha256(output / "fixed-config.json")
    report["recovery"] = {
        "source": str(args.source), "source_report_sha256": sha256(args.source / "report.json"),
        "copied_stages_are_not_new_trials": True, "stage_hashes": {},
    }
    histories, starts = {}, {}
    for arm in ARMS:
        stages = [r["training_decisions"] for r in report["rows"] if r["arm"] == arm]
        start = max(stages, default=0)
        history = json.loads((args.source / f"{arm}-training.json").read_text()) if start else []
        if len(history) != start:
            raise ValueError("Saved training history does not match the latest stage")
        if any(row["cue"] != cues[(i // 16) % 2] for i, row in enumerate(history)):
            raise ValueError("Saved cue order changed")
        for step in stages:
            for suffix in ("npz", "json"):
                name = f"{arm}-{step}.{suffix}"
                shutil.copyfile(args.source / name, output / name)
                report["recovery"]["stage_hashes"][name] = sha256(output / name)
        histories[arm], starts[arm] = history, start
        write_json(output / f"{arm}-training.json", history)
    write_json(output / "report.json", report)
    print("Recovery:", output, flush=True)
    for arm in ARMS:
        history, start = histories[arm], starts[arm]
        if start:
            label = output / f"{arm}-{start}"
            with np.load(label.with_suffix(".npz"), allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            c.restore(arrays, json.loads(label.with_suffix(".json").read_text()))
        else:
            c.restore(initial, initial_state, weights_only=True)
            warmup(c, seed + 1000)
        schedule = None if arm == "paired" else shuffled_schedule(
            histories["paired"], points, cues, seed
        )
        if schedule is not None and not np.array_equal(
            schedule[:start], np.array([row["reward"] for row in history])
        ):
            raise ValueError("Shuffled reward prefix differs; cannot resume exactly")
        for index in range(start, points[-1]):
            cue = cues[(index // 16) % 2]
            target = buttons[cues.index(cue)]
            action = c.choose(c.observe(patterns[cue]))[0]
            pressed = pressed_buttons(action)
            reward = float(schedule[index]) if schedule is not None else (
                1.0 if target in pressed else report.get("incorrect_reward", 0.0)
                if set(buttons) & set(pressed) else 0.0
            )
            c.reinforce(reward)
            history.append({"action": action, "cue": cue, "reward": float(reward)})
            count = index + 1
            if count in points:
                arrays, state = c.snapshot()
                retention = test_choices(c, seed=seed, cues=cues)
                report["rows"].append({
                    "arm": arm, "training_decisions": count, "retention": retention,
                    "score": score(retention, False, buttons=buttons, cues=cues),
                    "learning": state["plasticity"],
                })
                np.savez_compressed(output / f"{arm}-{count}.npz", **arrays)
                write_json(output / f"{arm}-{count}.json", state)
                write_json(output / f"{arm}-training.json", history)
                write_json(output / "report.json", report)
                print(arm, count, report["rows"][-1]["score"], flush=True)
                c.restore(arrays, state)
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
