"""Longer ROM-free visual-learning curve; no diagnostic weights enter Pokemon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from assay_feedback import earned_feedback, feedback_row, shuffled_feedback, validate_history
from evaluate_learning_choices import score, test_choices, warmup

from pokefly.actions import pressed_buttons
from pokefly.checkpoint import sha256
from pokefly.experiment import load_config
from pokefly.internal_brain import InternalBrain
from pokefly.pixel_brain import test_patterns
from pokefly.runner import run_directory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/sensorimotor-perturb-v1.json"))
    p.add_argument("--seed", type=int, default=501)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[512, 2048, 8192])
    p.add_argument("--reverse", action="store_true")
    p.add_argument(
        "--allow-reversal",
        action="store_true",
        help="Explicitly allow changing the cue contingency when continuing saved training",
    )
    p.add_argument(
        "--same-start-reversal", action="store_true",
        help="Start BOTH reversal arms from the same acquired paired neural checkpoint",
    )
    p.add_argument(
        "--buttons",
        nargs=2,
        default=["left", "right"],
        choices=("up", "down", "left", "right", "a", "b", "start"),
    )
    p.add_argument("--cues", nargs=2, default=["left", "right"], choices=tuple(test_patterns()))
    p.add_argument(
        "--correct-reward",
        type=float,
        default=1.0,
        help="ROM-free assay only: positive feedback strength; does not change game rewards",
    )
    p.add_argument(
        "--incorrect-reward",
        type=float,
        default=0.0,
        help="ROM-free assay only: optional feedback for the wrong competing direction",
    )
    p.add_argument(
        "--reward-delay-decisions", type=int, default=0,
        help="ROM-free only: delay feedback by this many decisions; no eligibility latching",
    )
    p.add_argument(
        "--continue-from",
        type=Path,
        help="Completed paired/shuffled curve; restore full training dynamics",
    )
    args = p.parse_args()
    if args.allow_reversal and not args.continue_from:
        p.error("Reversal continuation requires a completed source curve")
    if args.same_start_reversal and not (args.continue_from and args.allow_reversal):
        p.error("Same-start reversal requires explicit reversal continuation")
    buttons, cues = tuple(args.buttons), tuple(args.cues)
    if (
        len(set(buttons)) != 2
        or len(set(cues)) != 2
        or not any(
            set(buttons) <= channel
            for channel in ({"up", "down", "left", "right"}, {"a", "b", "start"})
        )
    ):
        p.error("Two distinct cues and two competing buttons from one channel required")
    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[0] < 32:
        p.error("Positive training checkpoints >=32 required")
    if not 0 <= args.reward_delay_decisions < checkpoints[-1]:
        p.error("Feedback delay must be nonnegative and shorter than the training budget")
    delay = args.reward_delay_decisions
    if not np.isfinite(args.correct_reward) or args.correct_reward <= 0:
        p.error("Correct-choice feedback must be positive and finite")
    if not -1 <= args.incorrect_reward <= 0:
        p.error("Incorrect-choice feedback must be between -1 and 0")
    output = run_directory("visual-learning-curve")
    c = InternalBrain(device="cuda", config=load_config(args.config).brain)
    initial, state = c.snapshot()
    patterns = test_patterns()
    previous_report, starting_step = None, 0
    if args.continue_from:
        previous_report = json.loads((args.continue_from / "report.json").read_text())
        starting_step = max(previous_report["checkpoints"])
        complete_arms = {
            row["arm"] for row in previous_report["rows"]
            if row["training_decisions"] == starting_step
        }
        if complete_arms != {"paired", "unpaired_within_cue"}:
            p.error("Finish both source arms before continuation")
        if args.same_start_reversal and previous_report["reverse_mapping"] == args.reverse:
            p.error("Same-start reversal must actually reverse the acquired contingency")
        if (
            previous_report["seed"] != args.seed
            or (previous_report["reverse_mapping"] != args.reverse and not args.allow_reversal)
            or checkpoints[0] <= starting_step
            or previous_report.get("correct_reward", 1.0) != args.correct_reward
            or previous_report.get("incorrect_reward", 0.0) != args.incorrect_reward
            or previous_report.get("reward_delay_decisions", 0) != delay
            or tuple(previous_report.get("cue_images", ["left", "right"])) != cues
            or tuple(previous_report.get("rewarded_button_pair", ["left", "right"])) != buttons
        ):
            p.error("Continuation must preserve seed/mapping and advance the completed curve")
        pretest = previous_report["pretest"]
    else:
        pretest = test_choices(c, seed=args.seed, cues=cues)
    report = {
        "config": str(args.config),
        "seed": args.seed,
        "reverse_mapping": args.reverse,
        "correct_reward": args.correct_reward,
        "incorrect_reward": args.incorrect_reward,
        "reward_delay_decisions": delay,
        "feedback_timing": "Feedback matures during continued neural activity; pending tail "
        "is saved, never flushed into a probe. Shuffling matches originating cue and "
        "delivery checkpoint interval, including a separate pending tail.",
        "cue_images": cues,
        "rewarded_button_pair": buttons,
        "protocol": "Continuous training; probes branch off and restore full training state. "
        "Synthetic button rewards only in this ROM-free assay. Check conditional accuracy, "
        "not merely total competing-button activity. Shuffling preserves rewards within cue.",
        "checkpoints": checkpoints,
        "pretest": pretest,
        "pre_score": score(pretest, args.reverse, buttons=buttons, cues=cues),
        "rows": [],
        "status": "running",
        "same_acquired_start_for_both_reversal_arms": args.same_start_reversal,
    }
    if previous_report:
        report["previous_report"] = str(args.continue_from / "report.json")
        report["previous_report_sha256"] = sha256(args.continue_from / "report.json")
        report["starting_step"] = starting_step
        if previous_report["reverse_mapping"] != args.reverse:
            report["phase"] = "explicit contingency reversal; not additional acquisition"
            report["pre_reversal_scores"] = {
                row["arm"]: score(row["retention"], args.reverse, buttons=buttons, cues=cues)
                for row in previous_report["rows"]
                if row["training_decisions"] == starting_step
            }
            if args.same_start_reversal:
                paired_score = report["pre_reversal_scores"]["paired"]
                report["pre_reversal_scores"] = {
                    arm: paired_score for arm in ("paired", "unpaired_within_cue")
                }
                label = args.continue_from / f"paired-{starting_step}"
                report["shared_reversal_initial_state_sha256"] = {
                    suffix: sha256(label.with_suffix(suffix)) for suffix in (".npz", ".json")
                }
    write_json(output / "report.json", report)
    paired_training = []
    if args.continue_from:
        old_training = json.loads((args.continue_from / "paired-training.json").read_text())
        validate_history(old_training, delay)
        paired_training = old_training.copy()
        if len(old_training) != starting_step:
            raise ValueError("Incomplete source training history")
    for arm in ("paired", "unpaired_within_cue"):
        training = []
        if args.continue_from:
            source_arm = "paired" if args.same_start_reversal else arm
            label = args.continue_from / f"{source_arm}-{starting_step}"
            with np.load(label.with_suffix(".npz"), allow_pickle=False) as archive:
                previous_arrays = {k: archive[k].copy() for k in archive.files}
            previous_state = json.loads(label.with_suffix(".json").read_text())
            c.restore(previous_arrays, previous_state)
            training = json.loads((args.continue_from / f"{source_arm}-training.json").read_text())
            if len(training) != starting_step:
                raise ValueError("Incomplete source training history")
            validate_history(training, delay)
        else:
            c.restore(initial, state, weights_only=True)
            warmup(c, args.seed + 1000)
        schedule = None
        if arm != "paired":
            schedule = shuffled_feedback(
                paired_training, checkpoints, cues, args.seed, delay, starting_step=starting_step,
            )
        for index in range(starting_step, checkpoints[-1]):
            cue_index = (index // 16) % 2
            cue = cues[cue_index]
            target = buttons[cue_index ^ int(args.reverse)]
            action = c.choose(c.observe(patterns[cue]))[0]
            pressed = pressed_buttons(action)
            earned = (
                (
                    args.correct_reward
                    if target in pressed
                    else args.incorrect_reward
                    if set(buttons) & set(pressed)
                    else 0.0
                )
                if schedule is None
                else schedule[index]
            )
            record = feedback_row(action, cue, earned, training, delay)
            c.reinforce(record['reward'])
            training.append(record)
            if arm == "paired":
                paired_training.append(record)
            count = index + 1
            if count in checkpoints:
                learned, learned_state = c.snapshot()
                retention = test_choices(c, seed=args.seed, cues=cues)
                row = {
                    "arm": arm,
                    "training_decisions": count,
                    "retention": retention,
                    "score": score(retention, args.reverse, buttons=buttons, cues=cues),
                    "learning": learned_state["plasticity"],
                    "pending_feedback_count": min(delay, count),
                    "pending_feedback_sum": sum(
                        earned_feedback(r, delay) for r in training[-delay:]
                    ) if delay else 0.0,
                }
                report["rows"].append(row)
                write_json(output / "report.json", report)
                np.savez_compressed(output / f"{arm}-{count}.npz", **learned)
                write_json(output / f"{arm}-{count}.json", learned_state)
                # Preserve action/reward history alongside every neural snapshot,
                # not only at the end of a potentially long arm. An interrupted
                # run is still labeled running; never call its final gate passed.
                write_json(output / f"{arm}-training.json", training)
                print(arm, count, row["score"], flush=True)
                c.restore(learned, learned_state)
        write_json(output / f"{arm}-training.json", training)
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
