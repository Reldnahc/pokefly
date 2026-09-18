"""One-factor actual-game internal-plasticity test; unchanged broad rewards.

Reuse an explicitly identified completed original-weight frozen/learning pair.
First verify a frozen 64-decision prefix exactly, then train from original
synapses with the same seed and intro. No synthetic/oracle weights or routes.
Any benefit still requires separate held-out frozen-weight transfer.
"""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from evaluate_saved_gameplay import measure

from pokefly.checkpoint import read_checkpoint, sha256
from pokefly.experiment import TrainOptions, load_config, train
from pokefly.rom import resolve_rom
from pokefly.runner import run_directory, write_json


def validate_one_factor(original, candidate, factor="learning_rate"):
    if factor not in ("learning_rate", "rule", "credit_timing"):
        raise ValueError("Supported factors are internal learning_rate, rule or credit_timing")
    old, new = copy.deepcopy(original), copy.deepcopy(candidate)
    if factor == "credit_timing":
        old_value, new_value = old.pop(factor), new.pop(factor)
    else:
        old_value = old["brain"]["plasticity"].pop(factor)
        new_value = new["brain"]["plasticity"].pop(factor)
    if old != new or new_value == old_value:
        raise ValueError(f"Candidate must differ ONLY in internal {factor}")
    return old_value, new_value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-report", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/visual-fast-learning-v2.json"))
    p.add_argument("--factor", choices=("learning_rate", "rule", "credit_timing"),
                   default="learning_rate")
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    if not 0 <= args.port <= 65535:
        p.error("Invalid dashboard port")
    source = json.loads(args.baseline_report.read_text())
    if source.get("status") != "completed" or source.get("continuation_source"):
        p.error("Completed fresh-intro pair required, not a continuation")
    if {r["mode"] for r in source["rows"]} != {"learn", "frozen"}:
        p.error("Matched original frozen and learning controls required")
    candidate = asdict(load_config(args.config))
    if sha256(Path(source["config"])) != source["config_sha256"]:
        raise ValueError("Source configuration changed")
    old_value, new_value = validate_one_factor(
        asdict(load_config(Path(source["config"]))), candidate, args.factor
    )
    control = next(r for r in source["rows"] if r["mode"] == "frozen")
    baseline = Path(control["run"])
    stored = json.loads((baseline / "config.json").read_text())
    options = stored["options"]
    assert options["intro"] and options["mode"] == "frozen"
    assert not any(options[k] for k in ("resume", "weights", "load_state"))
    assert options["seed"] == source["seed"]
    assert options["steps"] == source["steps"] == control["samples"]
    assert json.loads((baseline / "summary.json").read_text())["reason"] == "step_limit"
    arrays, _ = read_checkpoint(baseline / "latest-checkpoint.json")
    np.testing.assert_array_equal(arrays["weights"], arrays["base"])
    output = run_directory({"learning_rate": "visual-learning-rate-gameplay",
                            "rule": "visual-plasticity-rule-gameplay",
                            "credit_timing": "visual-credit-timing-gameplay"}[args.factor])
    config = output / "fixed-config.json"
    write_json(config, candidate)
    report = {
        "scope": __doc__,
        "seed": source["seed"],
        "steps": source["steps"],
        "config": str(config),
        "config_sha256": sha256(config),
        "factor": args.factor,
        f"old_{args.factor}": old_value,
        f"new_{args.factor}": new_value,
        "baseline_report": str(args.baseline_report),
        "baseline_report_sha256": sha256(args.baseline_report),
        "reused_controls_not_new_trials": source["rows"],
        "rows": [],
        "status": "running",
    }
    write_json(output / "report.json", report)
    common = dict(
        rom=resolve_rom(None, Path.cwd()),
        device="cuda",
        seed=source["seed"],
        config=config,
        intro=True,
        hz=0,
        port=args.port,
    )
    prefix = train(
        TrainOptions(**common, steps=64, mode="frozen", dashboard=False, checkpoint_every=0)
    )
    keys = (
        "action",
        "buttons",
        "spikes_total",
        "groups",
        "motor_rates_hz",
        "telemetry",
        "reward",
        "reward_events",
        "input_window",
    )

    def rows(path):
        return [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]

    expected, actual = rows(baseline)[:64], rows(prefix)
    assert len(expected) == len(actual) == 64
    assert [{k: r[k] for k in keys} for r in expected] == [{k: r[k] for k in keys} for r in actual]
    assert sha256(prefix / "start.state") == sha256(baseline / "start.state")
    report["prefix_verification"] = {"run": str(prefix), "samples": 64, "all_fields_exact": True}
    write_json(output / "report.json", report)
    path = train(
        TrainOptions(
            **common, steps=source["steps"], mode="learn", dashboard=True, checkpoint_every=500
        )
    )
    assert sha256(path / "start.state") == sha256(baseline / "start.state")
    route = [r["telemetry"]["y"] for r in rows(path) if r["telemetry"]["map"] == 12]
    report["rows"].append({**measure(path), "route1_minimum_y": min(route) if route else None})
    if json.loads((path / "summary.json").read_text())["reason"] != "step_limit":
        write_json(output / "report.json", report)
        raise RuntimeError("Stopped; partial candidate evidence preserved")
    report["status"] = "completed"
    write_json(output / "report.json", report)
    print(json.dumps(report["rows"][-1]), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
