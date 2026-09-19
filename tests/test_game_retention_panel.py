import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("evaluate_game_retention_panel")


def sources(tmp_path):
    return [(tmp_path / str(seed), {}, {"source_launch_seed": seed, "rom_sha1": "test"})
            for seed in (401, 402)]


def test_shared_original_requires_same_model_rom_and_distinct_sources(tmp_path, monkeypatch):
    m = script(monkeypatch)
    data = sources(tmp_path)
    m.validate_sources(data, [3601])  # Explicit partial panels are allowed.
    for seeds in ([], [3601, 3601], [401, 3601]):
        with pytest.raises(ValueError):
            m.validate_sources(data, seeds)
    with pytest.raises(ValueError):
        m.validate_sources([data[0], data[0]], [3601, 3602])
    changed = copy.deepcopy(data)
    changed[1][1]["frames"] = 48
    with pytest.raises(ValueError, match="exact saved model"):
        m.validate_sources(changed, [3601])
    changed = copy.deepcopy(data)
    changed[1][2]["rom_sha1"] = "different"
    with pytest.raises(ValueError, match="different ROMs"):
        m.validate_sources(changed, [3601])


@pytest.mark.parametrize("seeds", [[3601], [3601, 3602]])
def test_shared_control_panel_freezes_all_sources_and_avoids_duplicate_originals(
    tmp_path, monkeypatch, seeds,
):
    m = script(monkeypatch)
    data, snapshots = sources(tmp_path), {}
    for index, (checkpoint, _, provenance) in enumerate(data):
        checkpoint.mkdir()
        (checkpoint / "brain.npz").write_bytes(b"fake protocol source")
        provenance["brain_sha256"] = m.sha256(checkpoint / "brain.npz")
        snapshots[str(checkpoint)] = ({"weights": np.array([2.0 + index])}, {})
    output = tmp_path / "output"
    output.mkdir()
    calls = []

    def train(options):
        calls.append(options)
        path = output / f"run-{len(calls)}"
        path.mkdir()
        (path / "start.state").write_bytes(b"identical scripted fresh-game setup")
        (path / "summary.json").write_text(json.dumps({"reason": "step_limit"}))
        (path / "brain.npz").write_bytes(b"fake final checkpoint")
        weight = (snapshots[str(options.weights)][0]["weights"].copy()
                  if options.weights else np.array([1.0]))
        snapshots[str(path / "latest-checkpoint.json")] = (
            {"weights": weight, "base": np.array([1.0])},
            {"directory": str(path), "experiment": {"mode": "frozen"}},
        )
        return path

    def checked_source(path, *, heldout_seeds):
        assert heldout_seeds == seeds
        return data[int(path.name) - 401]

    monkeypatch.setattr(m, "completed_game_source", checked_source)
    monkeypatch.setattr(m, "read_checkpoint", lambda path: snapshots[str(path)])
    monkeypatch.setattr(m, "train", train)
    monkeypatch.setattr(m, "measure", lambda path: {"samples": 16,
                                                  "weight_updates_this_evaluation": 0})
    monkeypatch.setattr(m, "run_directory", lambda name: output)
    monkeypatch.setattr(m, "resolve_rom", lambda *args: Path("ignored.gb"))
    monkeypatch.setattr(sys, "argv", ["evaluate_game_retention_panel.py", "--source-runs",
                                     "401", "402", "--steps", "16", "--seeds", *map(str, seeds)])
    m.main()
    assert len(calls) == 3 * len(seeds)
    assert sum(o.weights is None for o in calls) == len(seeds)
    assert all(o.mode == "frozen" and o.intro and not o.load_state and not o.resume for o in calls)
    assert [o.weights for o in calls[:3]] == [None, data[0][0], data[1][0]]
    if len(seeds) == 2:
        assert [o.weights for o in calls[3:]] == [data[1][0], data[0][0], None]
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["single_seed_partial_panel"] == (len(seeds) == 1)
    assert report["one_shared_original_per_seed"]
