import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from pokefly import history
from pokefly.checkpoint import sha256
from pokefly.experiment import ExperimentConfig, TrainOptions
from pokefly.runner import write_json


def generation(root, name, *, config=None, mode="learn", rom="rom"):
    path = root / name / "checkpoints" / "step-00000500-unit"
    path.mkdir(parents=True)
    np.savez(path / "brain.npz", weights=np.array([1., 2.], np.float32))
    (path / "game.state").write_bytes(b"unit-only emulator fixture")
    write_json(path / "state.json", {
        "format": 1, "rom_sha1": rom,
        "experiment": {"config": asdict(config or ExperimentConfig()), "mode": mode},
    })
    write_json(path / "complete.json", {
        name: sha256(path / name) for name in ("brain.npz", "game.state", "state.json")
    })
    return path


@pytest.fixture
def normal(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "validate_rom", lambda path: "rom")
    config = ExperimentConfig()
    options = TrainOptions(rom=Path("unit-only.gb"), intro=True)
    pointer = history.history_path(tmp_path, config, "rom")
    first = generation(tmp_path, "first", config=config)

    def first_run(effective, *, on_checkpoint):
        assert effective.resume is None and effective.weights is None
        assert effective.intro and on_checkpoint is not None
        on_checkpoint(first)
        return first.parent.parent

    history.launch_with_history(options, root=tmp_path, runner=first_run)
    return tmp_path, options, config, pointer, first


def test_normal_restart_resumes_own_game_but_intro_retains_only_synapses(normal):
    root, options, config, pointer, first = normal
    generation(root, "newer-research-run")  # Never adopted by a latest-directory scan.
    seen = []

    def inspect(effective, *, on_checkpoint):
        seen.append(effective)
        assert on_checkpoint is not None
        return root

    history.launch_with_history(replace(options, intro=False), root=root, runner=inspect)
    assert seen[-1].resume == first and seen[-1].weights is None
    history.launch_with_history(options, root=root, runner=inspect)
    assert seen[-1].weights == first and seen[-1].resume is None
    assert seen[-1].intro and seen[-1].load_state is None
    assert history.load_history(pointer, config, "rom") == first


def test_every_complete_checkpoint_advances_history_and_prior_generations_survive(normal):
    root, options, config, pointer, first = normal
    second, third = generation(root, "second"), generation(root, "third")
    original = sha256(first / "brain.npz")

    def checkpoints(effective, *, on_checkpoint):
        for path in (second, third):
            on_checkpoint(path)
            assert history.load_history(pointer, config, "rom") == path
        raise RuntimeError("simulated application failure after completed checkpoint")

    with pytest.raises(RuntimeError, match="simulated"):
        history.launch_with_history(options, root=root, runner=checkpoints)
    assert history.load_history(pointer, config, "rom") == third
    assert sha256(first / "brain.npz") == original
    with history.history_lock(pointer):
        pass  # Error released the session lock; no stale PID cleanup required.


@pytest.mark.parametrize("mode", ["frozen", "no-reward"])
def test_control_runs_cannot_advance_normal_learning_history(normal, mode):
    root, options, _, pointer, first = normal
    before = pointer.read_bytes()

    def control(effective, *, on_checkpoint):
        assert effective.mode == mode and effective.weights == first
        assert effective.resume is None and on_checkpoint is None

    history.launch_with_history(replace(options, mode=mode), root=root, runner=control)
    assert pointer.read_bytes() == before


def test_fresh_brain_is_explicit_and_keeps_old_checkpoints(normal):
    root, options, _, pointer, first = normal
    before = pointer.read_bytes()

    def fresh(effective, *, on_checkpoint):
        assert effective.weights is None and effective.resume is None
        assert on_checkpoint is not None

    history.launch_with_history(options, root=root, runner=fresh, fresh_brain=True)
    assert pointer.read_bytes() == before  # No completed new checkpoint yet.
    assert (first / "complete.json").is_file()
    with pytest.raises(ValueError, match="Fresh brain"):
        history.launch_with_history(replace(options, weights=first), root=root, fresh_brain=True)


@pytest.mark.parametrize("change", ["json", "manifest", "weights", "identity", "list", "missing"])
def test_corrupt_history_fails_closed_instead_of_forgetting_learning(normal, change):
    root, options, _, pointer, first = normal
    if change == "json":
        pointer.write_text("invalid", encoding="utf-8")
    elif change == "manifest":
        (first / "complete.json").write_text("{}", encoding="utf-8")
    elif change == "weights":
        np.savez(first / "brain.npz", weights=np.array([3., 4.]))
    elif change == "list":
        write_json(pointer, [])
    else:
        value = json.loads(pointer.read_text())
        if change == "missing":
            del value["checkpoint"]
        else:
            value["rom_sha1"] = "wrong"
        write_json(pointer, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not run a fresh brain after damaged history")

    with pytest.raises(ValueError):
        history.launch_with_history(options, root=root, runner=forbidden)


def test_models_and_roms_have_separate_history_and_explicit_checkpoint_adoption(normal):
    root, options, config, pointer, _ = normal
    altered = replace(config, frames=48)
    assert history.history_path(root, altered, "rom") != pointer
    assert history.history_path(root, config, "other-rom") != pointer
    previous = generation(root, "explicit-other-model", config=altered)

    def adopt(effective, *, on_checkpoint):
        assert effective.weights == previous
        on_checkpoint(previous)

    history.launch_with_history(replace(options, weights=previous), root=root, runner=adopt)
    other = history.history_path(root, altered, "rom")
    assert history.load_history(other, altered, "rom") == previous
    assert history.load_history(pointer, config, "rom") != previous


@pytest.mark.parametrize("mode,rom", [("frozen", "rom"), ("no-reward", "rom"), ("learn", "other")])
def test_history_cannot_be_advanced_by_wrong_rom_or_control_checkpoint(normal, mode, rom):
    root, _, config, pointer, _ = normal
    wrong = generation(root, "wrong", mode=mode, rom=rom)
    before = pointer.read_bytes()
    with pytest.raises(ValueError, match="model, ROM or learning mode"):
        history.save_history(pointer, wrong, config, "rom")
    assert pointer.read_bytes() == before


def test_normal_history_refuses_stage_specific_start(normal):
    root, options, _, _, _ = normal
    with pytest.raises(ValueError, match="whole new game or exact resume"):
        history.launch_with_history(replace(options, intro=False, load_state=Path("battle.state")),
                                    root=root)


def test_only_one_normal_session_can_own_the_same_brain(tmp_path):
    path = tmp_path / "history.json"
    with history.history_lock(path):
        with pytest.raises(ValueError, match="already has an active session"):
            with history.history_lock(path):
                raise AssertionError("Second normal session must not acquire the same history")
    with history.history_lock(path):
        pass


def test_cli_keeps_research_explicit_and_routes_normal_history_flags(monkeypatch):
    from pokefly import cli, experiment

    calls = []
    monkeypatch.setattr(cli, "resolve_rom", lambda *args: Path("unit-only.gb"))
    monkeypatch.setattr(experiment, "train", lambda options: calls.append(("research", options)))

    def remembered(options, *, fresh_brain):
        calls.append(("normal", options, fresh_brain))

    monkeypatch.setattr(history, "launch_with_history", remembered)
    assert cli.main(["train"]) == 0
    assert calls[-1][0] == "research"
    assert cli.main(["train", "--remember", "--intro"]) == 0
    assert calls[-1][0] == "normal" and calls[-1][1].intro and not calls[-1][2]
    assert cli.main(["train", "--remember", "--fresh-brain"]) == 0
    assert calls[-1][2]
    count = len(calls)
    assert cli.main(["train", "--fresh-brain"]) == 2
    assert len(calls) == count
