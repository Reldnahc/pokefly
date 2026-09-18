from dataclasses import asdict
from pathlib import Path

import pytest

from pokefly.cli import _parser
from pokefly.experiment import ExperimentConfig, load_config


def test_configuration_is_explicit_roundtrippable_and_strict():
    c = ExperimentConfig()
    assert ExperimentConfig.from_dict(asdict(c)) == c
    with pytest.raises(TypeError):
        ExperimentConfig.from_dict({"house_exit_reward": 100})
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"brain": {"noise_hz": float("inf")}})
    with pytest.raises(ValueError):
        ExperimentConfig(frames=1)


def test_training_is_separate_from_legacy_and_intro_is_opt_in():
    parser = _parser()
    train = parser.parse_args(["train"])
    assert train.mode == "learn" and train.dashboard and not train.intro
    assert parser.parse_args(["play"]).command == "play"
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])


def test_legacy_checkpoint_does_not_silently_switch_arbitration():
    raw = {"brain": {"motor": {"threshold_hz": 0.5}}}
    assert ExperimentConfig.from_dict(raw).brain.motor.arbitration == "parallel-v2"
    assert (
        ExperimentConfig.from_dict(raw, checkpoint=True).brain.motor.arbitration == "exclusive-v1"
    )
    current = asdict(ExperimentConfig())
    assert ExperimentConfig.from_dict(current, checkpoint=True) == ExperimentConfig()
    assert "arbitration" not in raw["brain"]["motor"]


def test_button_timing_is_versioned_not_silently_migrated():
    raw = {"brain": {"motor": {"arbitration": "sustained-v3"}}}
    assert ExperimentConfig.from_dict(raw, checkpoint=True).button_timing == "simultaneous-v1"
    candidate = ExperimentConfig.from_dict({**raw, "button_timing": "serial-v2"})
    assert ExperimentConfig.from_dict(asdict(candidate), checkpoint=True) == candidate
    with pytest.raises(ValueError):
        ExperimentConfig(button_timing="unknown")
    with pytest.raises(ValueError):
        ExperimentConfig(button_timing="serial-v2", frames=3)


def test_preserved_serial_profile_changes_only_button_delivery():
    old = asdict(load_config(Path("configs/sensorimotor-bounded-v1.json")))
    new = asdict(load_config(Path("configs/sensorimotor-bounded-serial-v2.json")))
    assert old["button_timing"] == "simultaneous-v1"
    assert new["button_timing"] == "serial-v2"
    new["button_timing"] = old["button_timing"]
    assert new == old  # Same brain, retinal timing, reward settings and frame budget.


def test_showcase_default_uses_tested_visual_model_without_changing_rewards_or_decoder():
    previous = load_config(Path("configs/sensorimotor-bounded-serial-v2.json"))
    showcase = load_config(Path("configs/visual-release-wide-v3.json"))
    launcher = Path("scripts/start.ps1").read_text()
    assert "$Profile = 'visual-release-wide-v3'" in launcher
    assert showcase.brain.dynamics.visual_model == "calibrated-rate-v1"
    assert showcase.brain.dynamics.isolate_nonvisual_sensory
    assert showcase.brain.dynamics.graded_release == 1.0
    assert showcase.brain.intrinsic_calibration == (
        "fly-data/intrinsic-neutral-visual-release-wide-v3.npz"
    )
    assert showcase.brain.plasticity.rule == "sensorimotor-perturb-v3"
    assert showcase.visual_timing == "stream-v1"
    assert showcase.button_timing == previous.button_timing == "serial-v2"
    assert showcase.rewards == previous.rewards
    assert showcase.brain.motor == previous.brain.motor
    assert showcase.frames == previous.frames
    # A new launcher default must not change the baseline or migrate old brains.
    assert _parser().parse_args(["train"]).config is None
    assert "if (-not $Resume -and -not $Weights -and $Profile -ne 'baseline')" in launcher
    assert "Resume/Weights restore the saved model profile" in launcher


def test_setup_prepares_showcase_dependencies_without_overwriting_existing_calibration():
    setup = Path("scripts/setup.ps1").read_text()
    assert "scripts/fetch_visual_reference.py" in setup
    assert (
        "if (-not (Test-Path -LiteralPath "
        "'fly-data\\intrinsic-neutral-visual-release-wide-v3.npz'))"
    ) in setup
    assert "--visual-model calibrated-rate-v1 --graded-release 1 --wide-bias" in setup
    assert "--export fly-data/intrinsic-neutral-visual-release-wide-v3.npz" in setup


def test_missing_reward_timing_retains_legacy_checkpoint_behavior():
    assert ExperimentConfig.from_dict({}, checkpoint=True).rewards.timing == "encounter-end-v1"
    config = ExperimentConfig.from_dict({"rewards": {"timing": "last-faint-v3"}})
    assert ExperimentConfig.from_dict(asdict(config), checkpoint=True) == config


def test_missing_learning_scope_retains_legacy_checkpoint_behavior():
    old = {
        "brain": {
            "plasticity": {"rule": "sensorimotor-perturb-v3"},
            "dynamics": {"profile": "hybrid-v1"},
        }
    }
    assert ExperimentConfig.from_dict(old, checkpoint=True).brain.plasticity.scope == (
        "motor-inputs-v1"
    )
    candidate = ExperimentConfig.from_dict(
        {
            "brain": {
                "plasticity": {"rule": "sensorimotor-perturb-v3", "scope": "premotor-one-hop-v2"},
                "dynamics": {"profile": "hybrid-v1"},
            }
        }
    )
    assert ExperimentConfig.from_dict(asdict(candidate), checkpoint=True) == candidate
