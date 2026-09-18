from dataclasses import asdict

import pytest

from pokefly.cli import _parser
from pokefly.experiment import ExperimentConfig


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
