from types import SimpleNamespace

import numpy as np
import pytest

from pokefly.motors import MOTOR_TYPES, MotorConfig, MotorDecoder, populations


def decoder(arbitration="parallel-v2"):
    return MotorDecoder(
        {name: np.array([i]) for i, name in enumerate(MOTOR_TYPES)},
        MotorConfig(arbitration=arbitration),
    )


@pytest.mark.parametrize("index,action", enumerate(MOTOR_TYPES))
def test_all_seven_fixed_mappings(index, action):
    d = decoder()
    spikes = np.zeros(7)
    spikes[index] = 3
    chosen, rates = d.choose(spikes, 0.24)
    assert chosen == action and rates[action] > 0
    assert d.choose(np.zeros(7), 0.24)[0] == "wait"


def test_per_neuron_normalization_not_population_size():
    groups = {name: np.array([i]) for i, name in enumerate(MOTOR_TYPES)}
    groups["down"] = np.array([1, 7, 8, 9])
    d = MotorDecoder(groups)
    counts = np.zeros(10)
    counts[0] = 2
    counts[[1, 7, 8, 9]] = 1
    assert d.choose(counts, 0.24)[0] == "up"


def test_legacy_ties_rotate_and_start_has_fixed_cooldown():
    d = decoder("exclusive-v1")
    assert [d.choose(np.ones(7), 0.24)[0] for _ in range(7)] == list(MOTOR_TYPES)
    counts = np.zeros(7)
    counts[-1] = 12
    assert d.choose(counts, 0.24)[0] == "wait"
    clone = decoder("exclusive-v1")
    clone.restore(d.state())
    for _ in range(10):
        assert d.choose(counts, 0.24) == clone.choose(counts, 0.24)


@pytest.mark.parametrize("button,index", [("a", 4), ("b", 5), ("start", 6)])
def test_function_never_suppresses_active_up(button, index):
    d = decoder()
    counts = np.zeros(7)
    counts[0], counts[index] = 1, 100
    action, _ = d.choose(counts, 0.24)
    assert action == "up+" + button
    assert d.choose(np.zeros(7), 0.24)[0] == "wait"


def test_two_channels_tie_rotate_independently_and_resume_exactly():
    d = decoder()
    assert [d.choose(np.ones(7), 0.24)[0] for _ in range(4)] == [
        "up+a",
        "down+b",
        "left+start",
        "right+a",
    ]
    clone = decoder()
    clone.restore(d.state())
    for _ in range(15):
        assert d.choose(np.ones(7), 0.24) == clone.choose(np.ones(7), 0.24)
    assert d.state() == clone.state()


def test_start_cooldown_does_not_block_movement_or_other_function():
    d = decoder()
    counts = np.array([1, 0, 0, 0, 1, 0, 100])
    assert d.choose(counts, 0.24)[0] == "up+start"
    assert d.choose(counts, 0.24)[0] == "up+a"
    counts[4] = 0
    assert d.choose(counts, 0.24)[0] == "up"
    assert d.choose(counts, 0.24)[0] == "up"
    assert d.choose(counts, 0.24)[0] == "up"
    assert d.choose(counts, 0.24)[0] == "up+start"


def test_channels_keep_thresholds_and_mutual_exclusion():
    d = decoder()
    # Strong opposing direction still wins; no artificial Up boost or diagonal.
    assert d.choose(np.array([1, 2, 0, 0, 0, 0, 0]), 0.24)[0] == "down"
    d = decoder()
    assert d.choose(np.array([0.01, 0, 0, 0, 100, 0, 0]), 0.24)[0] == "a"


def test_old_decoder_state_requires_explicit_old_config():
    d = decoder("exclusive-v1")
    d.choose(np.ones(7), 0.24)
    old = d.state()
    assert "arbitration" not in old
    with pytest.raises(ValueError, match="arbitration"):
        decoder().restore(old)
    with pytest.raises(ValueError, match="arbitration"):
        decoder("exclusive-v1").restore(decoder().state())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_spikes_cannot_press_buttons(bad):
    with pytest.raises(ValueError):
        decoder().choose(np.full(7, bad), 0.24)


def test_motor_registry_fails_closed_on_a_substitute():
    brain = SimpleNamespace(cells=lambda kinds, side=None: np.array([0, 1]))
    assert len(populations(brain, np.array([10331, 16949]))["a"]) == 2
    with pytest.raises(ValueError, match="body IDs"):
        populations(brain, np.array([12, 13]))
    with pytest.raises(ValueError):
        MotorConfig(threshold_hz=0)
