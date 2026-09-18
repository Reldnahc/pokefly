from types import SimpleNamespace

import numpy as np
import pytest

from pokefly.pixel_brain import PixelBrain, paired_changes


def test_probe_checks_spike_timing_not_only_totals():
    first = {"cells": np.array([[True, False], [False, True]])}
    second = {"cells": np.array([[False, True], [True, False]])}
    delta = paired_changes(first, second)["cells"]
    assert delta == {
        "different_neuron_steps": 4,
        "affected_neurons": 2,
        "first_spikes": 2,
        "second_spikes": 2,
    }
    assert paired_changes(first, first)["cells"]["different_neuron_steps"] == 0


def test_probe_rejects_mismatched_populations():
    with pytest.raises(ValueError):
        paired_changes({"a": np.zeros((2, 2))}, {"b": np.zeros((2, 2))})
    with pytest.raises(ValueError):
        paired_changes({"a": np.zeros((2, 2))}, {"a": np.zeros((1, 2))})


def test_pixel_observer_only_passes_fixed_retinal_drive():
    frame = np.zeros((144, 160, 4), np.uint8)
    drive = np.array([0.25, 0.75], np.float32)
    calls = []

    def encode(received):
        assert received is frame
        return drive

    def step(**kwargs):
        calls.append(kwargs)
        return np.array([1, 3])

    controller = PixelBrain.__new__(PixelBrain)
    controller.retina = SimpleNamespace(encode=encode)
    controller.brain = SimpleNamespace(n=5, step=step)
    controller.brain_steps = 3
    controller.groups = {"test": np.array([1, 4])}
    observation = controller.observe(frame)
    assert all(set(call) == {"eye_drive"} and call["eye_drive"] is drive for call in calls)
    np.testing.assert_equal(observation.counts, [0, 3, 0, 3, 0])
    assert observation.groups == {"test": 3}
    assert observation.spikes == 6


def test_invalid_brain_steps_fails_before_loading_model():
    with pytest.raises(ValueError, match="positive"):
        PixelBrain(brain_steps=0)
