from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from pokefly.dynamics import DynamicsConfig, HybridDynamics, nonvisual_sensory_cells
from pokefly.experiment import ExperimentConfig
from pokefly.internal_brain import CheckpointNoise
from pokefly.plasticity import DANTargetedPlasticity, PlasticityConfig, dan_target_gates


def toy(config=None, *, motor=False):
    matrix = sparse.csc_matrix(
        np.array([[0, 0, 0, 0], [-1, 0, 0, 0], [0, 0.2, 0.8, 0], [0, 0, 0.1, 0]], np.float32)
    )
    b = SimpleNamespace(
        n=4,
        xp=np,
        device="cpu",
        visual=np.array([0]),
        _visual=np.array([0]),
        superclass=np.array(
            ["sensory", "ol_intrinsic", "cb_intrinsic", "cb_motor" if motor else "cb_sensory"]
        ),
        weights=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        v=np.zeros((4, 1), np.float32),
        fired=np.empty(0, np.int64),
        steps=0,
        dt=0.02,
        decay=np.float32(np.exp(-0.2)),
        gain=3.0,
        tonic=0.14,
        noise_hz=0.0,
        noise_amp=0.22,
        eye_gain=0.62,
        rng=CheckpointNoise(np, 42),
    )
    return HybridDynamics(
        b,
        {"Kenyon_cells": np.array([2]), "lamina_L1_L5": np.array([1])},
        config or DynamicsConfig(profile="hybrid-v1"),
    )


def test_graded_vision_is_continuous_inhibitory_and_not_spikes():
    dark, bright = toy(), toy()
    for _ in range(100):
        assert not np.isin(dark.step(np.array([0.0])), [0, 1]).any()
        assert not np.isin(bright.step(np.array([1.0])), [0, 1]).any()
    assert dark.brain.v[1, 0] > bright.brain.v[1, 0]  # Histaminergic sign preserved.
    assert 0 < dark.release[1] < 1  # A graded value, not a binary spike.
    assert bright.release[0] > dark.release[0]
    np.testing.assert_array_equal(bright.brain.weights, dark.brain.weights)


def test_current_fast_path_does_not_bypass_calibration_or_diagnostic_overrides():
    h = toy()
    assert h.step_input is None  # Legacy lamina currents must NOT be discarded.
    release = np.ones(4, np.float32)
    full = h.current(release)
    h.step_input = SimpleNamespace(dense=lambda _: np.full((4, 1), 7, np.float32))
    np.testing.assert_array_equal(h._step_current(release), np.full((4, 1), 7))
    np.testing.assert_array_equal(h.current(release), full)  # Public measurement stays full.
    original_current = h.current
    h.current = lambda values: original_current(values) + 0.125
    np.testing.assert_array_equal(h._step_current(release), full + 0.125)


def test_adaptation_and_release_resume_exactly():
    first, restored = toy(), toy()
    first.brain.v[2] = 2.0  # A controlled initial KC spike primes adaptation.
    for _ in range(20):
        first.step(np.array([0.3]))
    restored.brain.v[:] = first.brain.v
    restored.brain.fired = first.brain.fired.copy()
    restored.brain.steps = first.brain.steps
    restored.brain.rng.generator.bit_generator.state = first.brain.rng.generator.bit_generator.state
    restored.restore({"hybrid_" + k: v.copy() for k, v in first.arrays().items()})
    for _ in range(20):
        np.testing.assert_array_equal(first.step(np.array([0.8])), restored.step(np.array([0.8])))
        np.testing.assert_array_equal(first.brain.v, restored.brain.v)
        np.testing.assert_array_equal(first.release, restored.release)
    assert first.adaptation[2] > 0
    with pytest.raises(ValueError):
        restored.restore({"hybrid_adaptation": np.full((4, 1), np.nan)})


def test_motor_adaptation_is_spike_triggered_and_anatomically_selective():
    config = DynamicsConfig(profile="hybrid-v1", motor_adaptation_increment=0.02)
    h = toy(config, motor=True)
    before = h.brain.weights.copy()
    h.brain.v[3] = 2
    assert 3 in h.step(np.array([0.3]))
    np.testing.assert_array_equal(h.adaptive_motor, [3])
    assert h.motor_adaptation[3, 0] == pytest.approx(0.02)
    assert not h.motor_adaptation[:3].any()
    h.step(np.array([0.3]))
    assert h.motor_adaptation[3, 0] == pytest.approx(0.02 * np.exp(-0.02 / 3))
    assert h.brain.v[3, 0] == pytest.approx(h.brain.tonic - h.motor_adaptation[3, 0])
    np.testing.assert_array_equal(before, h.brain.weights)
    h.reset()
    assert not h.motor_adaptation.any()
    old = toy()
    assert old.motor_adaptation is None and "motor_adaptation" not in old.arrays()


def test_motor_adaptation_resume_and_legacy_default():
    config = DynamicsConfig(profile="hybrid-v1", motor_adaptation_increment=0.02)
    a, b = toy(config, motor=True), toy(config, motor=True)
    a.brain.noise_hz = b.brain.noise_hz = 50
    for _ in range(80):
        a.step(np.array([0.3]))
    assert a.motor_adaptation[3, 0] > 0
    b.brain.v[:] = a.brain.v
    b.brain.rng.generator.bit_generator.state = a.brain.rng.generator.bit_generator.state
    b.restore({"hybrid_" + k: v.copy() for k, v in a.arrays().items()})
    for _ in range(30):
        np.testing.assert_array_equal(a.step(np.array([0.7])), b.step(np.array([0.7])))
        for key in a.arrays():
            np.testing.assert_array_equal(a.arrays()[key], b.arrays()[key])
        np.testing.assert_array_equal(a.brain.v, b.brain.v)
    bad = {"hybrid_" + k: v.copy() for k, v in a.arrays().items()}
    bad["hybrid_motor_adaptation"][0] = 0.1
    with pytest.raises(ValueError, match="outside"):
        b.restore(bad)
    assert (
        ExperimentConfig.from_dict({}, checkpoint=True).brain.dynamics.motor_adaptation_increment
        == 0
    )
    with pytest.raises(ValueError):
        DynamicsConfig(motor_adaptation_increment=0.02)
    with pytest.raises(ValueError):
        DynamicsConfig(profile="hybrid-v1", motor_adaptation_seconds=0)


def test_config_roundtrip_and_validation():
    raw = {
        "brain": {"dynamics": {"profile": "hybrid-v1"}, "plasticity": {"rule": "dan-targeted-v1"}}
    }
    value = ExperimentConfig.from_dict(raw)
    assert ExperimentConfig.from_dict(asdict(value)) == value
    for bad in (
        {"profile": "unknown"},
        {"graded_release": 0},
        {"kc_tonic": 1},
        {"graded_seconds": float("nan")},
        {"isolate_nonvisual_sensory": "false"},
        {"isolate_nonvisual_sensory": 1},
        {"isolate_nonvisual_sensory": True},  # Baseline must not silently ignore it.
        {"spike_temperature": 0.05},
        {"profile": "hybrid-v1", "spike_temperature": -1},
    ):
        with pytest.raises(ValueError):
            DynamicsConfig(**bad)


def test_stochastic_spikes_report_actual_probability_and_resume_exactly():
    config = DynamicsConfig(profile="hybrid-v1", spike_temperature=0.05)
    first, restored = toy(config), toy(config)
    for h in (first, restored):
        h.track_score = True
    for _ in range(20):
        first.step(np.array([0.3]))
    restored.brain.v[:] = first.brain.v
    restored.brain.rng.generator.bit_generator.state = first.brain.rng.generator.bit_generator.state
    restored.restore({"hybrid_" + k: v.copy() for k, v in first.arrays().items()})
    for _ in range(20):
        previous = first.release.copy()
        np.testing.assert_array_equal(first.step(np.array([0.8])), restored.step(np.array([0.8])))
        np.testing.assert_array_equal(first.previous_release, previous)
        np.testing.assert_array_equal(first.brain.v, restored.brain.v)
        np.testing.assert_array_equal(first.last_probability, restored.last_probability)
        assert np.all(first.last_probability[:2] == 0)  # Graded vision never spikes.
        assert np.all((first.last_probability >= 0) & (first.last_probability <= 1))


def test_isolation_selects_sensory_annotations_but_preserves_visual_and_motor_cells():
    brain = SimpleNamespace(
        n=6,
        visual=np.array([0]),
        superclass=np.array(
            [
                "sensory",
                "cb_sensory",
                "vnc_sensory",
                "sensory_descending",
                "cb_intrinsic",
                "descending_neuron",
            ]
        ),
    )
    np.testing.assert_array_equal(nonvisual_sensory_cells(brain), [1, 2, 3])
    with pytest.raises(ValueError, match="annotations"):
        nonvisual_sensory_cells(SimpleNamespace(n=6, visual=np.array([0])))
    with pytest.raises(ValueError, match="no nonvisual"):
        nonvisual_sensory_cells(SimpleNamespace(n=1, visual=np.array([0]), superclass=["sensory"]))


def test_isolation_only_masks_incoming_current_without_changing_weights_or_outgoing_signals():
    config = DynamicsConfig(profile="hybrid-v1", isolate_nonvisual_sensory=True)
    h = toy(config)
    original = h.brain.weights.copy()
    release = np.array([0.7, 0.3, 1.0, 0.4], np.float32)
    reference = h.matrix @ release
    measured = h.current(release)[:, 0]
    np.testing.assert_allclose(measured[:3], reference[:3])
    assert reference[3] != 0 and measured[3] == 0
    np.testing.assert_array_equal(h.brain.weights, original)
    # A sensory -> KC edge must still conduct; columns are NOT masked.
    h.matrix = sparse.csc_matrix(([0.4], ([2], [3])), shape=(4, 4), dtype=np.float32)
    assert h.current(np.array([0, 0, 0, 1], np.float32))[2, 0] == np.float32(0.4)


def test_isolated_neurons_keep_tonic_and_neural_noise_and_resume():
    config = DynamicsConfig(profile="hybrid-v1", isolate_nonvisual_sensory=True)
    first, restored = toy(config), toy(config)
    for h in (first, restored):
        h.brain.noise_hz = 50  # Deterministic diagnostic: a noise increment every step.
    first.step(np.array([0.2]))
    assert first.brain.v[3, 0] == pytest.approx(first.brain.tonic + first.brain.noise_amp)
    for _ in range(20):
        first.step(np.array([0.4]))
    restored.brain.v[:] = first.brain.v
    restored.brain.fired = first.brain.fired.copy()
    restored.brain.rng.generator.bit_generator.state = first.brain.rng.generator.bit_generator.state
    restored.restore({"hybrid_" + k: v.copy() for k, v in first.arrays().items()})
    for _ in range(20):
        np.testing.assert_array_equal(first.step(np.array([0.8])), restored.step(np.array([0.8])))
        np.testing.assert_array_equal(first.brain.v, restored.brain.v)


def test_isolation_config_is_explicit_and_old_checkpoints_default_off():
    original = ExperimentConfig.from_dict(
        {"brain": {"dynamics": {"profile": "hybrid-v1"}}}, checkpoint=True
    )
    assert original.brain.dynamics.isolate_nonvisual_sensory is False
    isolated = replace(
        original,
        brain=replace(
            original.brain,
            dynamics=replace(original.brain.dynamics, isolate_nonvisual_sensory=True),
        ),
    )
    assert ExperimentConfig.from_dict(asdict(isolated), checkpoint=True) == isolated


def test_quiescent_boundary_is_explicit_and_consumes_same_noise_stream():
    base = DynamicsConfig(profile="hybrid-v1", isolate_nonvisual_sensory=True)
    original, quiet = toy(base), toy(replace(base, quiescent_nonvisual_sensory=True))
    for h in (original, quiet):
        h.brain.noise_hz = 50
    for _ in range(30):
        original.step(np.array([0.8]))
        quiet.step(np.array([0.8]))
        assert quiet.brain.v[3, 0] == 0 and quiet.release[3] == 0
        assert 3 not in quiet.brain.fired
        assert (
            quiet.brain.rng.generator.bit_generator.state
            == original.brain.rng.generator.bit_generator.state
        )
    assert quiet.release[0] > 0  # Photoreceptor remains active.
    np.testing.assert_array_equal(original.brain.weights, quiet.brain.weights)
    with pytest.raises(ValueError, match="requires"):
        DynamicsConfig(profile="hybrid-v1", quiescent_nonvisual_sensory=True)


def targeted():
    return DANTargetedPlasticity(
        np.array([0, 1]),
        np.array([2, 2]),
        np.array([0.2, 0.3]),
        3,
        PlasticityConfig(rule="dan-targeted-v1"),
        positive_gate=np.array([1.0, 0.0]),
        negative_gate=np.array([0.0, 1.0]),
    )


def test_anatomical_targeting_and_depression_recovery():
    p = targeted()
    p.reinforce(1)
    np.testing.assert_array_equal(p.weights, p.base)
    for _ in range(100):
        p.observe(np.array([0, 1]), 0.02)
    p.reinforce(1, enabled=False)
    np.testing.assert_array_equal(p.weights, p.base)
    p.reinforce(1)
    assert p.weights[0] < p.base[0] and p.weights[1] == p.base[1]
    p.reinforce(-1)
    assert p.weights[1] < p.base[1]
    depressed = p.weights.copy()
    for _ in range(2000):
        p.observe(np.array([], np.int64), 0.02)
    p.reinforce(1)
    assert depressed[0] < p.weights[0] <= p.base[0]
    assert p.weights[1] == depressed[1]
    restored = targeted()
    restored.restore(p.arrays(), p.metrics())
    np.testing.assert_array_equal(restored.weights, p.weights)
    with pytest.raises(ValueError, match="targeting"):
        restored.restore(dict(p.arrays(), positive_gate=np.zeros(2)), p.metrics())


def test_dan_gate_requires_shared_existing_targets():
    matrix = sparse.csc_matrix(
        (np.array([0.4, 0.6, 0.2, 0.8]), (np.array([0, 2, 1, 2]), np.array([3, 3, 4, 4]))),
        shape=(5, 5),
    )
    b = SimpleNamespace(
        n=5,
        weights=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        cell_type=np.array(["KCg-d", "KCg-d", "MBON01", "PAM01", "PPL101"]),
    )
    positive, negative = dan_target_gates(b, np.array([0, 1]), np.array([2, 2]))
    np.testing.assert_array_equal(positive, [1, 0])
    np.testing.assert_array_equal(negative, [0, 1])
