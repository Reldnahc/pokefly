import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pokefly.checkpoint import sha256
from pokefly.dynamics import DynamicsConfig
from pokefly.internal_brain import BrainConfig
from pokefly.plasticity import PlasticityConfig
from pokefly.release_reference import (
    INTRINSIC_KEYS,
    REFERENCE_PLASTICITY,
    REFERENCE_RULE,
    intrinsic_payload_sha256,
    load_release_reference,
    physical_config,
    source_data_hashes,
    verify_intrinsic_copy,
)
from scripts import collect_release_reference


@pytest.fixture
def reference_artifact(tmp_path):
    config = BrainConfig(dynamics=DynamicsConfig(profile="hybrid-v1"),
                         intrinsic_calibration="original.npz",
                         plasticity=PlasticityConfig(rule=REFERENCE_PLASTICITY))
    data = tmp_path / "data"
    data.mkdir()
    for name in ("brain.npz", "weights.npz", "retina.npz"):
        (data / name).write_bytes(name.encode())
    arrays = {"body_ids": np.array([10, 11]), "bias": np.array([[.05], [-.2]], np.float32),
              "mask": np.array([True, True]), "protocol": np.array('{"rule":"original"}')}
    original = tmp_path / "original.npz"
    np.savez(original, **arrays)
    protocol = {
        "rule": REFERENCE_RULE, "game_frames_used": False, "rewards_used": False,
        "actions_selected": 0, "physical_config": physical_config(config),
        "intrinsic_payload_sha256": intrinsic_payload_sha256(arrays),
        "source_intrinsic_sha256": sha256(original),
        "data_sha256": source_data_hashes(data, config),
    }
    arrays.update(mean_release=np.array([.2, .8], np.float32),
                  reference_protocol=np.array(json.dumps(protocol)))
    return SimpleNamespace(path=tmp_path / "reference.npz", original=original, arrays=arrays,
                           protocol=protocol, config=config, data=data)


def write_and_load(artifact, *, config=None, body_ids=None):
    np.savez_compressed(artifact.path, **artifact.arrays)
    return load_release_reference(artifact.path, [10, 11] if body_ids is None else body_ids,
                                  config or artifact.config, data=artifact.data)


def test_reference_identity_is_compression_independent_but_payload_exact(reference_artifact):
    a = reference_artifact
    mean, info = write_and_load(a)
    np.testing.assert_array_equal(mean, a.arrays["mean_release"])
    assert info["rule"] == REFERENCE_RULE and info["sha256"] == sha256(a.path)
    assert intrinsic_payload_sha256(a.arrays) == a.protocol["intrinsic_payload_sha256"]
    with np.load(a.path, allow_pickle=False) as stored:
        assert intrinsic_payload_sha256(stored) == intrinsic_payload_sha256(a.arrays)
    changed = {key: a.arrays[key].copy() for key in INTRINSIC_KEYS}
    changed["bias"] = changed["bias"].astype(np.float64)
    assert intrinsic_payload_sha256(changed) != intrinsic_payload_sha256(a.arrays)


@pytest.mark.parametrize("mean", [[.2], [[.2], [.8]], [np.nan, .8], [.2, np.inf],
                                  [-.1, .8], [.2, 1.1], [0, 0]])
def test_reference_rejects_nonphysical_means(reference_artifact, mean):
    a = reference_artifact
    a.arrays["mean_release"] = np.array(mean)
    with pytest.raises(ValueError, match="Invalid fixed"):
        write_and_load(a)


@pytest.mark.parametrize("key,value", [
    ("rule", "unregistered"), ("game_frames_used", True), ("rewards_used", True),
    ("actions_selected", 1), ("intrinsic_payload_sha256", "wrong"),
])
def test_reference_rejects_wrong_collection_provenance(reference_artifact, key, value):
    a = reference_artifact
    a.protocol[key] = value
    a.arrays["reference_protocol"] = np.array(json.dumps(a.protocol))
    with pytest.raises(ValueError, match="provenance"):
        write_and_load(a)


def test_reference_checks_exact_neuron_model_offsets_and_connectome(reference_artifact):
    a = reference_artifact
    with pytest.raises(ValueError, match="neuron identity"):
        write_and_load(a, body_ids=[11, 10])
    with pytest.raises(ValueError, match="physical model"):
        write_and_load(a, config=replace(a.config, noise_hz=.5))
    a.arrays["bias"][0] += .01
    with pytest.raises(ValueError, match="provenance"):
        write_and_load(a)
    # Restore the exact recorded payload rather than rely on round-trip float arithmetic.
    a.arrays["bias"] = np.array([[.05], [-.2]], np.float32)
    (a.data / "weights.npz").write_bytes(b"different graph weights")
    with pytest.raises(ValueError, match="source asset identity"):
        write_and_load(a)


def test_reference_binds_retina_and_every_calibrated_visual_asset(reference_artifact):
    a = reference_artifact
    a.config = replace(a.config, dynamics=replace(a.config.dynamics,
                                                visual_model="calibrated-rate-v1"))
    directory = a.data / "visual-reference-c28066a5"
    directory.mkdir()
    files = ("manifest.json", "optic-v2.json", "optic-v2.bin",
             "fitted-params.json", "flyvis-params.json")
    for name in files:
        (directory / name).write_bytes(name.encode())
    a.protocol["physical_config"] = physical_config(a.config)
    a.protocol["data_sha256"] = source_data_hashes(a.data, a.config)
    a.arrays["reference_protocol"] = np.array(json.dumps(a.protocol))
    write_and_load(a)
    for path in [a.data / "retina.npz", *(directory / name for name in files)]:
        original = path.read_bytes()
        path.write_bytes(b"altered physical asset")
        with pytest.raises(ValueError, match="source asset identity"):
            write_and_load(a)
        path.write_bytes(original)


@pytest.mark.parametrize("missing", [*INTRINSIC_KEYS, "mean_release", "reference_protocol"])
def test_reference_missing_fields_fail_closed(reference_artifact, missing):
    a = reference_artifact
    del a.arrays[missing]
    with pytest.raises(ValueError, match="requires a frozen generic"):
        write_and_load(a)


def test_reference_excludes_only_nonphysical_settings(reference_artifact):
    a = reference_artifact
    altered = replace(a.config, brain_steps=24, intrinsic_calibration="new-copy.npz",
                      plasticity=PlasticityConfig(rule="sensorimotor-perturb-homeostatic-v5"))
    assert physical_config(altered) == physical_config(a.config)
    write_and_load(a, config=altered)
    with pytest.raises(ValueError, match="explicit reference"):
        replace(a.config, intrinsic_calibration=None)


def test_reference_extension_cannot_change_or_chain_intrinsic_payloads(reference_artifact):
    a = reference_artifact
    write_and_load(a)
    copied = verify_intrinsic_copy(a.original, a.path)
    assert copied["original_sha256"] == sha256(a.original)
    assert copied["extended_sha256"] == sha256(a.path)
    with pytest.raises(ValueError, match="unaugmented"):
        verify_intrinsic_copy(a.path, a.path)
    a.arrays["bias"][0] += .01
    # Even a self-consistent replacement is not the original frozen physiology.
    a.protocol["intrinsic_payload_sha256"] = intrinsic_payload_sha256(a.arrays)
    a.arrays["reference_protocol"] = np.array(json.dumps(a.protocol))
    write_and_load(a)
    with pytest.raises(ValueError, match="exact original intrinsic"):
        verify_intrinsic_copy(a.original, a.path)


def test_reference_experiment_changes_only_the_declared_internal_reference(
    reference_artifact, monkeypatch,
):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("evaluate_visual_learning_rate")
    a = reference_artifact
    write_and_load(a)
    old = asdict(module.load_config(Path("configs/visual-wide-homeostatic-v8.json")))
    new = asdict(module.load_config(Path("configs/visual-wide-anchored-v9.json")))
    old["brain"]["intrinsic_calibration"] = str(a.original)
    new["brain"]["intrinsic_calibration"] = str(a.path)
    assert module.validate_one_factor(old, new, "release_reference") == (
        "sensorimotor-perturb-homeostatic-v5", REFERENCE_PLASTICITY,
    )
    with pytest.raises(ValueError, match="ONLY"):
        module.validate_one_factor(old, new, "rule")  # Never silently accept a new asset.
    new["rewards"]["new_tile"] *= 2
    with pytest.raises(ValueError, match="ONLY"):
        module.validate_one_factor(old, new, "release_reference")
    new["rewards"]["new_tile"] /= 2
    new["brain"]["noise_amplitude"] *= 2
    with pytest.raises(ValueError, match="ONLY"):
        module.validate_one_factor(old, new, "release_reference")
    new["brain"]["noise_amplitude"] /= 2
    new["brain"]["intrinsic_calibration"] = str(a.original)
    with pytest.raises(ValueError, match="explicit moving-to-fixed"):
        module.validate_one_factor(old, new, "release_reference")


def test_collection_balances_generic_patterns_and_never_calls_game_learning_or_decoder(monkeypatch):
    monkeypatch.setattr(collect_release_reference, "test_patterns", lambda: {
        "black": np.zeros((144, 160, 3), np.uint8),
        "white": np.full((144, 160, 3), 255, np.uint8),
    })
    hybrid = SimpleNamespace(release=np.zeros(2, np.float32))
    calls, resets = [], []

    def step(*, eye_drive):
        calls.append(eye_drive)
        hybrid.release[:] = [eye_drive, 1 - eye_drive]

    def forbidden(*args, **kwargs):
        raise AssertionError("Generic collection must not choose buttons or learn")

    controller = SimpleNamespace(
        brain=SimpleNamespace(xp=np, n=2, step=step), hybrid=hybrid,
        retina=SimpleNamespace(encode=lambda frame: float(frame[0, 0, 0]) / 255),
        reset_dynamics=resets.append, observe=forbidden, choose=forbidden, reinforce=forbidden,
        decoder=SimpleNamespace(choose=forbidden), plasticity=SimpleNamespace(observe=forbidden),
    )
    mean, schedule, samples = collect_release_reference.collect(
        controller, cycles=2, warmup=2, scored_steps=3, seed=42,
    )
    expected = (128 / 255 + 1) / 3
    np.testing.assert_allclose(mean, [expected, 1 - expected], rtol=1e-7)
    assert samples == 18 and len(calls) == 30 and resets == [42]
    for cycle in range(2):
        assert sorted(row["stimulus"] for row in schedule if row["cycle"] == cycle) == [
            "black", "gray", "white",
        ]
    repeated, repeated_schedule, _ = collect_release_reference.collect(
        controller, cycles=2, warmup=2, scored_steps=3, seed=42,
    )
    np.testing.assert_array_equal(mean, repeated)
    assert repeated_schedule == schedule


@pytest.mark.parametrize("cycles,warmup,scored", [(0, 0, 1), (1, -1, 1), (1, 0, 0)])
def test_collection_rejects_invalid_budgets_before_touching_controller(cycles, warmup, scored):
    with pytest.raises(ValueError, match="Positive cycles"):
        collect_release_reference.collect(None, cycles=cycles, warmup=warmup,
                                          scored_steps=scored, seed=42)
