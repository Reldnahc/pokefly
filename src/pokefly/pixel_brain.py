"""Raw-pixel neural observation and matched-noise visual-pathway diagnostics.

This path has NO button policy, learned encoder, RAM access, or plasticity.
The old external-readout experiment remains in brain.py / runner.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pokefly.runtime import configure_runtime
from pokefly.vision import RetinaMap


@dataclass
class PixelObservation:
    drive: np.ndarray
    counts: np.ndarray
    groups: dict[str, int]
    steps: int
    continuous: dict | None = None

    @property
    def fired(self) -> np.ndarray:
        return np.flatnonzero(self.counts)

    @property
    def spikes(self) -> int:
        return int(self.counts.sum())


class PixelBrain:
    def __init__(self, *, device: str = "auto", seed: int = 64, brain_steps: int = 12):
        if brain_steps < 1:
            raise ValueError("brain_steps must be positive")
        data = configure_runtime()
        if not (data / "retina.npz").is_file():
            raise ValueError("Retina is missing; run 'pokefly vision-prepare' first")
        self.retina = RetinaMap.load(data / "retina.npz", data / "brain.npz")
        from flybrain import FlyBrain

        self.brain = FlyBrain(data=data, device=device, seed=seed)
        self.brain_steps = brain_steps
        cell_type = self.brain.cell_type.astype(str)
        self.groups = {
            "photoreceptors": self.brain.visual,
            "lamina_L1_L5": self.brain.cells(["L1", "L2", "L3", "L4", "L5"]),
            "visual_projection": self.brain.cells(["visual_projection"]),
            "Kenyon_cells": np.flatnonzero(np.char.startswith(cell_type, "KC")),
            "MBONs": np.flatnonzero(np.char.startswith(cell_type, "MBON")),
            "descending": self.brain.cells(["descending_neuron"]),
            "forward_Up": self.brain.cells(["DNg100"]),
            "backward_Down": self.brain.cells(["MDN"]),
            "steer_Left": self.brain.cells(["DNa02"], side="L"),
            "steer_Right": self.brain.cells(["DNa02"], side="R"),
            "proboscis_A": self.brain.cells(["CB0701"]),
            "escape_B": self.brain.cells(["DNp01"]),
            "song_Start": self.brain.cells(["pIP10"]),
        }

    def observe(self, frame: np.ndarray) -> PixelObservation:
        drive = self.retina.encode(frame)
        counts = np.zeros(self.brain.n, np.int32)
        for _ in range(self.brain_steps):
            fired = self.brain.step(eye_drive=drive)
            counts[fired] += 1
        return PixelObservation(
            drive,
            counts,
            {name: int(counts[indices].sum()) for name, indices in self.groups.items()},
            self.brain_steps,
        )


def test_patterns() -> dict[str, np.ndarray]:
    black = np.zeros((144, 160, 3), np.uint8)
    white = np.full_like(black, 255)
    left, top = black.copy(), black.copy()
    left[:, :80] = 255
    top[:72] = 255
    yy, xx = np.indices(black.shape[:2])
    checker = np.repeat((((xx // 8 + yy // 8) % 2) * 255)[:, :, None], 3, axis=2)
    return {
        "black": black,
        "white": white,
        "left": left,
        "right": 255 - left,
        "top": top,
        "bottom": 255 - top,
        "checker": checker.astype(np.uint8),
    }


def paired_changes(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, dict[str, int]]:
    """Compare *neuron x timestep* spike identities, not just population totals."""
    if first.keys() != second.keys():
        raise ValueError("Probe populations do not match")
    result = {}
    for group, a in first.items():
        b = second[group]
        if a.shape != b.shape:
            raise ValueError("Probe timesteps or populations do not match")
        difference = a != b
        result[group] = {
            "different_neuron_steps": int(difference.sum()),
            "affected_neurons": int(difference.any(axis=0).sum()),
            "first_spikes": int(a.sum()),
            "second_spikes": int(b.sum()),
        }
    return result


def probe_vision(*, device: str, seed: int, steps: int, trials: int) -> Path:
    """Necessary sensitivity checks, NOT proof of sight or biological validity."""
    if steps < 2 or trials < 1:
        raise ValueError("Probe steps must be >= 2 and trials must be positive")
    from pokefly.runner import run_directory, write_json

    controller = PixelBrain(device=device, seed=seed)
    output = run_directory("vision-probe")
    patterns = test_patterns()
    comparisons = []
    for trial in range(trials):
        records = {}
        # A repeated black condition detects uncontrolled noise/non-determinism.
        conditions = [*patterns, "black_repeat", "bypass_LPLC2_positive_control"]
        for name in conditions:
            controller.brain.reset(seed + trial)
            frame = patterns.get(name, patterns["black"])
            drive = controller.retina.encode(frame)
            traces = {
                key: np.zeros((steps, len(idx)), bool) for key, idx in controller.groups.items()
            }
            injection = []
            if name == "bypass_LPLC2_positive_control":
                # Intentionally bypasses the retina ONLY in this labeled control.
                injection = [(controller.brain.cells(["LPLC2"]), 0.8)]
            for step in range(steps):
                fired = controller.brain.step(eye_drive=drive, inject=injection)
                mask = np.zeros(controller.brain.n, bool)
                mask[fired] = True
                for group, idx in controller.groups.items():
                    traces[group][step] = mask[idx]
            records[name] = traces
            print(f"probe seed={seed + trial} stimulus={name} complete", flush=True)
        for first, second in (
            ("black", "black_repeat"),
            ("black", "white"),
            ("left", "right"),
            ("top", "bottom"),
            ("black", "checker"),
            ("black", "bypass_LPLC2_positive_control"),
        ):
            comparisons.append(
                {
                    "seed": seed + trial,
                    "first": first,
                    "second": second,
                    "groups": paired_changes(records[first], records[second]),
                }
            )
    repeatable = all(
        all(g["different_neuron_steps"] == 0 for g in pair["groups"].values())
        for pair in comparisons
        if pair["second"] == "black_repeat"
    )
    spatial = [p for p in comparisons if p["first"] in ("left", "top")]
    motor_sensitive = all(
        pair["groups"]["descending"]["different_neuron_steps"] > 0 for pair in spatial
    )
    learning_sensitive = all(
        pair["groups"]["Kenyon_cells"]["different_neuron_steps"] > 0 for pair in spatial
    )
    report = {
        "device": controller.brain.device,
        "steps_per_stimulus": steps,
        "seconds_per_stimulus": steps * controller.brain.dt,
        "seeds": list(range(seed, seed + trials)),
        "retina": controller.retina.summary(),
        "populations": {k: len(v) for k, v in controller.groups.items()},
        "matched_noise_repeatable": repeatable,
        "spatial_descending_sensitivity": motor_sensitive,
        "spatial_Kenyon_sensitivity": learning_sensitive,
        "necessary_sensitivity_gate": repeatable and motor_sensitive and learning_sensitive,
        "minimum_spatially_affected_Kenyon_fraction": min(
            pair["groups"]["Kenyon_cells"]["affected_neurons"]
            / max(1, len(controller.groups["Kenyon_cells"]))
            for pair in spatial
        ),
        "candidate_motor_minimum_affected_neurons": {
            group: min(pair["groups"][group]["affected_neurons"] for pair in spatial)
            for group in controller.groups
            if group.endswith(("_Up", "_Down", "_Left", "_Right", "_A", "_B", "_Start"))
        },
        "ready_for_autonomous_control": False,
        "A_neurons_identified": True,
        "biological_vision_validated": False,
        "internal_learning_enabled": False,
        "scope": "Static patterns; same seed per pair; default upstream LIF dynamics. "
        "Tests spike output, not subthreshold voltage or understanding. "
        "LPLC2 bypass is a separate diagnostic, never the pixel controller.",
        "comparisons": comparisons,
    }
    write_json(output / "report.json", report)
    print(json_summary(report), flush=True)
    print(f"Full visual-pathway report: {output / 'report.json'}", flush=True)
    return output


def json_summary(report: dict) -> str:
    import json

    keys = (
        "matched_noise_repeatable",
        "spatial_descending_sensitivity",
        "spatial_Kenyon_sensitivity",
        "necessary_sensitivity_gate",
    )
    return json.dumps({key: report[key] for key in keys}, indent=2)
