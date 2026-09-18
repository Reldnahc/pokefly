"""Explicit hybrid experiment, not a fitted or biologically validated fly model.

Original weights/signs remain; an explicit experimental gate can isolate incoming
network current to nonvisual sensory cells. Photoreceptors and L1--L5 use graded
release; KC-specific spike adaptation reduces recurrent self-sustaining firing.
Constants are engineering hypotheses, frozen before any gameplay evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class DynamicsConfig:
    profile: str = "baseline"
    graded_vision: bool = True
    graded_seconds: float = 0.04
    graded_release: float = 0.25
    lamina_rest: float = 0.6
    kc_tonic: float = 0.02
    kc_adaptation_increment: float = 0.25
    kc_adaptation_seconds: float = 1.0
    isolate_nonvisual_sensory: bool = False
    quiescent_nonvisual_sensory: bool = False
    spike_temperature: float = 0.0
    motor_adaptation_increment: float = 0.0
    motor_adaptation_seconds: float = 3.0
    visual_model: str = "legacy-v1"

    def __post_init__(self):
        if self.profile not in ("baseline", "hybrid-v1"):
            raise ValueError("Unknown dynamics profile")
        if self.visual_model not in ("legacy-v1", "calibrated-rate-v1"):
            raise ValueError("Unknown internal visual-neuron model")
        if self.visual_model != "legacy-v1" and (
            self.profile != "hybrid-v1" or not self.graded_vision
        ):
            raise ValueError("Calibrated visual neurons require graded hybrid dynamics")
        if not isinstance(self.isolate_nonvisual_sensory, bool):
            raise ValueError("isolate_nonvisual_sensory must be boolean")
        if not isinstance(self.quiescent_nonvisual_sensory, bool):
            raise ValueError("quiescent_nonvisual_sensory must be boolean")
        if self.quiescent_nonvisual_sensory and not self.isolate_nonvisual_sensory:
            raise ValueError("Quiescent sensory boundary requires incoming isolation")
        values = {k: v for k, v in asdict(self).items() if k not in ("profile", "visual_model")}
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Dynamics parameters must be finite")
        if not isinstance(self.graded_vision, bool):
            raise ValueError("graded_vision must be boolean")
        if self.isolate_nonvisual_sensory and self.profile != "hybrid-v1":
            raise ValueError("Sensory isolation requires hybrid-v1 dynamics")
        if not 0 <= self.spike_temperature <= 0.5 or (
            self.spike_temperature and self.profile != "hybrid-v1"
        ):
            raise ValueError("Stochastic spiking requires hybrid dynamics and temperature <= 0.5")
        if min(self.graded_seconds, self.kc_adaptation_seconds) <= 0:
            raise ValueError("Dynamics time constants must be positive")
        if not 0 < self.graded_release <= 1 or not 0 <= self.lamina_rest <= 1:
            raise ValueError("Invalid graded release/rest")
        if not 0 <= self.kc_tonic <= 0.14 or not 0 <= self.kc_adaptation_increment <= 2:
            raise ValueError("Invalid KC dynamics")
        if (
            not 0 <= self.motor_adaptation_increment <= 2
            or self.motor_adaptation_seconds <= 0
            or (self.motor_adaptation_increment and self.profile != "hybrid-v1")
        ):
            raise ValueError("Motor adaptation requires hybrid dynamics and valid constants")


def nonvisual_sensory_cells(brain) -> np.ndarray:
    """Anatomical selection used in the original audit; never game/motor dependent."""
    classes = np.asarray(getattr(brain, "superclass", None))
    if classes.shape != (brain.n,):
        raise ValueError("Sensory isolation requires per-neuron superclass annotations")
    selected = np.char.find(classes.astype(str), "sensory") >= 0
    selected[brain.visual] = False  # Preserve every original visual input cell.
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("Sensory isolation selected no nonvisual sensory cells")
    return indices


class HybridDynamics:
    def __init__(self, brain, groups, config: DynamicsConfig):
        self.brain, self.config = brain, config
        xp = brain.xp
        self.kc = xp.asarray(groups["Kenyon_cells"])
        self.receptors = xp.asarray(brain.visual if config.graded_vision else [], dtype=xp.int64)
        self.lamina = xp.asarray(
            groups["lamina_L1_L5"] if config.graded_vision else [], dtype=xp.int64
        )
        self.graded = xp.concatenate((self.receptors, self.lamina))
        self.graded_host = self.graded if xp is np else self.graded.get()
        self.visual_circuit = None
        if config.visual_model == "calibrated-rate-v1":
            from pokefly.calibrated_vision import CalibratedVisualCircuit

            self.visual_circuit = CalibratedVisualCircuit(device=brain.device)
            if np.setdiff1d(self.graded_host, self.visual_circuit.global_indices).size:
                raise ValueError("Calibrated circuit must cover all original graded visual cells")
            self.graded_host = self.visual_circuit.global_indices
            self.graded = xp.asarray(self.graded_host)
        self.isolated_host = (
            nonvisual_sensory_cells(brain)
            if config.isolate_nonvisual_sensory
            else np.empty(0, np.int64)
        )
        self.isolated = xp.asarray(self.isolated_host)
        self.matrix = (
            sparse.csc_matrix(
                (brain.weights, brain.indices, brain.indptr), shape=(brain.n, brain.n), copy=False
            )
            if brain.device == "cpu"
            else None
        )
        self.intrinsic_bias = None  # Optional fixed, neutral-calibration model data.
        self.track_noise = False
        self.track_score = False
        # Anatomy only; not the button registry. Optional intrinsic current,
        # independent of reward, image semantics, decoder state or game RAM.
        self.adaptive_motor = xp.asarray(
            np.flatnonzero(
                np.isin(
                    brain.superclass.astype(str), ["descending_neuron", "cb_motor", "vnc_motor"]
                )
            )
            if config.motor_adaptation_increment
            else [],
            dtype=xp.int64,
        )
        self.reset()

    def reset(self):
        b = self.brain
        self.adaptation = b.xp.zeros((b.n, 1), b.xp.float32)
        self.release = b.xp.zeros(b.n, b.xp.float32)
        self.last_noise = None  # Consumed in the same neural step; not persistent state.
        self.last_probability = self.previous_release = None
        self.motor_adaptation = (
            b.xp.zeros((b.n, 1), b.xp.float32) if self.config.motor_adaptation_increment else None
        )
        if self.visual_circuit is not None:
            self.visual_circuit.reset()
            normalized = self.visual_circuit.rate / np.float32(self.visual_circuit.maximum_rate)
            b.v[self.graded, 0] = normalized
            self.release[self.graded] = normalized * np.float32(self.config.graded_release)

    def current(self, release):
        if self.brain.device == "cuda":
            result = self.brain.synaptic_input.dense(release)
        else:
            result = (self.matrix @ release)[:, None]
        if len(self.isolated_host):
            # Incoming network current only: no weight edits, cell silencing,
            # outgoing mask, tonic/noise change, or normalization change.
            result[self.isolated, 0] = 0
        return result

    def step(self, eye_drive=None, inject=()):
        b, c = self.brain, self.config
        xp = b.xp
        if self.track_score:
            self.previous_release = self.release.copy() if xp is np else self.release.get()
        current = self.current(self.release) * b.gain
        if self.intrinsic_bias is not None:
            current += self.intrinsic_bias
        self.adaptation *= np.float32(np.exp(-b.dt / c.kc_adaptation_seconds))
        # Preserve the previous graded state while integrating the spiking cells.
        old_graded = b.v[self.graded].copy()
        b.v *= b.decay
        b.v += current + b.tonic - self.adaptation
        if self.motor_adaptation is not None:
            self.motor_adaptation *= np.float32(np.exp(-b.dt / c.motor_adaptation_seconds))
            b.v -= self.motor_adaptation
        b.v[self.kc] += np.float32(c.kc_tonic - b.tonic)
        noise = b.rng.random((b.n, 1)) < b.noise_hz * b.dt
        b.v += noise * np.float32(b.noise_amp)
        if self.track_noise:
            self.last_noise = noise[:, 0] if xp is np else noise[:, 0].get()
        if self.visual_circuit is not None:
            vision = self.visual_circuit
            if eye_drive is None:
                eye_drive = np.zeros(len(b.visual), np.float32)
            # Fixed dimensional conversion: reference [0,5] -> old graded
            # [0,1] state; maximum release stays unchanged at 0.25 by default.
            # Existing outside -> visual edges supply current without counting
            # any original visual-internal connection a second time.
            rate = vision.advance_drive(
                eye_drive, b.dt, external_current=vision.incoming_current(self.release, b.gain)
            )
            b.v[self.graded, 0] = rate / np.float32(vision.maximum_rate)
        elif c.graded_vision:
            target = xp.clip(c.lamina_rest + current[self.graded], 0, 1)
            if eye_drive is None:
                eye_drive = np.zeros(len(b.visual), np.float32)
            drive = xp.asarray(eye_drive, dtype=xp.float32).reshape(-1, 1)
            # Light enters the original receptors; no features or neuron bypass.
            target[: len(self.receptors)] = xp.clip(drive + current[self.receptors], 0, 1)
            decay = np.float32(np.exp(-b.dt / c.graded_seconds))
            b.v[self.graded] = decay * old_graded + (1 - decay) * target
        elif eye_drive is not None:
            b.v[b._visual] += xp.asarray(eye_drive, dtype=xp.float32)[:, None] * b.eye_gain
        if c.quiescent_nonvisual_sensory:
            # Experimental inactive-modality boundary, not an Up-specific change.
            # Consume the same whole-brain noise draws, but hold unsupported
            # peripheral sensory cells at rest before spike generation.
            b.v[self.isolated] = 0
        for idx, amount in inject:
            b.v[xp.asarray(idx)] += b._amount(amount)
        if c.spike_temperature:
            # Optional escape-noise model. Independent of the background
            # current noise above. The zero-temperature legacy path is exact.
            logits = (b.v[:, 0] - 1) / np.float32(c.spike_temperature)
            probability = 1 / (1 + xp.exp(-xp.clip(logits, -80, 80)))
            probability[self.graded] = 0
            if c.quiescent_nonvisual_sensory:
                probability[self.isolated] = 0
            active = b.rng.random((b.n,)) < probability
            if self.track_score:
                self.last_probability = probability if xp is np else probability.get()
        else:
            active = b.v[:, 0] >= 1
        active[self.graded] = False
        fired = xp.flatnonzero(active)
        b.v[fired] = 0
        self.adaptation[self.kc, 0] += active[self.kc] * np.float32(c.kc_adaptation_increment)
        if self.motor_adaptation is not None:
            self.motor_adaptation[self.adaptive_motor, 0] += active[
                self.adaptive_motor
            ] * np.float32(c.motor_adaptation_increment)
        self.release.fill(0)
        self.release[fired] = 1
        self.release[self.graded] = b.v[self.graded, 0] * np.float32(c.graded_release)
        b.fired, b.steps = fired, b.steps + 1
        return fired if xp is np else fired.get()

    def arrays(self):
        return {
            "adaptation": self.adaptation,
            "release": self.release,
            **(
                {"motor_adaptation": self.motor_adaptation}
                if self.motor_adaptation is not None
                else {}
            ),
            **(
                {"visual_voltage": self.visual_circuit.voltage}
                if self.visual_circuit is not None else {}
            ),
        }

    def restore(self, arrays):
        for name, target in self.arrays().items():
            value = np.asarray(arrays["hybrid_" + name], np.float32)
            if name == "visual_voltage":
                self.visual_circuit.restore_voltage(value)
                self.visual_circuit.steps = round(self.brain.steps * self.brain.dt / 0.004)
                continue
            if value.shape != target.shape or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"Invalid hybrid checkpoint {name}")
            if name == "release" and (value > 1).any():
                raise ValueError("Release must be in [0,1]")
            if name == "motor_adaptation":
                selected = self.adaptive_motor
                selected = selected if self.brain.xp is np else selected.get()
                outside = np.ones(self.brain.n, bool)
                outside[selected] = False
                if np.any(value[outside]):
                    raise ValueError("Motor adaptation outside anatomical target population")
            setattr(self, name, self.brain.xp.asarray(value.copy()))

    def telemetry(self):
        def host(value):
            return value if isinstance(value, np.ndarray) else value.get()

        v = host(self.brain.v[:, 0])
        return {
            "graded_neurons": self.graded_host.tolist(),
            "graded_values": v[self.graded_host].tolist(),
            "graded_mean": float(v[self.graded_host].mean()) if len(self.graded_host) else 0.0,
            "graded_units": "normalized release state [0,1], last neural step; NOT spikes or Hz",
        }
