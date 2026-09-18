"""Pixel-driven fly with fixed motor outputs and experimental internal learning."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pokefly.compartments import compartment_gates
from pokefly.dynamics import DynamicsConfig, HybridDynamics
from pokefly.motors import MOTOR_GROUPS, MotorConfig, MotorDecoder, populations
from pokefly.pixel_brain import PixelBrain, PixelObservation
from pokefly.plastic_edges import csr_offsets_for_edges, sensorimotor_targets
from pokefly.plasticity import (
    CompartmentPlasticity,
    DANTargetedPlasticity,
    EligibilityPlasticity,
    NeuralPerturbationPlasticity,
    PlasticityConfig,
    SensorimotorPlasticity,
    dan_target_gates,
)
from pokefly.runtime import configure_runtime
from pokefly.score_plasticity import LikelihoodPlasticity


@dataclass(frozen=True)
class BrainConfig:
    brain_steps: int = 12
    noise_hz: float = 1.2
    noise_amplitude: float = 0.22
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    intrinsic_calibration: str | None = None

    def __post_init__(self):
        if not 1 <= self.brain_steps <= 1000:
            raise ValueError("brain_steps must be in [1,1000]")
        if not np.isfinite([self.noise_hz, self.noise_amplitude]).all():
            raise ValueError("Noise settings must be finite")
        if not 0 <= self.noise_hz <= 50 or not 0 <= self.noise_amplitude <= 2:
            raise ValueError("Noise rate must be 0..50 Hz and amplitude 0..2")
        if self.intrinsic_calibration is not None:
            if not isinstance(self.intrinsic_calibration, str) or not self.intrinsic_calibration:
                raise ValueError("Intrinsic calibration must name an explicit file")
            if self.dynamics.profile != "hybrid-v1":
                raise ValueError("Intrinsic calibration requires hybrid dynamics")
        if (
            self.plasticity.rule.startswith("sensorimotor-perturb-")
            and self.dynamics.profile != "hybrid-v1"
        ):
            raise ValueError("Neural-perturbation credit requires hybrid dynamics")
        if self.plasticity.rule.startswith("sensorimotor-score-") and (
            self.dynamics.spike_temperature <= 0 or self.plasticity.slow_eligibility_seconds
        ):
            raise ValueError("Likelihood credit requires stochastic spiking and one trace")


class CheckpointNoise:
    """CPU PCG64 stream converted to the backend; neural noise only, never buttons.

    Replaces CuPy's hardware-sized private RNG buffer with a portable documented
    RNG state. Same Bernoulli distribution as upstream; not the same noise draws.
    """

    def __init__(self, xp, seed: int):
        self.xp, self.generator = xp, np.random.default_rng(seed)

    def random(self, shape):
        return self.xp.asarray(self.generator.random(shape))


class InternalBrain(PixelBrain):
    def __init__(self, *, device="auto", seed=64, config: BrainConfig | None = None):
        config = config or BrainConfig()
        super().__init__(device=device, seed=seed, brain_steps=config.brain_steps)
        self.config = config
        self.seed = seed
        self.brain.noise_hz, self.brain.noise_amp = config.noise_hz, config.noise_amplitude
        self.brain.rng = CheckpointNoise(self.brain.xp, seed)
        with np.load(configure_runtime() / "brain.npz", allow_pickle=False) as meta:
            self.body_ids = meta["ids"].copy()
        self.motors = populations(self.brain, self.body_ids)
        self.groups.update({MOTOR_GROUPS[k]: v for k, v in self.motors.items()})
        self.groups["visual_Kenyon_candidates"] = self.brain.cells(["KCg-d", "KCab-p"])
        self.decoder = MotorDecoder(self.motors, config.motor)
        b = self.brain
        is_mbon = np.zeros(b.n, bool)
        is_mbon[self.groups["MBONs"]] = True
        pre, offsets = [], []
        plastic_kcs = self.groups[
            "visual_Kenyon_candidates"
            if config.plasticity.rule != "centered-v1"
            else "Kenyon_cells"
        ]
        for cell in plastic_kcs:
            candidates = np.arange(b.indptr[cell], b.indptr[cell + 1])
            selected = candidates[is_mbon[b.indices[candidates]]]
            pre.extend([int(cell)] * len(selected))
            offsets.extend(selected)
        self.csc_offsets = np.asarray(offsets, np.int64)
        sensorimotor = config.plasticity.rule.startswith("sensorimotor-")
        if sensorimotor:
            # Anatomy only, independent of the selected motor-to-button registry.
            eligible_post = sensorimotor_targets(b, config.plasticity.scope)
            eligible_sign = (
                b.weights != 0
                if config.plasticity.rule == "sensorimotor-score-v3"
                else b.weights > 0
            )
            self.csc_offsets = np.flatnonzero(eligible_post[b.indices] & eligible_sign)
            pre = np.searchsorted(b.indptr, self.csc_offsets, side="right") - 1
        post = b.indices[self.csc_offsets]
        args = (np.asarray(pre), post, b.weights[self.csc_offsets], b.n, config.plasticity)
        if sensorimotor:
            rule = (
                LikelihoodPlasticity
                if config.plasticity.rule.startswith("sensorimotor-score-")
                else NeuralPerturbationPlasticity
                if config.plasticity.rule.startswith("sensorimotor-perturb-")
                else SensorimotorPlasticity
            )
            self.plasticity = rule(*args)
        elif config.plasticity.rule == "compartment-ema-v1":
            self.plasticity = CompartmentPlasticity(
                *args, gates=compartment_gates(b, np.asarray(pre), post)
            )
        elif config.plasticity.rule == "dan-targeted-v1":
            positive, negative = dan_target_gates(b, np.asarray(pre), post)
            self.plasticity = DANTargetedPlasticity(
                *args, positive_gate=positive, negative_gate=negative
            )
        else:
            self.plasticity = EligibilityPlasticity(*args)
        self.csr_offsets = None
        if b.device == "cuda":
            from pokefly.deterministic import DeterministicCUDAInput

            b.synaptic_input = DeterministicCUDAInput(b)
            csr_offsets = csr_offsets_for_edges(
                self.plasticity.pre, post, b._W.indptr.get(), b._W.indices.get()
            )
            self.csr_offsets = b.xp.asarray(csr_offsets)
            if not np.array_equal(b._W.data[self.csr_offsets].get(), self.plasticity.base):
                raise RuntimeError("CPU/CUDA initial plastic weights disagree")
        self.hybrid = None
        if config.dynamics.profile == "hybrid-v1":
            self.hybrid = HybridDynamics(b, self.groups, config.dynamics)
            self.hybrid.track_noise = isinstance(self.plasticity, NeuralPerturbationPlasticity)
            self.hybrid.track_score = isinstance(self.plasticity, LikelihoodPlasticity)
            b.step = self.hybrid.step
        self.intrinsic_calibration_info = None
        if config.intrinsic_calibration:
            path = Path(config.intrinsic_calibration)
            with np.load(path, allow_pickle=False) as calibration:
                if not np.array_equal(calibration["body_ids"], self.body_ids):
                    raise ValueError("Intrinsic calibration neuron identity mismatch")
                bias = np.asarray(calibration["bias"], np.float32)
                allowed = np.char.find(b.superclass.astype(str), "sensory") < 0
                allowed[self.hybrid.graded_host] = False
                if (
                    bias.shape != (b.n, 1)
                    or not np.isfinite(bias).all()
                    or (bias < -0.140001).any()
                    or (bias > 0.200001).any()
                    or np.any(bias[~allowed])
                ):
                    raise ValueError("Invalid intrinsic calibration offsets")
                self.hybrid.intrinsic_bias = b.xp.asarray(bias.copy())
            self.intrinsic_calibration_info = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rule": "uniform-neutral-rate-homeostasis-v1",
                "scope": "frozen internal excitability offsets; no game/reward/motor labels",
            }
        self.sensory_isolation = {"enabled": False, "neurons": 0}
        if self.hybrid and len(self.hybrid.isolated_host):
            isolated = self.hybrid.isolated_host
            self.groups["isolated_nonvisual_sensory"] = isolated
            self.sensory_isolation = {
                "enabled": True,
                "mode": "nonvisual-incoming-v1",
                "neurons": len(isolated),
                "body_ids_sha256": hashlib.sha256(
                    self.body_ids[isolated].astype("<i8").tobytes()
                ).hexdigest(),
                "selection": "superclass contains sensory; exclude brain.visual",
                "effect": "zero incoming network current; outgoing, tonic and noise retained",
            }
            if config.dynamics.quiescent_nonvisual_sensory:
                self.sensory_isolation.update(
                    mode="nonvisual-quiescent-v1",
                    effect="incoming current isolated; unsupported sensory voltages held at rest",
                )

    def observe(self, frame: np.ndarray) -> PixelObservation:
        drive = self.retina.encode(frame)
        counts = np.zeros(self.brain.n, np.int32)
        for _ in range(self.brain_steps):
            fired = self.brain.step(eye_drive=drive)
            counts[fired] += 1
            self._observe_plasticity(fired)
        return self.observation(counts, drive)

    def integrate_frame(self, frame: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """One neural update from one raw image; used by temporal-vision trials."""
        drive = self.retina.encode(frame)
        fired = self.brain.step(eye_drive=drive)
        counts[fired] += 1
        self._observe_plasticity(fired)
        return drive

    def _observe_plasticity(self, fired):
        if isinstance(self.plasticity, LikelihoodPlasticity):
            self.plasticity.observe(
                fired,
                self.brain.dt,
                release=self.hybrid.previous_release,
                probability=self.hybrid.last_probability,
                membrane_decay=float(self.brain.decay),
                gain=self.brain.gain,
                temperature=self.config.dynamics.spike_temperature,
            )
        elif isinstance(self.plasticity, NeuralPerturbationPlasticity):
            self.plasticity.observe(
                fired,
                self.brain.dt,
                perturbation=self.hybrid.last_noise,
                probability=self.brain.noise_hz * self.brain.dt,
            )
        else:
            self.plasticity.observe(fired, self.brain.dt)

    def observation(self, counts: np.ndarray, drive: np.ndarray) -> PixelObservation:
        return PixelObservation(
            drive,
            counts,
            {k: int(counts[v].sum()) for k, v in self.groups.items()},
            self.brain_steps,
            self.hybrid.telemetry() if self.hybrid else None,
        )

    def choose(self, observation: PixelObservation) -> tuple[str, dict[str, float]]:
        return self.decoder.choose(observation.counts, observation.steps * self.brain.dt)

    def reinforce(self, reward: float, *, enabled=True) -> dict:
        result = self.plasticity.reinforce(reward, enabled=enabled)
        if result["changed_this_reward"]:
            self.sync_weights()
        return result

    def sync_weights(self) -> None:
        self.brain.weights[self.csc_offsets] = self.plasticity.weights
        if self.csr_offsets is not None:
            self.brain._W.data[self.csr_offsets] = self.brain.xp.asarray(self.plasticity.weights)

    def identity(self) -> dict:
        return {
            "numerics": "pcg64-fixed-cuda-row-reduction-v1",
            "config": asdict(self.config),
            "retina": self.retina.metadata,
            "motor_body_ids": {k: self.body_ids[v].tolist() for k, v in self.motors.items()},
            "device": self.brain.device,
            "n": self.brain.n,
            **(
                {"intrinsic_calibration": self.intrinsic_calibration_info}
                if self.intrinsic_calibration_info
                else {}
            ),
            **(
                {"sensory_isolation": self.sensory_isolation}
                if self.sensory_isolation["enabled"]
                else {}
            ),
        }

    def reset_dynamics(self, seed: int) -> None:
        """New trial: retain weights, reset noise, eligibility, and motor traces."""
        self.brain.reset(seed)
        self.brain.rng = CheckpointNoise(self.brain.xp, seed)
        if self.hybrid:
            self.hybrid.reset()
        self.decoder = MotorDecoder(self.motors, self.config.motor)
        self.plasticity.pre_trace.fill(0)
        self.plasticity.post_baseline.fill(0)
        self.plasticity.eligibility.fill(0)
        self.plasticity.dopamine = 0.0
        self.plasticity.last_changed = 0
        if isinstance(self.plasticity, (CompartmentPlasticity, SensorimotorPlasticity)):
            self.plasticity.reset_modulation()

    def snapshot(self) -> tuple[dict, dict]:
        def host(value):
            return value.copy() if isinstance(value, np.ndarray) else value.get()

        arrays = {
            **{key: value.copy() for key, value in self.plasticity.arrays().items()},
            "voltage": host(self.brain.v),
            "fired": host(self.brain.fired),
            **(
                {"hybrid_" + k: host(v) for k, v in self.hybrid.arrays().items()}
                if self.hybrid
                else {}
            ),
        }
        state = {
            "identity": self.identity(),
            "steps": self.brain.steps,
            "rng": self.brain.rng.generator.bit_generator.state,
            "decoder": self.decoder.state(),
            "plasticity": self.plasticity.metrics(),
        }
        return arrays, state

    def restore(self, arrays, state: dict, *, weights_only=False) -> None:
        identity = self.identity()
        stored = dict(state["identity"])
        # Old baseline checkpoints predate the explicit, inactive dynamics settings.
        stored["config"] = dict(stored["config"])
        stored["config"].setdefault("intrinsic_calibration", None)
        stored["config"].setdefault("dynamics", asdict(DynamicsConfig()))
        stored["config"]["dynamics"] = dict(stored["config"]["dynamics"])
        stored["config"]["dynamics"].setdefault("isolate_nonvisual_sensory", False)
        stored["config"]["dynamics"].setdefault("quiescent_nonvisual_sensory", False)
        stored["config"]["dynamics"].setdefault("spike_temperature", 0.0)
        stored["config"]["dynamics"].setdefault("motor_adaptation_increment", 0.0)
        stored["config"]["dynamics"].setdefault("motor_adaptation_seconds", 3.0)
        stored["config"]["motor"] = dict(stored["config"]["motor"])
        stored["config"]["motor"].setdefault("arbitration", "exclusive-v1")
        stored["config"]["motor"].setdefault("direction_trace_seconds", 1.0)
        stored["config"]["plasticity"] = dict(stored["config"]["plasticity"])
        stored["config"]["plasticity"].setdefault("rule", "centered-v1")
        stored["config"]["plasticity"].setdefault("activity_reference_hz", 5.0)
        stored["config"]["plasticity"].setdefault("reward_expectation_seconds", 30.0)
        stored["config"]["plasticity"].setdefault("normalize_inputs", False)
        stored["config"]["plasticity"].setdefault("input_budget_fraction", 0.0)
        stored["config"]["plasticity"].setdefault("slow_eligibility_seconds", 0.0)
        stored["config"]["plasticity"].setdefault("trace_mixing", "mean-v1")
        stored["config"]["plasticity"].setdefault("scope", "motor-inputs-v1")
        if weights_only:
            # New seed/backend allowed for retention evaluation; circuit settings stay fixed.
            stored["device"] = identity["device"]
        if stored != identity:
            raise ValueError("Checkpoint neural configuration/data/backend mismatch")
        self.plasticity.restore(arrays, state["plasticity"])
        if not weights_only:
            v, fired = np.asarray(arrays["voltage"]), np.asarray(arrays["fired"])
            if v.shape != self.brain.v.shape or not np.isfinite(v).all():
                raise ValueError("Invalid checkpoint voltage")
            if fired.ndim != 1 or (fired < 0).any() or (fired >= self.brain.n).any():
                raise ValueError("Invalid checkpoint spikes")
            self.brain.v = self.brain.xp.asarray(v.copy())
            self.brain.fired = self.brain.xp.asarray(fired.copy())
            self.brain.steps = int(state["steps"])
            self.brain.rng.generator.bit_generator.state = state["rng"]
            self.decoder.restore(state["decoder"])
            if self.hybrid:
                self.hybrid.restore(arrays)
        else:
            self.reset_dynamics(self.seed)
        self.sync_weights()
