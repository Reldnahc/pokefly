"""Experimental plasticity on EXISTING KC -> MBON connections.

The preserved centered rule has a global synthetic reward gate. The selective
visual-KC variant uses shared DAN targets as a coarse modulation proxy. Neither
is calibrated dopamine or a resolved compartment model; neither learns adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

RULE_VERSION = "kc-mbon-centered-eligibility-v1"


@dataclass(frozen=True)
class PlasticityConfig:
    learning_rate: float = 0.02
    pre_trace_seconds: float = 0.2
    eligibility_seconds: float = 2.0
    post_baseline_seconds: float = 10.0
    minimum_factor: float = 0.25
    maximum_factor: float = 4.0
    rule: str = "centered-v1"
    activity_reference_hz: float = 5.0
    reward_expectation_seconds: float = 30.0
    normalize_inputs: bool = False
    input_budget_fraction: float = 0.0
    slow_eligibility_seconds: float = 0.0

    def __post_init__(self):
        if self.rule not in (
            "centered-v1",
            "dan-targeted-v1",
            "compartment-ema-v1",
            "sensorimotor-rstdp-v1",
            "sensorimotor-rstdp-v2",
            "sensorimotor-perturb-v1",
            "sensorimotor-perturb-v2",
            "sensorimotor-perturb-v3",
            "sensorimotor-score-v1",
            "sensorimotor-score-v2",
            "sensorimotor-score-v3",
        ):
            raise ValueError("Unknown plasticity rule")
        if not isinstance(self.normalize_inputs, bool):
            raise ValueError("normalize_inputs must be boolean")
        if not all(np.isfinite(v) for k, v in asdict(self).items() if k != "rule"):
            raise ValueError("Plasticity parameters must be finite")
        if not 0 <= self.learning_rate <= 1:
            raise ValueError("learning_rate must be in [0, 1]")
        if min(self.pre_trace_seconds, self.eligibility_seconds, self.post_baseline_seconds) <= 0:
            raise ValueError("Plasticity time constants must be positive")
        if self.activity_reference_hz <= 0:
            raise ValueError("Activity reference must be positive")
        if self.reward_expectation_seconds <= 0:
            raise ValueError("Reward expectation time constant must be positive")
        if self.slow_eligibility_seconds < 0 or (
            self.slow_eligibility_seconds
            and (
                self.slow_eligibility_seconds <= self.eligibility_seconds
                or not self.rule.startswith("sensorimotor-")
            )
        ):
            raise ValueError("Optional slow eligibility requires a longer sensorimotor trace")
        if not 0 <= self.input_budget_fraction < 1:
            raise ValueError("Input budget fraction must be in [0,1)")
        if self.normalize_inputs and self.input_budget_fraction:
            raise ValueError("Choose exact normalization or a bounded input budget")
        if not 0 < self.minimum_factor <= 1 <= self.maximum_factor <= 10:
            raise ValueError("Plasticity bounds must contain the original weight (max <= 10)")


class EligibilityPlasticity:
    def __init__(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        base: np.ndarray,
        n: int,
        config: PlasticityConfig | None = None,
    ):
        self.pre, self.post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
        self.base = np.asarray(base, np.float32).copy()
        if self.pre.shape != self.post.shape or self.pre.shape != self.base.shape:
            raise ValueError("Plastic edge arrays do not align")
        if not len(base) or not np.isfinite(self.base).all() or (self.base == 0).any():
            raise ValueError("Plasticity requires finite, nonzero existing edges")
        if min(self.pre.min(), self.post.min()) < 0 or max(self.pre.max(), self.post.max()) >= n:
            raise ValueError("Plastic neuron index out of range")
        self.config, self.n = config or PlasticityConfig(), n
        self.weights = self.base.copy()
        self.pre_trace = np.zeros(n, np.float32)
        self.post_baseline = np.zeros(n, np.float32)
        self.eligibility = np.zeros(len(base), np.float32)
        self.updates = 0
        self.dopamine = 0.0
        self.last_changed = 0

    def observe(self, fired: np.ndarray, dt: float) -> None:
        spike = np.zeros(self.n, np.float32)
        spike[fired] = 1
        c = self.config
        self.pre_trace *= np.exp(-dt / c.pre_trace_seconds)
        self.pre_trace += spike * (1 - np.exp(-dt / c.pre_trace_seconds))
        # Center postsynaptic activity against its PREVIOUS baseline, not future reward.
        coincidence = self.pre_trace[self.pre] * (spike[self.post] - self.post_baseline[self.post])
        decay = np.exp(-dt / c.eligibility_seconds)
        self.eligibility *= decay
        self.eligibility += (1 - decay) * coincidence
        baseline_decay = np.exp(-dt / c.post_baseline_seconds)
        self.post_baseline *= baseline_decay
        self.post_baseline += (1 - baseline_decay) * spike

    def reinforce(self, reward: float, *, enabled: bool = True) -> dict:
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        self.dopamine = float(np.tanh(reward))
        self.last_changed = 0
        if enabled and self.dopamine != 0 and self.config.learning_rate != 0:
            old = self.weights.copy()
            # Scale by original edge magnitude; signs and sparsity cannot change.
            factor = self.weights / self.base
            factor += self.config.learning_rate * self.dopamine * self.eligibility
            factor = np.clip(factor, self.config.minimum_factor, self.config.maximum_factor)
            self.weights[:] = self.base * factor
            self.last_changed = int(np.count_nonzero(self.weights != old))
            self.updates += 1
        return self.metrics()

    def metrics(self) -> dict:
        delta = self.weights - self.base
        return {
            "rule": RULE_VERSION,
            "plastic_edges": len(self.base),
            "changed_edges": int(np.count_nonzero(delta)),
            "changed_this_reward": self.last_changed,
            "reward_updates": self.updates,
            "l1_weight_change": float(np.abs(delta).sum(dtype=np.float64)),
            "max_relative_change": float(np.max(np.abs(delta / self.base))),
            "eligibility_l1": float(np.abs(self.eligibility).sum(dtype=np.float64)),
            "dopamine_surrogate": self.dopamine,
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            k: getattr(self, k)
            for k in ("pre", "post", "base", "weights", "pre_trace", "post_baseline", "eligibility")
        }

    def restore(self, arrays, metadata: dict) -> None:
        for key in ("pre", "post", "base"):
            if not np.array_equal(arrays[key], getattr(self, key)):
                raise ValueError(f"Checkpoint plastic edge identity mismatch: {key}")
        checked = {}
        for key in ("weights", "pre_trace", "post_baseline", "eligibility"):
            value = np.asarray(arrays[key], np.float32)
            if value.shape != getattr(self, key).shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint {key}")
            checked[key] = value.copy()
        factor = checked["weights"] / self.base
        if (factor < self.config.minimum_factor - 1e-6).any() or (
            factor > self.config.maximum_factor + 1e-6
        ).any():
            raise ValueError("Checkpoint weights exceed sign-preserving bounds")
        for key, value in checked.items():
            setattr(self, key, value)
        self.updates = int(metadata["reward_updates"])
        self.dopamine = float(metadata["dopamine_surrogate"])
        self.last_changed = int(metadata["changed_this_reward"])


class SensorimotorPlasticity(EligibilityPlasticity):
    """Experimental reward-modulated covariance on existing excitatory inputs.

    Anatomical targets are ALL descending and motor neurons, not named buttons.
    Only presynaptic/postsynaptic spikes and scalar reward enter the rule. The
    decoder never supplies an action or gradient. This is a broader engineering
    model, NOT a validated fly dopamine compartment or molecular mechanism.
    Background: Florian 2007, Neural Computation 19:1468-1502.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        signed = self.config.rule == "sensorimotor-score-v3"
        if not signed and (self.base <= 0).any():
            raise ValueError("Sensorimotor rule requires existing excitatory edges")
        # Signed-score variant has separate local excitatory/inhibitory resource
        # budgets. It never cancels opposite signs to evade a total-input bound.
        self.resource_group = self.post * 2 + (self.base < 0) if signed else self.post
        self.resource_size = self.n * 2 if signed else self.n
        self.resource_base = np.abs(self.base) if signed else self.base
        self.input_budget = np.bincount(
            self.resource_group, weights=self.resource_base, minlength=self.resource_size
        )
        self.reset_modulation()

    def reset_modulation(self):
        self.reward_mean = np.array(0.0, np.float64)
        self.feedback_elapsed = np.array(0.0, np.float64)
        self.prediction_error = np.array(0.0, np.float64)
        self.slow_eligibility = (
            np.zeros_like(self.eligibility) if self.config.slow_eligibility_seconds else None
        )

    def _slow_trace_options(self, dt):
        return (
            {
                "slow_eligibility": self.slow_eligibility,
                "slow_decay": np.exp(-dt / self.config.slow_eligibility_seconds),
            }
            if self.slow_eligibility is not None
            else {}
        )

    def observe(self, fired, dt):
        from pokefly.fast_plasticity import sensorimotor_eligibility

        spike = np.zeros(self.n, np.float32)
        spike[fired] = 1
        c = self.config
        decay = np.exp(-dt / c.pre_trace_seconds)
        self.pre_trace *= decay
        centered = c.rule == "sensorimotor-rstdp-v2"
        pre_decay = decay
        if not centered:
            self.pre_trace += (1 - decay) * spike
        incoming = (
            self.pre_trace - np.float32(decay) * self.post_baseline if centered else self.pre_trace
        )
        scale = dt * c.activity_reference_hz
        decay = np.exp(-dt / c.eligibility_seconds)
        sensorimotor_eligibility(
            self.pre,
            self.post,
            incoming,
            self.post_baseline,
            spike,
            self.eligibility,
            scale,
            decay,
            **({"incoming_floor": -5.0} if centered else {}),
            **self._slow_trace_options(dt),
        )
        if centered:
            self.pre_trace += (1 - pre_decay) * spike
        decay = np.exp(-dt / c.post_baseline_seconds)
        self.post_baseline *= decay
        self.post_baseline += (1 - decay) * spike
        self.feedback_elapsed += dt

    def reinforce(self, reward, *, enabled=True):
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        self.dopamine = float(np.tanh(reward))
        self.prediction_error[...] = self.dopamine - self.reward_mean
        self.last_changed = 0
        if enabled:
            if self.config.learning_rate and self.prediction_error != 0:
                old = self.weights.copy()
                factor = self.weights / self.base
                eligibility = self.factor_eligibility()
                factor += self.config.learning_rate * float(self.prediction_error) * eligibility
                np.clip(factor, self.config.minimum_factor, self.config.maximum_factor, out=factor)
                if self.config.normalize_inputs or self.config.input_budget_fraction:
                    # Local synaptic-resource competition for every anatomical
                    # target neuron, not an action-frequency quota. Bounds hold
                    # even when the iterative budget projection is approximate.
                    for _ in range(12):
                        total = np.bincount(
                            self.resource_group,
                            weights=self.resource_base * factor,
                            minlength=self.resource_size,
                        )
                        fraction = self.config.input_budget_fraction
                        target = np.clip(
                            total[self.resource_group],
                            self.input_budget[self.resource_group] * (1 - fraction),
                            self.input_budget[self.resource_group] * (1 + fraction),
                        )
                        ratio = total[self.resource_group] / target
                        if np.max(np.abs(ratio - 1)) < 1e-5:
                            break
                        factor /= ratio
                        np.clip(
                            factor,
                            self.config.minimum_factor,
                            self.config.maximum_factor,
                            out=factor,
                        )
                self.weights[:] = self.base * factor
                self.last_changed = int(np.count_nonzero(old != self.weights))
                self.updates += 1
            decay = np.exp(-float(self.feedback_elapsed) / self.config.reward_expectation_seconds)
            self.reward_mean *= decay
            self.reward_mean += (1 - decay) * self.dopamine
        self.feedback_elapsed[...] = 0
        return self.metrics()

    def factor_eligibility(self):
        return (
            0.5 * (self.eligibility + self.slow_eligibility)
            if self.slow_eligibility is not None
            else self.eligibility
        )

    def metrics(self):
        return {
            **super().metrics(),
            "rule": (
                "sensorimotor-pre-post-covariance-v2"
                if self.config.rule == "sensorimotor-rstdp-v2"
                else "sensorimotor-reward-covariance-v1"
            ),
            "modulation": "experimental internal motor plasticity; no decoder feedback",
            "reward_mean": float(self.reward_mean),
            "internal_reward_error": float(self.prediction_error),
            **(
                {
                    "slow_eligibility_l1": float(
                        np.abs(self.slow_eligibility).sum(dtype=np.float64)
                    ),
                    "eligibility_mix": "equal fast/slow local traces",
                }
                if self.slow_eligibility is not None
                else {}
            ),
        }

    def arrays(self):
        return {
            **super().arrays(),
            "reward_mean": self.reward_mean,
            "prediction_error": self.prediction_error,
            "feedback_elapsed": self.feedback_elapsed,
            **(
                {"slow_eligibility": self.slow_eligibility}
                if self.slow_eligibility is not None
                else {}
            ),
        }

    def restore(self, arrays, metadata):
        checked = {}
        if self.slow_eligibility is not None:
            slow = np.asarray(arrays["slow_eligibility"], np.float32)
            if slow.shape != self.eligibility.shape or not np.isfinite(slow).all():
                raise ValueError("Invalid slow eligibility checkpoint")
            checked["slow_eligibility"] = slow.copy()
        for key in ("reward_mean", "prediction_error", "feedback_elapsed"):
            value = np.asarray(arrays[key], np.float64)
            if value.shape != () or not np.isfinite(value):
                raise ValueError(f"Invalid sensorimotor checkpoint {key}")
            checked[key] = value.copy()
        if abs(checked["reward_mean"]) > 1 or abs(checked["prediction_error"]) > 2:
            raise ValueError("Invalid sensorimotor modulation bounds")
        if checked["feedback_elapsed"] < 0:
            raise ValueError("Invalid sensorimotor elapsed time")
        super().restore(arrays, metadata)
        for key, value in checked.items():
            setattr(self, key, value)


class NeuralPerturbationPlasticity(SensorimotorPlasticity):
    """Experimental local credit from the simulator's existing neural noise.

    Uses an actual neuron's independent current perturbation, not its action,
    as an innovation signal. No added perturbations, weights or external critic.
    Inspired by Fiete & Seung (2006), DOI:10.1103/PhysRevLett.97.048104;
    our finite Bernoulli-current/filtered-trace version is an approximation,
    NOT their conductance model or a validated fly molecular mechanism.
    """

    def observe(self, fired, dt, *, perturbation=None, probability=None):
        from pokefly.fast_plasticity import sensorimotor_eligibility

        if perturbation is None or probability is None or not 0 <= probability <= 1:
            raise ValueError("Actual neural perturbations and their probability are required")
        noise = np.asarray(perturbation)
        if noise.shape != (self.n,) or (
            noise.dtype != np.bool_ and not np.isin(noise, [0, 1]).all()
        ):
            raise ValueError("Invalid neural perturbation mask")
        noise = noise.astype(np.float32, copy=False)
        spike = np.zeros(self.n, np.float32)
        spike[fired] = 1
        c = self.config
        # Presynaptic history before this step's perturbation: no future spikes.
        decay = np.exp(-dt / c.pre_trace_seconds)
        self.pre_trace *= decay
        centered = c.rule == "sensorimotor-perturb-v3"
        incoming = (
            self.pre_trace - np.float32(decay) * self.post_baseline if centered else self.pre_trace
        )
        sensorimotor_eligibility(
            self.pre,
            self.post,
            incoming,
            np.full(self.n, probability, np.float32),
            noise,
            self.eligibility,
            dt * c.activity_reference_hz,
            np.exp(-dt / c.eligibility_seconds),
            # Clipping rare positive innovations biases the zero-mean trace.
            # Keep v1 reproducible; v2 uses a linear eligibility filter.
            clip_eligibility=c.rule == "sensorimotor-perturb-v1",
            incoming_floor=-5.0 if centered else 0.0,
            **self._slow_trace_options(dt),
        )
        self.pre_trace += (1 - decay) * spike
        decay = np.exp(-dt / c.post_baseline_seconds)
        self.post_baseline *= decay
        self.post_baseline += (1 - decay) * spike
        self.feedback_elapsed += dt

    def metrics(self):
        return {
            **super().metrics(),
            "rule": (
                "sensorimotor-centered-input-perturbation-v3"
                if self.config.rule == "sensorimotor-perturb-v3"
                else "sensorimotor-linear-perturbation-v2"
                if self.config.rule == "sensorimotor-perturb-v2"
                else "sensorimotor-neural-perturbation-v1"
            ),
            "modulation": "local existing neural-noise credit; no decoder or external critic",
        }


class DANTargetedPlasticity(EligibilityPlasticity):
    """Visual-KC rule with connection-derived, edge-specific modulatory channels.

    PAM=positive and PPL1=negative are COARSE experimental assignments, not a
    measured valence for every DAN. Shared DAN->KC and DAN->MBON connections
    approximate target overlap, NOT anatomical synapse-compartment annotations.
    Reward drives a synthetic release level through those channels. No claim is
    made to simulate measured dopamine concentrations or DAN receptor kinetics.
    """

    def __init__(self, *args, positive_gate, negative_gate, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_gate = self._check_gate(positive_gate)
        self.negative_gate = self._check_gate(negative_gate)

    def _check_gate(self, value):
        gate = np.asarray(value, np.float32).copy()
        if (
            gate.shape != self.base.shape
            or not np.isfinite(gate).all()
            or ((gate < 0) | (gate > 1)).any()
        ):
            raise ValueError("DAN gate must align with edges and be in [0,1]")
        return gate

    def observe(self, fired, dt):
        spike = np.zeros(self.n, np.float32)
        spike[fired] = 1
        decay = np.exp(-dt / self.config.pre_trace_seconds)
        self.pre_trace *= decay
        self.pre_trace += (1 - decay) * spike
        # KC activity tags existing output synapses for a later reward pulse.
        decay = np.exp(-dt / self.config.eligibility_seconds)
        self.eligibility *= decay
        self.eligibility += (1 - decay) * np.clip(
            self.pre_trace[self.pre] / (dt * self.config.activity_reference_hz), 0, 1
        )

    def reinforce(self, reward, *, enabled=True):
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        self.dopamine = float(np.tanh(reward))
        self.last_changed = 0
        if enabled and self.dopamine != 0 and self.config.learning_rate != 0:
            old = self.weights.copy()
            gate = self.positive_gate if reward > 0 else self.negative_gate
            factor = self.weights / self.base
            # Active-input depression; inactive-input recovery toward the base.
            # This is a bounded, simplified depression/recovery hypothesis, not
            # a reproduction of the full published incentive-circuit model.
            factor -= (
                self.config.learning_rate
                * abs(self.dopamine)
                * gate
                * (self.eligibility + factor - 1)
            )
            factor = np.clip(factor, self.config.minimum_factor, self.config.maximum_factor)
            self.weights[:] = self.base * factor
            self.last_changed = int(np.count_nonzero(old != self.weights))
            self.updates += 1
        return self.metrics()

    def metrics(self):
        return {
            **super().metrics(),
            "rule": "visual-kc-dan-overlap-depression-recovery-v1",
            "positive_gated_edges": int(np.count_nonzero(self.positive_gate)),
            "negative_gated_edges": int(np.count_nonzero(self.negative_gate)),
            "modulation": "synthetic PAM/PPL1 target-overlap proxy, not resolved compartments",
        }

    def arrays(self):
        return {
            **super().arrays(),
            "positive_gate": self.positive_gate,
            "negative_gate": self.negative_gate,
        }

    def restore(self, arrays, metadata):
        for key in ("positive_gate", "negative_gate"):
            if not np.array_equal(arrays[key], getattr(self, key)):
                raise ValueError("Checkpoint DAN targeting mismatch")
        super().restore(arrays, metadata)


def dan_target_gates(brain, pre, post):
    """Shared presynaptic DANs, using existing signed-normalized graph magnitudes.

    No new fast synapses, manual MBON valences, or fabricated compartment IDs.
    Max-normalization is over the whole selected edge set, preserving specificity.
    """
    gates = []
    kinds = brain.cell_type.astype(str)
    for prefix in ("PAM", "PPL1"):
        gate = np.zeros(len(pre), np.float64)
        for dan in np.flatnonzero(np.char.startswith(kinds, prefix)):
            indices = slice(brain.indptr[dan], brain.indptr[dan + 1])
            targets = np.zeros(brain.n, np.float32)
            targets[brain.indices[indices]] = np.abs(brain.weights[indices])
            gate += np.sqrt(targets[pre] * targets[post])
        if gate.max(initial=0) > 0:
            gate /= gate.max()
        gates.append(gate.astype(np.float32))
    if not np.any(gates[0]):
        raise ValueError("No shared PAM targets for the selected visual-KC edges")
    return gates


class CompartmentPlasticity(DANTargetedPlasticity):
    """Experimental expectation-centered modulation on a partial compartment map.

    Reward expectation is an internal DAN adaptation state (a recency average),
    NOT an external critic, action-value table, or fitted decoder. Positive
    prediction error depresses active KC inputs in that compartment; negative
    error potentiates them. Compare Rajagopalan et al. 2023, PNAS e2221415120.
    This is not their full model or a validated biological RPE circuit.
    """

    def __init__(self, *args, gates, **kwargs):
        gates = np.asarray(gates, np.float32)
        if gates.ndim != 2 or gates.shape[0] != 3:
            raise ValueError("Three explicit compartment gates required")
        super().__init__(
            *args,
            positive_gate=np.maximum(gates[0], gates[1]),
            negative_gate=gates[2],
            **kwargs,
        )
        self.compartment_gates = np.stack([self._check_gate(g) for g in gates])
        self.reset_modulation()

    def reset_modulation(self):
        self.reward_expectation = np.zeros(3, np.float64)
        self.prediction_error = np.zeros(3, np.float64)
        self.feedback_elapsed = np.array(0.0, np.float64)

    def observe(self, fired, dt):
        super().observe(fired, dt)
        self.feedback_elapsed += dt

    def reinforce(self, reward, *, enabled=True):
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        self.dopamine = float(np.tanh(reward))
        release = np.array([max(self.dopamine, 0)] * 2 + [max(-self.dopamine, 0)])
        self.prediction_error[:] = release - self.reward_expectation
        self.last_changed = 0
        if enabled:
            signal = self.prediction_error @ self.compartment_gates
            if self.config.learning_rate and np.any(signal):
                old = self.weights.copy()
                factor = self.weights / self.base
                factor -= self.config.learning_rate * signal * self.eligibility
                factor = np.clip(factor, self.config.minimum_factor, self.config.maximum_factor)
                self.weights[:] = self.base * factor
                self.last_changed = int(np.count_nonzero(old != self.weights))
                self.updates += 1
            decay = np.exp(-float(self.feedback_elapsed) / self.config.reward_expectation_seconds)
            self.reward_expectation *= decay
            self.reward_expectation += (1 - decay) * release
        self.feedback_elapsed[...] = 0
        return self.metrics()

    def metrics(self):
        return {
            **super().metrics(),
            "rule": "partial-compartment-expectation-centered-v1",
            "modulation": (
                "partial type-level map; internal reward-average proxy, not resolved DAN kinetics"
            ),
            "compartment_gated_edges": np.count_nonzero(self.compartment_gates, axis=1).tolist(),
            "reward_expectation": self.reward_expectation.tolist(),
            "prediction_error": self.prediction_error.tolist(),
        }

    def arrays(self):
        return {
            **super().arrays(),
            **{
                k: getattr(self, k)
                for k in (
                    "compartment_gates",
                    "reward_expectation",
                    "prediction_error",
                    "feedback_elapsed",
                )
            },
        }

    def restore(self, arrays, metadata):
        if not np.array_equal(arrays["compartment_gates"], self.compartment_gates):
            raise ValueError("Checkpoint compartment identity mismatch")
        checked = {}
        for key in ("reward_expectation", "prediction_error", "feedback_elapsed"):
            value = np.asarray(arrays[key], np.float64)
            if value.shape != getattr(self, key).shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint {key}")
            checked[key] = value.copy()
        if (checked["reward_expectation"] < 0).any() or (checked["reward_expectation"] > 1).any():
            raise ValueError("Invalid checkpoint reward expectation")
        if checked["feedback_elapsed"] < 0 or (np.abs(checked["prediction_error"]) > 1).any():
            raise ValueError("Invalid checkpoint feedback state")
        super().restore(arrays, metadata)
        for key, value in checked.items():
            setattr(self, key, value)
