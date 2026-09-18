"""Optional local likelihood-score credit for stochastic connectome neurons.

Inspired by Florian (2007), doi:10.1162/neco.2007.19.6.1468. This implementation
uses discrete logistic escape noise and hard reset, NOT measured fly physiology
or the paper's exact neuron model. Existing motor/descending inputs only:
excitatory in v1/v2, both original signs in v3. No learned readout, action
identity, cue label, or RAM input.

The optional centered-v4 variant uses deviations from the neuron's prior
release average. This is a local covariance/innovation approximation, NOT the
exact likelihood gradient of the original uncentered physical neuron model.

Projected-v5 retains the original score, but removes the update component
that changes estimated mean input to each postsynaptic cell. Its constraint
uses only local release averages, not cue/action labels or desired rates.
Bounds and resource projection can subsequently make the constraint inexact.
"""

import numpy as np

from pokefly.fast_plasticity import likelihood_eligibility, mean_input_projection
from pokefly.plasticity import SensorimotorPlasticity


class LikelihoodPlasticity(SensorimotorPlasticity):
    def reset_modulation(self):
        super().reset_modulation()
        self.membrane_trace = np.zeros_like(self.eligibility)
        self.release_baseline = (
            np.zeros(self.n, np.float32)
            if self.config.rule in (
                "sensorimotor-score-centered-v4", "sensorimotor-score-projected-v5"
            )
            else None
        )
        if self.config.slow_eligibility_seconds:
            raise ValueError("Likelihood rule currently supports one eligibility trace")

    def observe(self, fired, dt, *, release, probability, membrane_decay, gain, temperature):
        release, probability = np.asarray(release), np.asarray(probability)
        if (
            release.shape != (self.n,)
            or probability.shape != (self.n,)
            or not np.isfinite(release).all()
            or not np.isfinite(probability).all()
            or ((release < 0) | (release > 1)).any()
            or ((probability < 0) | (probability > 1)).any()
            or not 0 < temperature <= 0.5
            or not 0 <= membrane_decay < 1
        ):
            raise ValueError("Invalid stochastic neural state for likelihood eligibility")
        spike = np.zeros(self.n, np.float32)
        spike[fired] = 1
        likelihood_eligibility(
            self.pre,
            self.post,
            self.base,
            release,
            spike,
            probability,
            self.membrane_trace,
            self.eligibility,
            membrane_decay,
            np.exp(-dt / self.config.eligibility_seconds),
            gain,
            temperature,
            **(
                {"release_baseline": self.release_baseline}
                if self.config.rule == "sensorimotor-score-centered-v4"
                else {}
            ),
        )
        if self.release_baseline is not None:
            decay = np.exp(-dt / self.config.post_baseline_seconds)
            self.release_baseline *= decay
            self.release_baseline += (1 - decay) * release
        self.feedback_elapsed += dt

    def metrics(self):
        return {
            **super().metrics(),
            "rule": (
                "sensorimotor-mean-input-projected-score-v5"
                if self.config.rule == "sensorimotor-score-projected-v5"
                else
                "sensorimotor-centered-spike-innovation-v4"
                if self.config.rule == "sensorimotor-score-centered-v4"
                else "sensorimotor-signed-likelihood-score-v3"
                if self.config.rule == "sensorimotor-score-v3"
                else "sensorimotor-preconditioned-likelihood-score-v2"
                if self.config.rule == "sensorimotor-score-v2"
                else "sensorimotor-logistic-likelihood-score-v1"
            ),
            "modulation": (
                "local score projected against running mean input; no action/cue labels"
                if self.config.rule == "sensorimotor-score-projected-v5"
                else
                "local pre-centered spike innovation (approximate, not exact score gradient)"
                if self.config.rule == "sensorimotor-score-centered-v4"
                else "local stochastic-spike score; no decoder feedback or external critic"
            ),
        }

    def factor_eligibility(self):
        # Positive diagonal preconditioning, using original anatomy only.
        # v1's factor gradient makes absolute updates proportional to base^2;
        # v2 removes one base factor so weak existing inputs can participate.
        value = super().factor_eligibility()
        if self.config.rule == "sensorimotor-score-projected-v5":
            # D = diag(1/base) is the same positive preconditioner as v2.
            # With a = base * E[release], the constrained ascent direction is
            # Dg - Da * (a.T Dg)/(a.T Da), separately for each target cell.
            # Thus a.T direction = 0 before bounds/resource competition.
            # This suppresses generic input drift, not a button or spike quota.
            return mean_input_projection(
                value / self.base, self.base, self.release_baseline[self.pre], self.post, self.n
            )
        return (
            value / np.abs(self.base)
            if self.config.rule
            in ("sensorimotor-score-v2", "sensorimotor-score-v3", "sensorimotor-score-centered-v4")
            else value
        )

    def arrays(self):
        return {
            **super().arrays(),
            "membrane_trace": self.membrane_trace,
            **(
                {"release_baseline": self.release_baseline}
                if self.release_baseline is not None
                else {}
            ),
        }

    def restore(self, arrays, metadata):
        trace = np.asarray(arrays["membrane_trace"], np.float32)
        if (
            trace.shape != self.membrane_trace.shape
            or not np.isfinite(trace).all()
            or (
                self.config.rule != "sensorimotor-score-centered-v4" and (trace < 0).any()
            )
        ):
            raise ValueError("Invalid membrane eligibility checkpoint")
        baseline = None
        if self.release_baseline is not None:
            baseline = np.asarray(arrays["release_baseline"], np.float32)
            if (
                baseline.shape != (self.n,)
                or not np.isfinite(baseline).all()
                or ((baseline < 0) | (baseline > 1)).any()
            ):
                raise ValueError("Invalid presynaptic release baseline")
        super().restore(arrays, metadata)
        self.membrane_trace = trace.copy()
        if baseline is not None:
            self.release_baseline = baseline.copy()
