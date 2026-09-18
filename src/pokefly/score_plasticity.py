"""Optional local likelihood-score credit for stochastic connectome neurons.

Inspired by Florian (2007), doi:10.1162/neco.2007.19.6.1468. This implementation
uses discrete logistic escape noise and hard reset, NOT measured fly physiology
or the paper's exact neuron model. Existing motor/descending inputs only:
excitatory in v1/v2, both original signs in v3. No learned readout, action
identity, cue label, or RAM input.
"""

import numpy as np

from pokefly.fast_plasticity import likelihood_eligibility
from pokefly.plasticity import SensorimotorPlasticity


class LikelihoodPlasticity(SensorimotorPlasticity):
    def reset_modulation(self):
        super().reset_modulation()
        self.membrane_trace = np.zeros_like(self.eligibility)
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
        )
        self.feedback_elapsed += dt

    def metrics(self):
        return {
            **super().metrics(),
            "rule": (
                "sensorimotor-signed-likelihood-score-v3"
                if self.config.rule == "sensorimotor-score-v3"
                else "sensorimotor-preconditioned-likelihood-score-v2"
                if self.config.rule == "sensorimotor-score-v2"
                else "sensorimotor-logistic-likelihood-score-v1"
            ),
            "modulation": "local stochastic-spike score; no decoder feedback or external critic",
        }

    def factor_eligibility(self):
        # Positive diagonal preconditioning, using original anatomy only.
        # v1's factor gradient makes absolute updates proportional to base^2;
        # v2 removes one base factor so weak existing inputs can participate.
        value = super().factor_eligibility()
        return (
            value / np.abs(self.base)
            if self.config.rule in ("sensorimotor-score-v2", "sensorimotor-score-v3")
            else value
        )

    def arrays(self):
        return {**super().arrays(), "membrane_trace": self.membrane_trace}

    def restore(self, arrays, metadata):
        trace = np.asarray(arrays["membrane_trace"], np.float32)
        if (
            trace.shape != self.membrane_trace.shape
            or not np.isfinite(trace).all()
            or (trace < 0).any()
        ):
            raise ValueError("Invalid membrane eligibility checkpoint")
        super().restore(arrays, metadata)
        self.membrane_trace = trace.copy()
