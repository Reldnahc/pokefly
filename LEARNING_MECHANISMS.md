# Learning mechanism notes

2026-09-19. Research notes, not an implemented new circuit or a gameplay result.
The registered whole-game practice and retained-weight tests take priority.

## What the current fly learns

The wide-v3 showcase and timing-only v7 candidate modify existing positive
connections into anatomical descending/motor neurons. The update combines
local presynaptic history, the target neuron's existing noise perturbation,
and a scalar reward signal. The decoder and visual calibration do not learn.
See `src/pokefly/plasticity.py` and `src/pokefly/plastic_edges.py`.

Their reward baseline is a running scalar average, not a visual-state value
prediction. The main eligibility trace has a 0.6-neural-second time constant.
Consequently, verifying that a victory reward arrives at the final knockout
does not establish that the earlier move choice still receives useful credit.
The actual-game reward ledger and the learning rule must be evaluated separately.

The installed metadata includes 4,064 KC-prefixed cells, 97 MBON-prefixed cells,
316 PAM-prefixed cells and 16 PPL1-prefixed cells. Merely containing these cells
does not implement their biological memory mechanisms. The active sensorimotor
rule does not also train KC-to-MBON connections. Historical compartment-only
variants exist; they failed their earlier controlled association tests. Do not
present them as a new untested solution or assume longer traces alone solve it.

## What the literature supports—and does not

A reduced mushroom-body model uses output-neuron feedback to dopamine neurons
to learn reinforcement predictions. Some of its circuitry is hypothesized,
and its cue-choice experiments are not whole-game control. It is motivation
to investigate internal prediction, not a drop-in validated Pokémon learner.
[Bennett et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8105414/).

Physiological work during olfactory conditioning found dopamine/MBON feedback
interactions between short- and long-term memory units. It supports studying
multiple internal learning timescales and feedback, but does not supply our
MaleCNS model's missing cell-specific parameters or demonstrate visual game
learning. [Huang et al., 2024](https://www.nature.com/articles/s41586-024-07819-w).

Our inference: internal predictive memory may ultimately be needed for more
reliable delayed credit. That is not evidence that it explains the current
failures; motor bias, visual representation and credit noise are alternatives.

## Constraints on a future investigation

First finish the registered whole-game retained tests and the queued internal
homeostatic candidate. The latter addresses accumulated mean-input bias, not
long-horizon prediction. Its numerical success is not behavioral success.

Any subsequent memory-circuit proposal needs a reproducible anatomical mapping,
explicitly justified signs/modulation, controlled acquisition and retention,
then full-game tests carrying synapses across new games. Our partial type-level
compartment assignments are not fine synaptic anatomy. Do not invent MBON
button labels, import a trained external critic, train on game-memory features,
or initialize Pokémon with weights from synthetic cue/oracle assays.

No new battles may start from a battle save. Learning to reach each battle
again is part of the task. Exact continuation of the fly's own interrupted
game remains distinct from resetting it to a selected stage.
