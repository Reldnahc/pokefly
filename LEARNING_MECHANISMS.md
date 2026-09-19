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

Whole-game replays of source 402's two victories now measure 8/9 decisions
from the last accepted move selection to the earlier final-knockout reward
(1.92/2.16 neural seconds). Passive decay of an isolated 0.6 s trace contribution
would leave 4.08%/2.73%. This does not measure the net eligibility trace after
other activity, establish which move caused the win, or establish an effective
replacement time constant. Both replays reproduce every logged state/reward;
see the 00:49 UTC entry in `FOLLOWTHROUGH_RESULTS.md` for evidence paths.

The installed metadata includes 4,064 KC-prefixed cells, 97 MBON-prefixed cells,
316 PAM-prefixed cells and 16 PPL1-prefixed cells. Merely containing these cells
does not implement their biological memory mechanisms. The active sensorimotor
rule does not also train KC-to-MBON connections. Historical compartment-only
variants exist; they failed their earlier controlled association tests. Do not
present them as a new untested solution or assume longer traces alone solve it.

A read-only sparse-graph count finds 4,407 direct visual-KC-to-MBON edges,
10,164 visual-KC-to-PAM edges, 1,500 visual-KC-to-PPL1 edges, and 2,119/419
MBON-to-PAM/PPL1 edges. Thus some relevant anatomical connections exist; this
does not establish their functional learning signals. The installed builder
assigns fast connection signs from broad neurotransmitter classes, treating
types outside its inhibitory list as positive. That is not dopamine receptor
kinetics or a learned reward predictor. No extra plastic scope or critic was
enabled by this connectivity inspection.

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

The Bennett model is specifically not a ready-made temporally extended critic:
its Discussion distinguishes its trial-level Rescorla-Wagner model from a
future time-resolved TD extension. Copying its cue-choice procedure would also
not demonstrate control through this project's unchanged motor neurons.
[Bennett et al., Discussion, Prediction 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8105414/).

An additional reduced incentive-circuit model provides type-level memory-loop
identities and source code, but explicitly includes inferred connections and
behavioral readout assumptions. Its anatomical tables are useful for checking
which existing cells/edges can support a hypothesis, not a license to relabel
MBONs as Pokemon buttons or add a separate action selector. No code or learned
weights from that model are deployed here.
[Gkanias et al., 2022](https://elifesciences.org/articles/75611).

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
