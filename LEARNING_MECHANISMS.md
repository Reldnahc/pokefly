# Learning mechanism notes

2026-09-19. Research notes, not an implemented new circuit or a gameplay result.
The registered whole-game practice and retained-weight tests take priority.

Current status at 07:40 UTC: v11's alternative local likelihood rule failed
the required two-seed memory screen (one cue at 53.70%, below 55%), despite
positive overall association effects. The unchanged v10 model is now getting
four more WHOLE fresh games per retained brain: 96k total decisions across six
attempts. Final retained tests will compare each child with its 24k parent and
the original brain. This is a training-dose test, not a new circuit or proof
that more experience will work; see `FOLLOWTHROUGH_RESULTS.md` for artifact IDs.

### Delayed-feedback diagnostic infrastructure (software only)

`probe_visual_curve.py --reward-delay-decisions N` can now deliver an assay
choice's scalar feedback N decisions later, while the existing neural dynamics
and eligibility continue normally. It does not latch credit, choose an action,
change Pokemon rewards, or add a learned external component. Cue presentation
remains the existing continuous 16-decision alternation, not a blank delay;
this distinction matters when interpreting any future result.

Raw histories distinguish feedback earned by a choice from feedback delivered
now. A final pending tail is saved without an extra training flush; continuation
and interrupted acquisition preserve it exactly. Shuffling matches originating
cue and DELIVERY checkpoint interval, with the pending tail shuffled separately,
so different feedback amounts cannot explain a paired/control difference.
The retention auditor refuses to pool different delays. Zero delay preserves
the historical permutation, and the saved non-unit correct-reward recovery bug
is fixed. Synthetic assay synapses remain excluded from gameplay.

This is tested infrastructure, NOT a neural learning result or a queued new
study. CPU regression: 599 Python tests, 26 JavaScript tests and lint,
`verify-cfd77e710d704809af5c378822176f81`. The current whole-game extension and
its retained tests retain priority; no delayed neural assay has run.

### Command-channel evidence gap and read-only wiring check

The strong-v10 association result tests left/right choices. The recorded older
A/B and Up/Down failures used weaker-vision models; they do not establish the
current model's command-channel capacity. No strong-v10 A/B assay has run.

A CPU-only count of the installed original sparse graph (25,582,938 edges;
166,700 neurons) finds 233 positive incoming edges to the two mapped A cells
and 842 to the two B cells. Of these, B has 321 edges from neurons classified
as visual_projection; A has none from that superclass. A nevertheless has
positive inputs from descending, intrinsic, ascending and other motor cells:
indirect visual information is possible. The mapped Up cells likewise have
only three direct positive visual_projection inputs, and Down has none. These
are graph/sign-classification counts, NOT measured response strengths or proof
that any channel lacks usable vision or learning capacity.

All 193 intrinsic neurons with direct edges into A are also parents of some
descending neuron. Thus the existing optional DN-based one-hop plastic scope
already includes those neurons; changing its selection to mention A would not
by itself add this missing layer. No scope, weight, neuron or mapping was
changed by the check. The fixed-neutral-screen motor-drift probe could test
tonic activity after training, but it does not test A/B cue discrimination or
gameplay skill. None of these observations changes the active whole-game
extension protocol or constitutes a queued new neural study.

### Individual weight limits are not broadly saturated

A read-only CPU check of both families' completed 24k, 42k and 60k checkpoints
finds zero of 495,962 plastic edges within 1e-5 of either individual factor
limit (0.25 or 4.0). At 60k, median absolute changes from original factors are
0.06163 and 0.06184; factor ranges are 0.25007..2.09829 and 0.25451..2.13254.
Thus wholesale clipping at individual bounds is not observed in these saved
brains. This does not exclude transient clipping, mean-input/resource limits,
tonic motor drift, or ineffective learning. It is not another neural trial.

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

A later whole-opening replay of v10 practice4501 finds50 decisions/12 neural
seconds between final accepted move and its legacy encounter-end victory
reward. An isolated0.6-second contribution would retain2.061e-9, again not a
measurement of the net trace after intervening activity. The reward itself is
correct and the loss in practice4502 is independently verified with zero player
HP/no surviving party. See the05:20 UTC follow-through entry. Current frozen
tests are unchanged; this does not demonstrate that timing alone is a solution.

Historical delayed-credit failures also have a limited scope. The30-second
covariance/dual profiles and impulse/noise-dual profiles used
intrinsic-neutral-v1, uncalibrated hybrid vision and individual0.75..1.25 bounds;
the noise-dual profile used perturb-v2. They were NOT the stronger calibrated
visual circuit plus anchored mean-input rule in v10. Their registered failures
still stand, but do not establish that a strong-vision/anchored delayed variant
must fail. No such new variant is implemented or run here, and the conditional
v11 queue has not been changed by this read-only comparison.

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

### Upstream recheck, September19

The current [fly.ai README](https://github.com/alextitonis/fly.ai) still lists
internal plasticity as missing/future work. Its interface-learning examples
are not an internally learning replacement for Pokefly. The current
[DOOMFLY protocol](https://github.com/nftechie/doomfly/blob/main/docs/doom-live-training.md)
explicitly retains failed visual, conditioning and survival gates; enabling its
learning rule on a stream did not establish retained skill.

The authors of [fly-api](https://github.com/dtch1997/fly-api) report odor-specific
KC-to-MBON depression, but their navigation demonstration uses a reduced
olfactory circuit and an engineered lower-MBON-valence steering/arrival rule.
That is not the unchanged anatomical motor pathway required here. It does not
provide a validated pixel-driven, delayed whole-game learner to import. These
are checks of the authors' documented methods, not independent replications;
no external code, dependencies, learned weights or replacement controller were
installed. The current tests and conditional candidate queue remain unchanged.

### Assay reward scale is not gameplay reward scale

Read-only counts from v10's completed first401 game (6,000 decisions) find
5,712 zero rewards,274 rewards of0.05,9 of0.1,4 of1.05 and one of3.0.
Its raw positive sum is21.8; summing positive tanh(reward) gives18.7079.
In contrast, the ROM-free paired501/601 assays each ran8,192 decisions and
delivered2,035/2,161 positive unit rewards, with positive tanh sums1,549.8441
and1,645.8050. Source artifacts are internal-learn-20260919T035428Z-c404bf and
visual-learning-curve-20260919T031016Z-36197e / ...-c436d2.

These are descriptive signal totals, NOT net synaptic updates, gradient
signal-to-noise measurements or a demonstrated cause of poor game learning.
The actual rule subtracts a reward baseline and multiplies local eligibility;
game and assay trajectories/delays also differ. A cue-capacity pass therefore
does not validate learning sensitivity at the game's reward scale/density.
Likewise, a strong-unit-reward assay failure is not proof that every lower-dose
task must fail. All older failures remain failures of their stated screens.
No reward values, learning parameters or registered current tests were changed
by this inspection. Any future reward-scale assay must be separately declared,
retain shuffled controls, and never export synthetic weights to Pokemon.

The ROM-free `probe_visual_curve.py` now accepts `--correct-reward` (positive,
finite, default1.0). For example,0.05 tests the existing tile-reward amplitude,
but still does NOT reproduce the game's sparse, delayed reward schedule.
Continuation cannot silently change that amplitude. The retention auditor
also refuses to pool different correct/incorrect feedback strengths; old
reports without these fields retain their original1.0/0.0 semantics.
This is measurement infrastructure only. No non-unit assay has been run,
no synthetic weights enter gameplay, and both the active v10 protocol and
conditional v11 protocol still use their previously registered unit reward.
CPU-only regression530 Python/26JavaScript/lint passes
(verify-9e4dc6bdd8b94ac8a71814f8aaf4ec3c). Re-auditing the SAME completed v10
memory panels still passes, visual-retention-gate-20260919T043929Z-b1c63b;
this is not a new neural trial or independent replication.

The v10 fixed-reference homeostatic candidate completed its24k-budget retained
game tests with faster starters for one lineage but inconsistent later progress;
both failed the registered preservation screen. V11's immediate cue-memory
tests now run. Neither local likelihood credit nor the prospectively enlarged
whole-game practice budget implements a learned long-horizon value predictor.
More exposure is a hypothesis to test, not a demonstrated fix for reward delay.

Any subsequent memory-circuit proposal needs a reproducible anatomical mapping,
explicitly justified signs/modulation, controlled acquisition and retention,
then full-game tests carrying synapses across new games. Our partial type-level
compartment assignments are not fine synaptic anatomy. Do not invent MBON
button labels, import a trained external critic, train on game-memory features,
or initialize Pokémon with weights from synthetic cue/oracle assays.

No new battles may start from a battle save. Learning to reach each battle
again is part of the task. Exact continuation of the fly's own interrupted
game remains distinct from resetting it to a selected stage.
