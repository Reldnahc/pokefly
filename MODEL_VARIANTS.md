# Neural variants and what the display means

The purpose is to let the fly **play**, not demand that it master Pokemon.
We assess screen-dependent activity, delivered actions, interaction with the
game, internal memory, and retained learning separately. Exploration totals
and the original house-exit milestone are measurements, not a route to teach.

## Running the explicit experimental variant

```powershell
.\scripts\start.ps1 -Intro
```

The launcher defaults to `sensorimotor-bounded-v1`, also selectable with
`train --config configs/sensorimotor-bounded-v1.json`. This includes the
incoming sensory isolation, fixed neutral-image calibration and bounded internal
sensorimotor learning described below. Use `-Profile sensory-isolated-v1` for
the previous model, or `-Profile hybrid-v1` for the unisolated control.
Bare `train` without a config
retains baseline neural dynamics. Fresh runs use the approved
`parallel-v2` button-adapter correction: one direction plus one A/B/Start command.
`--resume` restores its saved profile **and decoder**; older checkpoints remain
`exclusive-v1`. Do not supply another configuration during exact resume.
No ROM or original connectome data file is changed. No motor threshold, reward
category, reward amount, or movement/button assignment was changed in this work.

Other research profiles remain **opt-in**. Current findings and reproduction
details are in [FOLLOWTHROUGH_RESULTS.md](FOLLOWTHROUGH_RESULTS.md); historical
results are in [RESULTS.md](RESULTS.md). The recorded protocol is
in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).

| Profile | Change from sensory-isolated-v1 |
| --- | --- |
| `stream-v1` | Interleave neural updates with fresh raw frames during each pulse. |
| `endpoint-v1` | Timing control for stream; repeat the pulse's final frame. |
| `quiescent-v1` | Hold the unused nonvisual sensory population at rest. |
| `compartment-ema-v1` | Partial compartment gates and expectation-centered plasticity. |
| `intrinsic-v1` | Fixed neutral-image excitability calibration; earlier KC learning rule. |
| `sensorimotor-v1` | Calibrated dynamics plus broad motor-input covariance plasticity; locks in. |
| `sensorimotor-normalized-v1` | Exact local incoming-strength normalization; weak motor acquisition. |
| `sensorimotor-bounded-v1` | Current launcher: every eligible edge bounded to 0.75--1.25x original. |
| `sensorimotor-budget-v1` | Wider individual edge range, total incoming strength bounded +/-25%. |
| `sensorimotor-perturb-v1` | Same bounded input budget, credit from actual neural-noise perturbations. |
| `sensorimotor-perturb-v2` | Linear, unclipped noise eligibility to remove the v1 clipping bias. |
| `sensorimotor-perturb-v3` | Also centers presynaptic history against its local 10 s baseline. |
| `sensorimotor-centered-v1` | Pre/post-centered spike covariance, without neural-noise credit. |
| `sensorimotor-delayed-v1` | Default's bounded synapses, but a 30 s credit trace. |
| `sensorimotor-dual-v1` | Equal mixture of 0.6 s and 30 s local eligibility traces. |
| `sensorimotor-reset-v2` | Default rule with calibration fitted across repeated neural resets. |
| `compartment-reset-v2` | Partial KC/MBON rule with reset-conditioned calibration. |
| `sensorimotor-low-noise-v1` | Perturbation-v3 with smaller noise and separately refitted calibration. |

Use `-Profile NAME` for a **new** experiment. `-Resume` and `-Weights` restore
the saved profile; neither silently converts a saved brain into another model.

## Neutral intrinsic calibration and internal sensorimotor learning

`intrinsic-neutral-v1.npz` stores fixed voltage-drive offsets indexed by exact
neuron body IDs. Its checksum is part of checkpoint identity. On neutral gray
(RGB 128), every nonsensory spiking cell follows the SAME equation:
`bias += 0.001 * (1 Hz * dt - spike)`, clipped to `[-0.14, 0.2]`, for 10,000
neural steps with seed 707. Graded retinal/lamina cells and all sensory classes
are excluded. Offsets are frozen before gameplay; no game frames, motor labels,
reward or button quota enters fitting. Original connectome files are untouched.
The calibration is an engineering hypothesis, not measured cell-specific
physiology. Fly firing-rate homeostasis motivates investigating excitability,
not the chosen universal target ([primary study](https://elifesciences.org/articles/45717)).
Fresh-reset rates can differ sharply from the fitting trajectory: KC and MBON
rates fall substantially, so this is not a validated solution for memory dynamics.

Reproduction, without a ROM (refuses to overwrite an existing artifact):

```powershell
.\.venv\Scripts\python.exe scripts/probe_intrinsic.py --calibration-only `
  --device cuda --export fly-data/intrinsic-neutral-v1.npz
```

The initial sensorimotor variants select EXISTING positive inputs to every neuron whose
anatomical superclass is `descending_neuron`, `cb_motor` or `vnc_motor`:
495,962 edges across 2,129 targets. The separately described score-v3 also permits
existing inhibitory inputs while preserving their signs. Selection never consults the seven-button
registry. No readout/encoder is fitted, no connection is added and no sign changes.
This is a broader experimental plasticity hypothesis, not an anatomically
resolved dopamine mechanism. Fly motor self-learning is not necessarily a
mushroom-body-only process ([primary study](https://pubmed.ncbi.nlm.nih.gov/38779314/));
that does not validate this rule or these selected cells.

The covariance rule uses a 0.2 s presynaptic trace, postsynaptic firing centered
against a preceding 10 s baseline, a 0.6 s eligibility trace and a 1 Hz activity
reference. Eligibility is clipped to +/-5. Scalar feedback is `tanh(reward)`
minus a preceding 30 s reward average; learning rate is 0.02. Reward omission
can produce a negative internal prediction error, but no negative game reward
or revisit penalty was added. Dopamine here remains a synthetic teaching signal,
not a concentration or a measured molecular model. The broad idea is
[reward-modulated spiking plasticity](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf),
not an exact reproduction of that paper.

The default bounds each weight to 0.75--1.25 of its original value. The optional
budget variant instead permits 0.25--4x individual weights while projecting each
target's total positive incoming strength into 0.75--1.25 of its original sum.
Projection is bounded, iterative and approximate to 1e-5 relative tolerance;
it does not aim for any action distribution. Exact-normalization control fixes
the sum at 1x and is not used by default.

The optional perturbation rule uses the simulator's actual independent noise
events, centered against their known Bernoulli probability, instead of firing
minus a global firing baseline. Previous presynaptic activity supplies local
eligibility; no action information or external critic enters it. Noise draws,
noise amplitude, neural dynamics and button mapping are unchanged. This is a
finite-noise/filtered-trace approximation inspired by
[Fiete & Seung](https://doi.org/10.1103/PhysRevLett.97.048104), not their
conductance model, an exact gradient guarantee, or validated fly biology.

Version 1 clips eligibility to +/-5 like the covariance rule. That clipping is
not neutral for rare Bernoulli noise: positive innovations are rare and large,
so clipping makes expected eligibility negative even with fixed presynaptic
history. Version 2 keeps the eligibility filter linear (incoming activity is
still bounded, and weights retain the same bounds/input budget). An enumerated
two-outcome regression checks zero conditional mean. This corrects that specific
mathematical bias; it does not prove useful behavioral learning. Version 1 remains
unchanged for exact old-checkpoint continuation. Version 2 is opt-in and is not
the launcher default.

Perturbation-v3 uses the previous local 10 s spike average to center presynaptic
history too, clipping that input contrast to [-5,5]. The covariance-v2 rule in
`sensorimotor-centered-v1` tests the same local contrast with postsynaptic spike
innovation instead of noise. Its presynaptic trace uses only past spikes, not
the same step's new presynaptic spike. These are engineering hypotheses, not
external encoders or learned button classifiers.

`sensorimotor-perturb-normalized-v3` is a one-factor research control: same v3
rule/individual bounds but exact positive-input normalization rather than a
+/-25% total budget. Existing synapses compete without changing total incoming
strength. This tests global excitability drift versus cue-specific learning;
it is not a fitted motor quota and is not promoted. The frozen-input capacity
audit still permits both cue-current signs under this constraint, which is
only a first-order capacity estimate, not a learned solution.

The delayed-credit candidate changes eligibility from 0.6 to 30 neural seconds.
It improves the tested battle outcomes but weakens immediate operant acquisition.
The dual candidate maintains BOTH traces on the same existing synapses and
uses their arithmetic mean in the unchanged reward update. It has no game-state,
reward-category or action-dependent switch. Both traces are checkpointed; old
models receive an inactive `slow_eligibility_seconds=0` default and no new arrays.
The fused dual kernel matches two independent trace updates exactly.

Optional calibration research (not needed for the default):

```powershell
.\.venv\Scripts\python.exe scripts/probe_intrinsic.py --calibration-only `
  --device cuda --reset-every 200 --reset-probe-decisions 256 `
  --export fly-data/intrinsic-neutral-reset-v2.npz
.\.venv\Scripts\python.exe scripts/probe_intrinsic.py --calibration-only `
  --device cuda --reset-every 200 --noise-amplitude 0.08 `
  --export fly-data/intrinsic-neutral-reset-low-noise-v1.npz
```

The reset fit uses seed `707 + reset index` and the same neutral gray/uniform
homeostasis equation. The low-noise profile uses amplitude 0.08, not 0.22;
its calibration is separately fitted at that amplitude. No game or motor-label
feedback enters either fit. The reset fit improves KC/MBON activity but did not
establish visual learning. The low-noise candidate has not passed either.
Live verification explicitly skips optional profiles whose calibration artifact
has not been generated; it does not silently substitute another model.

Checkpoints retain weights, eligibility, presynaptic/baseline traces, slow reward
expectation and all dynamics. Exact resume keeps them; `--weights` intentionally
keeps connections but resets fast dynamics and reward expectation for a new trial.
Older configurations receive only inactive compatibility defaults on restore.
The faster fused eligibility kernel is tested bit-for-bit against NumPy 2.

## Input timing: stream-v1 and endpoint-v1

The default `snapshot-v1` still integrates 12 neural updates from one pre-action
image, selects buttons, executes 24 game frames, then applies accumulated reward.
Stream samples the actual raw framebuffer at game-frame offsets 2, 4, ..., 24,
integrating one neural update per image. No intermediate game frame is invented.
The original 20 ms neural timestep, 23 held frames plus one released frame,
decoder, general reward amounts, and pixel mapping are unchanged.

The two new modes prepare one explicit 12-step static input window at startup,
then repeat: select from the preceding window, execute the pulse and integrate
the next window, apply accumulated reward. `endpoint-v1` repeats the endpoint
image instead of interleaving images, keeping the same neural budget and reward
ordering as stream. This is a control for the changed eligibility timing, not
another visual model. Game time and neural time remain different clocks.

Pending spike counts, last retinal drive, frame hashes and sample offsets are
checkpointed. Exact resume does not add another startup window. The display
shows the **last actual frame** of the window that selected the displayed pulse;
its activity spans the whole window. It is not a replay of all intermediate
images. Changing images more frequently works, but did not consistently improve
Up or exploration in the matched frozen tests, so snapshot remains the default.

## Quiescent unused sensory boundary: quiescent-v1

This separately versioned candidate requires existing nonvisual sensory
isolation and uses precisely the same 11,931-cell anatomical mask. After drawing
the unchanged global neuronal-noise stream, it holds those cells' voltages at
zero before diagnostic injection and spike detection. Thus their normal
tonic/noise activity cannot drive the network; visual input is not silenced.
Original outgoing connections remain stored, and other cells receive exactly
the same random draws. This is a stronger experimental boundary condition, not
a biological correction or a selective forward-neuron intervention.

The circuit screen did not improve Up consistently. The setting defaults to
false, including restoration of older checkpoints. It was not promoted to the
normal experiment or combined with other changes to hunt for a favorable route.

## Persistent sensory isolation: sensory-isolated-v1

The user selected this controlled simplification for the pixels-only experiment,
following the forward-current audit. It combines the hybrid dynamics and targeted
internal plasticity below with `dynamics.isolate_nonvisual_sensory = true`.

The fixed anatomical mask selects `superclass` labels containing `sensory`,
excluding **every** cell in the original `brain.visual` population: 11,931 cells
in the installed MaleCNS model. Their incoming network current is set to zero
after synaptic propagation, before integration. It is identical to the tested
temporary intervention, not a mask fitted to Up performance.

- Original weight/sign arrays, visual input/feedback, and non-sensory currents
  remain unchanged. No direct DNg100 stimulation or inhibition removal is used.
- Isolated cells retain outgoing connections, tonic input and neuronal noise.
  They are not deleted or completely silenced; this is not a simulated intact fly.
- The setting, cell count and selected-body-ID checksum are stored in neural
  identity/checkpoints. Resuming an old checkpoint defaults isolation to **off**;
  switching models is never implicit, including weight-only retention runs.
- The dashboard labels the setting and clarifies that its connection count
  describes the original anatomical graph. RAM remains reward/measurement-only.

This is a persistent, reversible experimental profile, not a claim that real
flies lack sensory feedback. The unisolated control remains available. Changing
the experimental assumption later requires a new trial/configuration, not
silently editing a running or resumed experiment. See [RESULTS.md](RESULTS.md).

## Dynamics: hybrid-v1

All original signed connections remain. This is a local modeling hypothesis,
not a downloaded validated replacement nervous system or a reproduction of
flyvis. The constants below were fixed using non-game neural checks before the
gameplay evaluation. They have not been fitted to physiological recordings.

- 6,006 photoreceptors and 8,884 L1--L5 cells transmit graded values, not spikes.
- Photoreceptor target is `clip(pixel_drive + signed_network_current, 0, 1)`.
  Lamina target is `clip(0.6 + signed_network_current, 0, 1)`.
  Both follow the target with a 40 ms time constant, and transmit 0.25 times
  their normalized state through their existing signed outgoing connections.
  The retinal map, full-screen input, and histaminergic weight signs are unchanged.
- Other cells retain the baseline spiking equations. Each KC has tonic input
  0.02 instead of 0.14 and a spike-triggered adaptive current: +0.25 on a spike,
  exponential decay with a 1 s time constant, subtracted during integration.
  This is an engineering stabilization hypothesis, not a claim that these are
  measured KC parameters. No KC recurrent edges are cut or silently reweighted.
- Neuronal noise, 20 ms timestep, and all other fast
  connectome weights remain as before. Continuous propagation multiplies each
  connection by its actual graded value; it does not round activity to spikes.

Upstream's homogeneous spiking model does not represent the graded early
visual relay; its author documents that limitation in
[fly.ai's findings](https://github.com/alextitonis/fly.ai#what-we-found).
Our minimal hybrid relay addresses that specific modeling gap, not every
missing part of fly vision. Other visual neurons are still modeled as spiking.

`circuit-evaluate` compares baseline, KC-adaptation-only, and hybrid variants
with identical noise and static images. It reports full-window and post-startup
spike differences. A separate direct neural injection checks each of the seven
readouts. This injection is never available to the normal game controller.
All readouts passed, so there was no justification to fit per-button thresholds
or introduce balancing/forced actions. Splitting direction/function arbitration
now lets rare valid Up signals reach the game. Pixel-driven access to the forward
cells remains poor. [RESULTS.md](RESULTS.md) records the separate current audit:
nonvisual sensory recurrence contributes to persistent forward inhibition.
Those current-removal experiments are not enabled by this profile.

## Learning: dan-targeted-v1

The original global centered-eligibility rule is retained as `centered-v1`.
The experimental rule selects 335 `KCg-d`/`KCab-p` visual-KC candidates and their
4,407 existing KC-to-MBON edges. The choice of cell classes is motivated by
[Ganguly et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11228034/),
which describes direct and indirect visual input to those classes in FlyWire.
The population counts here come from our MaleCNS files, not that paper, and
matching type names does not validate every cell's visual function in our model.

For each plastic edge, shared DAN input to its KC and MBON defines an overlap
gate: sum `sqrt(abs(DAN->KC weight) * abs(DAN->MBON weight))` across the selected
DANs, then max-normalize across eligible edges. PAM and PPL1 define separate
channels. This produces 3,549 positive-channel and 3,675 negative-channel gated
edges. Neither channel creates new connections.

Reward still enters as one scalar. Positive reward drives the PAM proxy;
negative reward would drive the PPL1 proxy (current gameplay rewards do not
include punishments). This coarse valence assignment is a stated assumption,
**not validated valence for every DAN or a resolved compartment map**. Release
is synthetic; it is not derived from measured dopamine or simulated DAN spikes.
The baseline fast effects of DAN connections have not been reclassified.

KC spike probability uses a 0.2 s trace, converted to activity relative to
5 Hz and clipped to [0,1]. A 2 s eligibility trace retains this activity. For
original-weight multiplier `f`, eligible activity `e`, and anatomical gate `g`:

```text
f <- clip(f - 0.02 * abs(tanh(reward)) * g * (e + f - 1), 0.25, 4)
```

Active tagged inputs depress; inactive inputs recover toward the original
weight during modulation. With zero reward or frozen learning, weights do not
change. Signs and sparsity remain fixed. Depression/recovery is motivated by
[Gkanias et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8975552/),
but this bounded overlap-gated rule is our approximation, not their complete
incentive circuit. Anatomical compartments and receptor-specific effects remain
unfinished modeling work; this variant must not be described as solved dopamine.

`association-evaluate` is a separate raw-image task, never a Pokemon reward.
Left/right half-white images are presented with matched noise; A gets reward,
B does not. Controls freeze weights or alternate reward between A/B (the report
labels this balanced nonpredictive control `shuffled`; it is not random shuffling).
Every presentation starts with fresh dynamics and a black warmup, isolating
synaptic storage from transient activity. Tests have no reward. Reversal rewards B.
Fixed-input suppression measures a stored weight effect, not a behavioral choice.

The measured results are in [RESULTS.md](RESULTS.md). The new learning rule did
**not** establish reliable image-specific association, reversal, or improved
game exploration. More movement and changing weights are not sufficient evidence.

## Partial learning candidate: compartment-ema-v1

This opt-in candidate keeps the same 335 visual-KC candidates and 4,407 existing
edges, but only gates a conservative subset using these type-level assignments:

| Compartment proxy | DAN type | MBON types | KC type | Synthetic feedback |
| --- | --- | --- | --- | --- |
| gamma5 | PAM01 | MBON01, MBON27 | KCg-d | Positive reward |
| alpha1 | PAM11 | MBON07 | KCab-p | Positive reward |
| gamma1pedc | PPL101 | MBON11 | KCg-d | Negative reward |

Assignments are motivated by [Li et al. (2020), figures 22, 26 and 30](https://elifesciences.org/articles/62576).
They are **cross-dataset, partial type-level approximations**: our installed
MaleCNS metadata lacks synapse-level compartments and the finer DAN subtypes
needed to validate all these assignments. Missing types or an empty shared-input
gate fail loudly. The gate uses only existing shared DAN targets, the same
geometric-mean overlap as above, and normalization within each compartment.
In the installed data, 416, 258 and 186 edges respectively have nonzero gates;
all other plastic-edge slots have zero gates. No MBON is manually assigned a
button or an approach/avoidance function; no new fast connection is created.

Each compartment has an internal recency average of its synthetic release,
with a fixed 30-second **neural-time** constant. With `D = tanh(reward)`, positive
proxies receive `max(D,0)` and the negative proxy receives `max(-D,0)`. For
release `r`, old expectation `m`, eligibility `e`, and gate `g`:

```text
prediction_error = r - m
f <- clip(f - 0.02 * sum_compartments(prediction_error * g) * e, 0.25, 4)
m <- exp(-elapsed_neural_time / 30) * m
     + (1 - exp(-elapsed_neural_time / 30)) * r
```

Positive error depresses tagged inputs; reward omission after prior positive
feedback can potentiate them. This is an internal plasticity signal, **not** a
new negative game reward or a revisit punishment. From zero expectation, absent
reward leaves weights unchanged. Frozen mode never updates weights or the
expectation. Signs, sparsity and fixed action mapping remain intact.

Expectation-centering is motivated by [Rajagopalan et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/37733736/).
This recency average is a simple internal adaptation proxy, not a reproduction
of that circuit, a learned state-value function, temporal-difference learning,
or a solution to long-horizon credit assignment. It receives no action identity,
cue label, game coordinates or externally learned features. Actual DAN spikes
still do not drive the synthetic release. Exact checkpoints retain all feedback
state; weights-only loading starts with fresh modulation and neural dynamics.

`scripts/evaluate_learning_choices.py` tests actual fixed-decoder choices in a
separate ROM-free operant assay. Synthetic cue/choice rewards belong **only** to
that diagnostic, never the Pokemon reward observer. Counterbalanced assignments,
within-cue reward permutations, frozen controls, reward-free retention, reversal,
and disk reload distinguish stored weight changes from learned choice specificity.
The assay is a limited capacity screen, not a claim that arbitrary directions
must be learnable through these particular synapses.
The completed three-seed comparison did not establish reliable target-specific
choices or reversal for either rule. The candidate remains experimental and
opt-in; stored weight changes alone did not justify promotion.

## Optional likelihood-score model: sensorimotor-score-v1

This is a separate engineering hypothesis, not a promotion or a biological
validation. The existing connectome, graded retina, input isolation and fixed
button decoder remain. Spiking neurons use logistic escape probability
`p = sigmoid((voltage - 1) / 0.05)`, with independent checkpointed neural
randomness, in addition to the original background current noise. Graded cells
never spike. Temperature zero retains the original threshold model exactly.

For an existing excitatory synapse with original weight `base` and factor `f`,
the local membrane-sensitivity trace uses actual previous transmitter release:

```text
z = membrane_decay * z + previous_presynaptic_release
score = (postsynaptic_spike - p) * base * neural_gain * z / temperature
eligibility = exp(-dt / 0.6) * eligibility + score
z = 0 if the postsynaptic neuron spiked, otherwise z
f += 0.5 * (tanh(reward) - previous_reward_mean) * eligibility
```

As in the budget variant, factors are limited to 0.25--4 and each postsynaptic
positive-input total to +/-25% of its original value. The instantaneous score
has zero conditional expectation and matches a finite-difference derivative of
a fixed neural spike trajectory including hard resets. Discounting, online
updates and constraints mean this is NOT a guarantee of whole-game convergence.
The motivation is stochastic-neuron reward learning in
[Florian (2007)](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf);
our discrete logistic/hard-reset model is not that paper's exact simulation.
No desired spike train, chosen button, cue identity, RAM or external critic
enters the learner. All plastic connections still belong to the original fly.

The separately generated fixed calibration uses only neutral gray:

```powershell
.\.venv\Scripts\python.exe scripts/probe_intrinsic.py --calibration-only --device cuda --spike-temperature 0.05 --reset-every 200 --reset-probe-decisions 128 --export fly-data/intrinsic-neutral-score-v1.npz
```

Calibration files are immutable; do not overwrite an existing artifact. The
same neutral procedure and 1 Hz target apply to every nonsensory spiking cell,
without motor labels or game rewards. Initial held-out neutral probes yielded
KC 0.54--0.56 Hz and MBON 0.86--0.91 Hz. These are activity checks, not learning
results. Behavioral and actual-ROM checks are recorded in the active plan.

`sensorimotor-score-v2` uses the SAME firing model/calibration and raw score,
but divides eligibility by original synaptic magnitude before the factor
update, with learning rate 0.002. This positive diagonal preconditioner changes
absolute weight updates from proportional to `base^2` to `base`; it gives weak
existing inputs a less disadvantaged learning scale. Bounds and local input
budgets are unchanged. No task labels determine this scaling. It is another
opt-in hypothesis, not evidence of a successful fly visual learner.

`sensorimotor-score-v3` extends v2 to existing inhibitory as well as excitatory
inputs to the same anatomically selected descending/motor cells. It never adds
edges or changes signs. The membrane likelihood score includes the signed base
weight; preconditioning divides by its absolute magnitude. Excitatory and
inhibitory input-magnitude budgets are enforced separately per postsynaptic
cell, so opposite signs cannot cancel to evade the bounds. It is an opt-in
capacity test; no behavioral success has been established.

`sensorimotor-escape-v1` retains the positive-input v2 rule but tests narrower
stochastic firing (temperature 0.02) with background Bernoulli current noise
disabled. Neuronal escape noise remains active; this is not deterministic or
action-level exploration. Learning rate 0.0008 scales down with temperature.
A separately frozen neutral calibration keeps the uniform rate-fitting protocol:

```powershell
.\.venv\Scripts\python.exe scripts/probe_intrinsic.py --calibration-only --device cuda --spike-temperature 0.02 --noise-hz 0 --noise-amplitude 0 --reset-every 200 --reset-probe-decisions 128 --export fly-data/intrinsic-neutral-escape-v1.npz
```

This tests whether broad, redundant neural noise masks the small sensory
contrasts. It does not alter the pixels, decoder, anatomy or game rewards.
It is not a measured fly noise model and remains unpromoted.

## Optional fixed movement bouts: sensorimotor-sustained-v1

Same calibrated dual-trace brain, but a separately versioned `sustained-v3`
fixed decoder. All four direction rates use a 1.0 neural-second exponential
trace and remain eligible during brief spike gaps until their trace drops below
the unchanged 0.5 Hz threshold. Opposing neural activity can immediately win
the competition. A/B/Start retain the existing 0.15 s trace, current-spike
requirement and Start cooldown. Every game pulse still explicitly releases.
No synthetic direction, wall detection, action quota, RAM feature or reward
input enters this adapter. Its constants are not learned; this is an engineering
interpretation of a movement bout, not a simulated body or measured fly duration.

The normal decoder/default remain unchanged. `evaluate_movement_bouts.py`
compares original versus sustained decoding with frozen original weights,
matched recorded starts and two noise seeds. The starting-state reset is a
diagnostic intervention. Any improvement here establishes control capacity,
NOT learning. Sustained choices, including ones driven by residual traces,
must not be described as new spikes in the dashboard or reports.

## Optional Start cadence: sensorimotor-sustained-menu-v1

Same sustained/dual model; only the existing fixed Start cooldown changes
from 1 to 5 neural seconds. The pIP10 neurons must still generate every Start
command, and A/B remain available during the cooldown. There is no game-state
input, menu detector, learned button adapter, action quota, or reward change.
Five seconds is a preselected engineering probe, not validated fly physiology.

Reason: measurement-only replay of the 30,000-decision dual continuation found
45.62% of emulator frames inside Start or nested menus, with 2,198 openings.
Pokemon's [overworld input handler](https://github.com/pret/pokered/blob/a1a22aaf84d1675bcdbaeb194592379d586d838e/home/overworld.asm)
checks Start before directions. `scripts/audit_menu_time.py` verifies all
recorded states/rewards while measuring this; its hooks never reach the policy.

`scripts/evaluate_start_cadence.py` checks original frozen weights, exact
starting-state hashes and saved model settings before changing this one factor.
It explicitly reuses the completed sustained controls for seeds 1201/1202,
6,000 decisions each, not counting them as new trials. The candidate is opt-in
and not yet promoted. See FOLLOWTHROUGH_RESULTS for outcomes and limitations.

## Optional fixed serial button delivery: sensorimotor-serial-v1

Same brain/decoder/rewards as sustained-menu-v1; only `button_timing` changes
from `simultaneous-v1` to `serial-v2`. A paired request uses half its fixed
frame budget for direction and half for function, releasing after each pulse.
At24 frames:11 direction,1 released,11 function,1 released. Single-channel
requests are unchanged. Both selected buttons originate in the neural decoder;
no input detector, forced command, action quota or reward enters this adapter.

This addresses conflicting game input priorities, not learning. The
[naming handler](https://github.com/pret/pokered/blob/a1a22aaf84d1675bcdbaeb194592379d586d838e/engine/menus/naming_screen.asm#L124-L187)
prioritizes directions, whereas the overworld checks Start first. This can
mask valid neural commands when combined. Frozen copied-state and whole-game
comparisons are recorded in FOLLOWTHROUGH_RESULTS; it remains opt-in.

Timing is checkpointed and replayed. Missing timing means the exact legacy
simultaneous path, never an implicit migration. Stream input samples retain
their same absolute frame offsets across phase boundaries. Logs record actual
phase schedules; dashboard amber identifies commands within the interval,
not a claim that serial buttons were held simultaneously.

## Optional delayed-credit mechanism tests

`sensorimotor-impulse-dual-v1` keeps the bounded0.6/30-second covariance model
but tests fixed impulse-balanced trace mixing. For `r=tau_slow/tau_fast`,
`E=(E_fast+r*E_slow)/sqrt(1+r+4*r/(1+r))`. This equalizes the immediate contribution
of the same event, then preserves fast-kernel variance under a continuous
shared-white-innovation approximation. Real neural innovations are correlated;
this is not an exact noise normalization or a measured biological mechanism.

An isolated extra-spike counterfactual exposed a negative aftereffect from the
adaptive postsynaptic baseline: with identical future activity, the old mean
tag can reverse sign before a delayed reward. `sensorimotor-noise-dual-v1`
instead uses the existing perturb-v2 tag and known independent noise mean.
`sensorimotor-event-dual-v1` tests a local pre-trace times actual-postspike tag,
without adaptive postsynaptic subtraction. Both retain SAME internal reward
mean/RPE centering, bounds0.75..1.25, timescales and learning rate. They alter
only existing internal synapses and receive no action or stimulus labels.

None passed the512-earning-decision,32-delayed-decision operant diagnostic.
Preserving a local tag does not establish useful causal learning. They remain
unpromoted; no synthetic diagnostic weights enter Pokemon. Checkpoints restore
all traces and modulation; missing `trace_mixing` means exact old mean mixing.

## Optional outcome timing and centered-innovation tests

Reward timing is separate from the learning rule. `encounter-end-v1` is the
unchanged default, including old checkpoints missing the field. The opt-in
`confirmed-outcome-v2` delivers existing outcome rewards at validated victory/
capture hooks when safe. `last-faint-v3` also checks for the final trainer faint:
living live active player, zero live enemy HP, valid enemy party count/index,
and no other enemy with cached HP. Stale active party HP cannot by itself prove
a wild win. Ambiguous/double-KO cases retain the conservative fallback. The
once-paid encounter flag is checkpointed; categories and amounts are unchanged.

`sensorimotor-outcome-v3` changes only reward timing from dual-v1;
`sensorimotor-serial-outcome-v3` changes only reward timing from serial-v1.
Both are research candidates, not new defaults or claims of retained learning.

`sensorimotor-score-delayed-v2` uses the existing stochastic model and summed
score trace, extended from 0.6 to 30 seconds. Its fixed learning rate is
0.000287484 (approximately the stationary independent-innovation variance
rescaling of 0.002). This approximation does not account for correlated neural
activity. The 32-decision delayed motor assay did not pass its behavioral gate.

`sensorimotor-score-centered-v4` subtracts the presynaptic neuron's PRIOR
10-second release EMA while integrating its local membrane tag. The innovation
is still the actual spike minus its conditional firing probability. This tests
removing tonic-input variance to favor sensory contrasts; it is NOT the exact
likelihood gradient of the original uncentered neuron. Existing excitatory
descending/motor inputs only, same bounds/input budget, rate, 0.6-second trace,
stochastic dynamics and fixed decoder as score-v2. Its additional release EMA
and signed membrane trace are saved/reset exactly. No action/cue label, RAM
feature, fitted readout or external critic enters the rule. Visual acquisition,
retention and reversal remain experimental, not established by the equation.

## Optional intrinsic motor adaptation

`sensorimotor-adaptive-v1` adds a spike-triggered current only in anatomically
classified descending, central-brain motor and VNC motor neurons. Every spike
adds 0.02; the current decays with a three-second time constant and subtracts
from membrane drive. It is not a chosen-button timer, action quota, collision
detector or new sensory channel. The mechanism follows the general modeling
idea of [spike-triggered adaptation](https://doi.org/10.1152/jn.00686.2005), but
these constants are engineering hypotheses, NOT measured fly motor physiology
or the calibrated AdEx model from that paper.

The changed dynamics receives its own fixed, neutral-gray intrinsic calibration
using the existing all-nonsensory-cell homeostasis rule. No game image, reward
or motor identity is used to fit the biases. Consequently the gameplay
comparison tests adaptation PLUS its neutral calibration, not a pure one-factor
adaptation effect. The decoder/rewards remain those of serial-v1.

Generate its optional local calibration without overwriting existing files:

```powershell
.\.venv\Scripts\python.exe scripts/probe_intrinsic.py --device cuda `
  --calibration-only --motor-adaptation-increment 0.02 `
  --motor-adaptation-seconds 3 --reset-probe-decisions 128 `
  --export fly-data/intrinsic-neutral-adaptive-v1.npz
```

Three independent neutral probes keep every mapped motor population active
(roughly 0.91..1.12 Hz); this is a sanity check, not an action-frequency target.
The adaptation array is saved/reset explicitly and restricted to its anatomical
population on restore. Missing settings mean exactly zero adaptation, so old
models/checkpoints are not silently changed. Gameplay and learning benefits
are still being evaluated; this is not a new default.

## Oracle capacity controls are not trained brains

`probe_synaptic_capacity.py` is deliberately a separate supervised positive
control. It temporarily fits existing DNa02 inputs using fixed-source neural
counts, tests the resulting REAL recurrent activity and fixed decoder on new
noise streams, then discards the weights. It does not use the game or export a
brain. Positive-only and both-sign controls keep sign-preserving factor bounds,
local input budgets and predicted two-cue mean input fixed. These controls
can expose model limitations but must NEVER be reported as reward learning or
silently used to initialize Pokemon. Neither a fitted readout nor classifier
enters the application.

## Optional one-hop premotor learning scope

`sensorimotor-premotor-v1` keeps the perturb-v3 rule, neutral calibration,
original neural dynamics, pixel input and fixed decoder. Its explicit
`premotor-one-hop-v2` scope adds existing positive inputs to central-brain
intrinsic cells that directly project to any descending neuron. Selection uses
the connectome and superclass labels, not the button registry or game state.
Original descending/motor targets remain included. No new edges are added;
signs, factor bounds and per-cell input budgets stay fixed.

This expands the targets from 2,129 to 21,848 cells (4,300,611 positive edges).
It tests whether restricting plasticity to the last motor stage was limiting
learned sensory control. This is an engineering hypothesis, NOT an established
biological map of plastic synapses. Inactive learning must produce exactly the
same neural activity/actions as perturb-v3. Missing scope fields keep the old
`motor-inputs-v1` selection, including in checkpoints. Synthetic assay weights
are not game initialization. No learning benefit is assumed from broader scope.

## Display interpretation (kept out of the showcase layout)

- In snapshot mode the game frame is the exact pre-action image the fly received;
  in temporal modes it is the last image of the preceding input window. The
  displayed pulse follows that input. RAM is measurement/reward input only.
- The retina is a display-precision copy of the fixed grayscale/bilinear input.
- Amber dots are actual spikes within the sample. Their short glow is a visual
  persistence effect, not extra firing. Motor highlights use exact firing IDs.
- Blue dots in hybrid mode show normalized graded visual state at the last
  neural step. They are not spikes or firing rates. Population means include
  unpositioned neurons even when those neurons cannot be plotted.
- At most 6,000 positioned spiking cells are plotted per sample. Anatomical
  coordinates exist for 140,638 of 166,700 model cells. No positions, synapses,
  action proposals, or activity are fabricated for presentation.
- Synthetic D means the bounded reward modulator, not measured dopamine.
  Changed-edge counts establish implementation activity, not learned behavior.
  For the compartment candidate, hover details separately identify internal
  prediction errors; these must not be confused with a new game penalty.
- The dashboard is a local desktop showcase; human buttons are disabled during
  autonomous runs. Observation/manual diagnostics remain distinctly labeled.

Checkpoints include adaptive currents and continuous release as well as all
baseline state, learning traces, gates, weights, and random-stream state. Exact
same-backend resume and fractional sparse propagation are regression tested.
Browser layout verification remains unavailable when the in-app browser cannot
connect; DOM tests and live HTTP/SSE checks are not a substitute for visual QA.

## New opt-in mechanism tests (2026-09-18 13:23 UTC; not promoted)

`sensorimotor-score-projected-v5` keeps the score-v2 stochastic circuit and
original likelihood eligibility. Before a reward update, it projects the
preconditioned update against estimated mean incoming current, independently
for every anatomically selected target. The estimate is a 10-second running
presynaptic release mean. It has no cue labels, button identity, desired firing
rate, external critic or learned adapter. This is a local engineering
constraint, not a measured fly learning rule. Later clipping/resource
competition can make exact mean preservation inexact. Existing factors,
signs, sparse connections and 25% input budgets remain unchanged.

`sensorimotor-gain12-v1` changes whole-circuit synaptic gain from 3 to 12 and
requires its own frozen neutral-gray calibration. It is a signal-to-noise
sensitivity test, not proof that stronger coupling is biologically correct.
All cells share the gain. Calibration uses the existing uniform 1 Hz equation
on eligible non-sensory spiking cells, with no game, reward or motor labels:

```powershell
.venv\Scripts\python.exe scripts/probe_intrinsic.py --device cuda --calibration-only --synaptic-gain 12 --calibration-steps 10000 --reset-probe-decisions 256 --export fly-data/intrinsic-neutral-gain12-v1.npz
```

These two hypotheses are evaluated separately. Default gain remains 3;
historical checkpoints missing the field retain that exact value. Neither
candidate is the default launcher profile. Synthetic assay-trained weights
must never be loaded into Pokemon. See the preselected comparisons and all
negative evidence in `IMPROVEMENT_PLAN.md` / `FOLLOWTHROUGH_RESULTS.md`.

## User-authorized internal visual calibration (opt-in, not promoted)

`visual-rate-v1.json` transfers frozen, licensed visual-neuron parameters from
the pinned MaleCNS/flyvis reference described in `THIRD_PARTY_NOTICES.md`. It
does NOT import the reference's movement controller, virtual receptors, image
features or learned output policy. The unchanged raw-pixel retina drives the
same 6,006 original photoreceptors. 71,080 existing visual cells use rectified
rate dynamics with 4 ms substeps. All original selected-cell edges and signs
remain present; reference-covered magnitudes and type time constants/biases
are calibrated. Remaining selected-cell edges retain their original gain-3
magnitudes. Real R1-6 -> lamina conductances are normalized to the reference's
aggregate strength; there is no direct image-to-lamina injection.

Original connections across the visual/nonvisual boundary remain active in
both directions. A fixed, uniform conversion maps reference rates [0,5] to
normalized graded state [0,1], keeping the prior maximum release (0.25).
Boundary inputs use the original signed current, held for one 20 ms outer
step. This mixed-unit transfer is an explicit engineering hypothesis, not a
validated biological model. Original data files are not rewritten. The
standalone assay has zero outside input, explicitly unlike the complete fly.

The remaining spiking neurons receive a newly frozen uniform neutral-gray
calibration, never game/button/reward fitting. Only existing downstream
synapses undergo gameplay learning. Perturbation eligibility uses measured
continuous presynaptic release where appropriate, with prior-history timing;
graded cells do not fabricate spikes. Visual voltage, source fingerprints and
all downstream state are checkpointed. Old profiles default to `legacy-v1`
visual dynamics and remain unchanged. The fixed anatomical button decoder is
unchanged. Streaming pixels and serial delivery are explicit in this profile.

Preparation (data require network; subsequent calibration is ROM-free):

```powershell
.venv\Scripts\python.exe scripts/fetch_visual_reference.py
.venv\Scripts\python.exe scripts/audit_visual_reference.py
.venv\Scripts\python.exe scripts/probe_calibrated_vision.py
.venv\Scripts\python.exe scripts/probe_intrinsic.py --device cuda --visual-model calibrated-rate-v1 --calibration-only --calibration-steps 10000 --reset-probe-decisions 256 --export fly-data/intrinsic-neutral-visual-rate-v1.npz
```

Calibration exports are immutable. The optional `--parameters flyvis` assay
keeps zero-strength reference edges at their original magnitudes to preserve
connectivity; it is therefore NOT an exact flyvis reproduction. Neither model
has yet established reward-specific visual learning or reliable progression.
Synthetic assay/oracle weights are never gameplay initialization.

Exact resume also repairs a PyBoy 2.7 rendering omission: its window-line
counter is not serialized. Before attaching reward observers, a temporary
render is discarded by reloading the identical game bytes. The serialized
state is checked unchanged; no extra gameplay frame or reward reaches the
fly. This fixes a one-image discrepancy when resuming an open menu.
