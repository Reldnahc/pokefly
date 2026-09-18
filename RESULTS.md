# Pokefly evaluation: playing, not mastering

Current follow-through: [FOLLOWTHROUGH_RESULTS.md](FOLLOWTHROUGH_RESULTS.md).
Movement calibration now fixes the severe Up imbalance in tested runs, and
bounded internal motor learning passes retained preference/reversal controls.
Actual learning runs obtained starters; one won the first rival battle and
entered Route 1. Screen-specific and repeatably useful gameplay learning remain
under investigation. The historical negative comparison below is preserved,
not the current overall conclusion.

## Independent neural/learning experiments — September 17, 2026

Protocol recorded before implementation in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).
Local artifacts use September 18 UTC timestamps. The user's stopped run, root ROM
and original model are protected by a hash manifest. No route reward, Up bonus,
loop penalty, forced gameplay action, external learned policy or RAM action input
was introduced. These experiments assess participation and learning, not mastery.

### Current conclusion

Fresh-frame input, isolated candidate profiles, retained-weight loading, and
controlled assays are implemented. None of the completed forward/vision screens
supports changing the normal default. Saved weights have a measurable gameplay
effect, but a repeatable benefit has not been shown. The learning-rule comparison
is complete: neither rule establishes reliable learned choices or reversal.
All candidates remain opt-in and the normal default is unchanged.

### Preserved user run and retained learning

The user stopped `internal-learn-20260918T011254Z-25ffd6` cleanly at decision
68,811. It reached the first floor at 19,999 and Pallet Town at 20,053; those
destinations received only the existing generic novelty rewards. It recorded
171 new-tile and two new-area events, total reward 10.55, no battles or captures.
Up occurred on 2,901 decisions (4.22%); Down on 18,396 (26.73%).

Its checkpoint retains 1,872 changed edges among 4,407 eligible connections;
the maximum change is about 1.98% of an edge's original weight. This is real
saved internal plasticity, not evidence by itself that the route was learned.

Matched retained-versus-original comparison: same outdoor emulator state,
frozen weights, seeds 101/102/103, fresh neural dynamics, 1,000 decisions per run.
Baseline runs are shared with the temporal comparison below, not extra trials.

| Seed | Original / retained distinct positions | Original / retained Up pulses | Different actions / 1,000 |
| --- | --- | --- | ---: |
| 101 | 11 / 14 | 35 / 41 | 580 |
| 102 | 15 / 13 | 38 / 44 | 559 |
| 103 | 13 / 12 | 49 / 50 | 359 |

The small synaptic changes do affect a closed-loop trajectory. They do not
consistently improve exploration; the retained trials spend 86.0–88.0% of logged
decisions in the bottom two town rows and never enter another map or battle.
Later divergent screens amplify early differences, so hundreds of changed
actions do not mean hundreds of independently learned decisions. The trained
checkpoint came from one history, not three independent training replicates.

Exact legacy continuation also passes: replaying decisions 68,501–68,520 from
the user's older generation reproduces every trajectory field except run ID
and wall-clock timing. Both original checkpoint generations remain untouched.

### Does continuous raw vision solve forward access?

Eighteen frozen trials: two identical starting states (bedroom and the stopped
outdoor state), three seeds, three input timings, 1,000 decisions per condition.
No parameter was fitted to a game route. Snapshot is the existing one-image
window; stream observes fresh frames during the pulse; endpoint is its matched
budget/reward-ordering control. See [MODEL_VARIANTS.md](MODEL_VARIANTS.md).

| Start, seed | Snapshot Up / distinct positions | Stream Up / distinct positions |
| --- | --- | --- |
| Bedroom, 101 | 51 / 75 | 39 / 27 |
| Bedroom, 102 | 49 / 28 | 44 / 32 |
| Bedroom, 103 | 48 / 18 | 52 / 23 |
| Town, 101 | 35 / 11 | 30 / 11 |
| Town, 102 | 38 / 15 | 44 / 52 |
| Town, 103 | 49 / 13 | 51 / 15 |

Snapshot and endpoint produce identical action sequences in all six matched
frozen comparisons. Stream gets genuinely changing input within 859–991 of
1,000 windows, rather than repeating a framebuffer. It does **not** consistently
raise Up: 270/6,000 decisions (4.50%) for snapshot versus 260/6,000 (4.33%) for
stream. Exploration varies in both directions. One snapshot/endpoint bedroom
seed exits the house with frozen original weights; no stream seed does in this
budget. House exit can therefore occur without learning and is not evidence
that a learned route exists. No autonomous battle/capture occurred in these trials.

Stream passes exact-resume and actual pixel-display checks, but remains opt-in.
No held-out confirmation or combined-profile search was triggered by this
negative forward-access screen. These results do not rule out longer-term
benefits from richer visual dynamics; they do not demonstrate one either.

### What still suppresses Up?

The circuit audit contains 39 frozen conditions: three seeds, 1,536 neural steps
per condition, with current/voltage statistics discarding the first 384 steps.
Inputs include black, the stopped town screen, left/right half-images and a
moving grating. All four directions are measured. Diagnostic interventions are
separate from Pokemon and never become normal actions or weights.

On the town screen, seed 101, the two forward cells receive about +0.024/+0.024
excitatory increments versus -0.040/-0.041 inhibitory increments per update.
Mean post-reset voltage is 0.696/0.678 against threshold 1. CB0890 supplies the
largest typed inhibitory component in this case (about 19%), with several other
contributors. This does **not** establish runaway firing of one culprit: the
estimated contribution-weighted CB0890 rate is about 2.55 Hz, not the 50 Hz
timestep ceiling. The residual problem is distributed modeled excitation and
inhibition, whose physiological calibration remains unresolved.

| Town-screen condition | Up choices / 128 windows, seeds 101 / 102 / 103 |
| --- | --- |
| Original weights, current isolated profile | 4 / 7 / 9 |
| User's retained weights | 4 / 7 / 9 |
| Hold unused sensory cells completely quiescent | 3 / 5 / 7 |
| Remove all negative current into Up: diagnostic only | 43 / 39 / 40 |
| Multiply all eligible KC-to-MBON weights by four: capacity control only | 3 / 8 / 9 |

The retained weights leave every motor-count window and choice identical in
this particular fixed-image probe (384 decisions); they do not have zero effect
everywhere, as the dynamic gameplay comparison shows. Removing all Up inhibition
demonstrates access to the circuit, not a justified repair. Valid braking should
not be cut solely to get a desired key.

Completely quieting the unused sensory inputs barely changes the remaining Up
current balance and does not improve choices. Left-versus-right image sensitivity
remains: 84/87/81 of 128 actions differ with the current profile and 78/83/79 with
quiescence under matched noise. This establishes pixel sensitivity, not useful
visual recognition. `quiescent-v1` stays opt-in and was not advanced to gameplay
optimization or combined with stream to search for lucky navigation.

### Controlled internal-learning comparison

The final authoritative assay is in `learning-v2/` under the research directory
below. Both `dan-targeted-v1` and `compartment-ema-v1` are tested with three seeds,
two counterbalanced cue-to-direction assignments, paired, within-cue reward-
permuted and frozen arms: 36 arms total. Each arm has 256 acquisition and 256
reversal decisions. Retention uses 192 reward-free decisions across two cues and
two fresh noise streams, with fresh neural dynamics. The first 48-decision
choice-count probe is repeated after on-disk reload. This assay checks matching
action histograms; the separate live-resume checks compare full trajectories and
neural arrays. Only existing internal synapses can change.

The synthetic reward is for an actual fixed-decoder Left/Right choice **only in
this separate ROM-free assay**. No button is forced and none of this reward logic
is installed in Pokemon. Permuting within each cue preserves its reward amount
while breaking alignment with the selected command. A five-percentage-point
all-case improvement screen was specified before results; this is exploratory,
not a statistical significance test. Independent confirmation seeds 201–203 are
reserved for a candidate that passes the screen, not used to rescue failures.

All 36 arms completed. Neither rule passes the acquisition-and-reversal screen
in any of its six seed/mapping cases. The descriptive differences below are
percentage points in target-button rate among **all** test decisions, not
percent improvement or independent per-decision statistical estimates:

| Rule | Acquisition vs pretest, mean | Acquisition vs unpaired, mean | Reversal vs pre-reversal, mean | Passing cases |
| --- | ---: | ---: | ---: | ---: |
| Current DAN-targeted rule | +0.09 pp | +0.52 pp | -0.09 pp | 0 / 6 |
| Partial compartment / EMA candidate | +0.09 pp | -0.17 pp | 0.00 pp | 0 / 6 |

Acquisition changes span -1.04 to +1.56 points relative to pretest for both
rules, rather than a consistent target-specific shift. After reversal, neither
improves the new target rate over the unpaired control in any case. The candidate
has exactly the frozen control's complete per-cue action histograms in both
stages for seeds 101 and 102; seed 103 shows small effects, not reliable reversal.

Plasticity itself runs: paired arms finish with maximum relative edge changes
of 8.31–9.88% for the current rule and 4.62–5.38% for the candidate. All 72
saved stage checkpoints reproduce their first probe's action histogram, and
all frozen weights stay original. A weight effect and persistence are therefore
working; this tested rule/circuit/decoder combination does not show useful
learned choice specificity in this budget. That does not prove a biological fly
cannot learn, or that every possible internal rule must fail.

Neither rule advances to independent confirmation seeds or a new Pokemon
learning claim. No default is promoted and no reward is increased to mask the
negative result. The next justified research target is visual-memory-to-motor
signal transfer and neural dynamics calibration, with a simple retained-choice
test as a prerequisite to claiming improved Pokemon learning. Longer gameplay
alone has not been shown to fix the weak forward recruitment or credit assignment.

Snapshot API hardening now gives each snapshot its own plasticity arrays. The
assay was conservatively restarted after finding shared references; review then
showed that its pre-arm restore already detached the baseline, so actual arm
contamination was not established. Superseded partial artifacts are archived,
not used as final evidence. Each new arm asserts original starting weights.

### Verification, artifacts and reproduction

159 Python tests, 20 JavaScript tests, lint, formatting and all 15 live ROM/CUDA
checks pass. They cover baseline and every candidate: exact resume, unchanged
frozen/no-reward weights, pixel-to-retina display agreement, temporal-window
image hashes, combined-button pulses and live pacing. These candidate checks are
now part of the repeatable `scripts/test.ps1 -Live` suite.
The launcher's new `-Weights ... -Intro` path also ran successfully with a
bounded two-decision frozen check. Browser-rendered visual QA remains unavailable;
logic and live HTTP tests are not a substitute for it.

All 10 protected manifest entries still match after the final tests; the ROM
remains Git-ignored. All owned experiments and short-lived servers have exited,
with no matching Python process remaining. No original weights or saved user run
were overwritten. Defaults and game reward rules remain unchanged.

Research directory: `runs/neural-improvement-20260918T020816Z-3605f1/`.

- `protected.json` and `protocol.json`: immutable-source hashes and initial plan.
- `temporal-gameplay.json`: all 18 frozen trials and individual run paths.
- `motor-audit.json`: all 39 circuit conditions, currents and action windows.
- `retained-gameplay.json`: saved-weight gameplay and old-checkpoint continuation.
- `learning-v2/learning-choices.json`: authoritative learning arms and comparisons.
- `verification.json`: consolidated checks, all 15 live artifact paths and result hashes.
- Expanded full verification: `runs/verify-159488a31b694a808c26e197d8c06829`.
- Stream exact resume: `runs/internal-smoke-20260918T025142Z-6fe622/report.json`.
- Endpoint exact resume: `runs/internal-smoke-20260918T025159Z-29cdd5/report.json`.
- Compartment exact resume: `runs/internal-smoke-20260918T025230Z-30c13d/report.json`.
- Quiescent exact resume: `runs/internal-smoke-20260918T025215Z-701eed/report.json`.
- Weights launcher: `runs/internal-frozen-20260918T024219Z-0d1f8d/config.json`.

For independent reproduction, create a new ignored output directory and use
the existing stopped run as read-only input; do not overwrite the evidence:

```powershell
New-Item -ItemType Directory runs/research-repeat
.\.venv\Scripts\python.exe scripts/evaluate_temporal.py `
  --source-run runs/internal-learn-20260918T011254Z-25ffd6 --output runs/research-repeat
.\.venv\Scripts\python.exe scripts/audit_motor_learning.py `
  --source-run runs/internal-learn-20260918T011254Z-25ffd6 --output runs/research-repeat
.\.venv\Scripts\python.exe scripts/evaluate_retained.py `
  --source-run runs/internal-learn-20260918T011254Z-25ffd6 --output runs/research-repeat
.\.venv\Scripts\python.exe scripts/evaluate_learning_choices.py --output runs/research-repeat
```

## Earlier evaluations (preserved)

## Persistent sensory-isolated experiment — September 17, 2026

The user selected the tested sensory isolation as a persistent experimental
setting. `configs/sensory-isolated-v1.json` enables it, and `scripts/start.ps1`
now defaults to that profile. The unisolated hybrid and original baseline remain
selectable controls; bare `train` without a config still uses baseline dynamics.

This gates incoming network current to 11,931 nonvisual sensory cells selected
only by anatomical metadata, excluding every original visual input cell. It
does not delete neurons or stored synapses, suppress outgoing sensory signals,
change tonic input/noise, boost Up, change rewards, or feed RAM into actions.
The setting and selected-cell checksum are recorded in checkpoint identity and
display metadata. Old checkpoints retain isolation **off**.

### What the new tests show

Same bedroom state, seeds 64/65/66, frozen weights, 1,000 decisions per run.
Only the sensory-current gate differs; both use the corrected parallel decoder.

| Measurement | Unisolated hybrid | Sensory-isolated hybrid |
| --- | ---: | ---: |
| Up pulses by seed | 2, 3, 4 | 45, 48, 43 |
| Forward-neuron spikes by seed | 4, 5, 9 | 89, 83, 92 |
| Distinct tiles by seed | 12, 11, 11 | 32, 27, 32 |

The larger Up signal comes from changed internal neural dynamics, not adjusted
motor thresholds. All runs remain in the bedroom; no house exit, battle, or
capture occurs. These frozen trials establish improved access to forward input,
not purposeful navigation or learning. Isolated sensory cells still average
about 0.28 Hz from the unchanged tonic/noise process.

Two-seed static-image checks retain exact black-repeat trajectories and changed
actions for every left/right and top/bottom comparison. All four spatial pairs
still affect visual-KC candidates after startup. However, affected visual-KC
counts drop from 173–263 to 29–110, and black-screen mean KC activity drops from
1.88–1.91 Hz to 0.75–0.76 Hz. The intervention changes broader network state;
it is not a selective Up repair or established improvement in visual memory.
Association/retention efficacy needs separate reevaluation under this profile.

### Verification and evidence

- The permanent gate exactly matches the prior temporary diagnostic over 384
  neural steps: all spikes, final voltages, graded release and adaptation agree.
  No motor readout population is included in the isolated sensory mask.
- The isolated model passes exact 20-decision CUDA resume, retained-weight
  current checks, and fractional propagation against an independently masked
  SciPy reference. Frozen/no-reward weights stay unchanged.
- A real pre-isolation checkpoint still reproduces its recorded 20-decision
  continuation and all final neural arrays with isolation disabled.
- The actual HTTP/SSE stream labels isolation and preserves pixel/retina
  agreement and combined-button logs. A real launcher test confirms the new
  default. Browser-rendered layout verification remains unverified.
- 118 Python tests, 15 JavaScript tests, formatting/lint and all seven live
  ROM/CUDA smoke checks pass. ROM/connectome hashes remain unchanged.

Local artifacts (ignored by Git):

- Matched gameplay, visual checks, and diagnostic-to-production equivalence:
  `runs/sensory-isolation-evaluation-20260918T004607Z-913fe1/report.json`.
- Isolated exact resume: `runs/internal-smoke-20260918T004510Z-7a0b80/report.json`.
- Isolated stream: `runs/internal-learn-20260918T004504Z-1f0687/smoke.json`.
- Older checkpoint compatibility:
  `runs/internal-learn-20260918T004848Z-4cf015/pre-isolation-resume-verification.json`.
- Default launcher: `runs/internal-learn-20260918T004850Z-8c9ad7/config.json`.
- Full validation: `runs/verify-bc5528a4d46d466686bec6bcb107aa17`.

Reproduce the matched experiment with:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_sensory_isolation.py --device cuda `
  --load-state runs/internal-learn-20260917T225448Z-83e5fb/start.state
```

## Button fix and forward-circuit diagnosis — September 17, 2026

Historical audit below preceded the user's decision to persist sensory isolation.

The user approved the button correction and requested investigation of weak
forward activity. **Only the fixed button adapter changed in normal gameplay.**
No new reward, Up gain, threshold, motor mapping, neural current, or connectome
change was applied. Neural interventions below are isolated diagnostics.

### Corrected adapter: Up now reaches the game

The old global winner discarded every valid Up signal in the previous hybrid
learning runs: 22 forward spikes across 3,000 decisions, but zero Up pulses.
These neurons were rarely active, not completely silent. Eleven recorded
windows had Up as the strongest active direction but lost to A/B globally.

`parallel-v2` independently chooses one direction and one A/B/Start command.
It preserves per-cell rate normalization, thresholds, traces, silent-population
gating and Start cooldown; ties rotate within each channel. Every pressed key
is released after the same pulse. The display highlights both delivered keys.
Old checkpoints retain `exclusive-v1`; a real pre-fix hybrid checkpoint reproduced
all 20 subsequent decisions and every final neural array exactly.

New matched test: frozen weights, same bedroom state, seeds 64/65/66, 1,000
decisions each. All neural/reward settings are identical between conditions.

| Measurement | Old exclusive decoder | Corrected parallel decoder |
| --- | ---: | ---: |
| Up pulses by seed | 0, 0, 0 | 2, 3, 4 |
| Total forward-neuron spikes | 20 | 18 |
| Distinct tiles by seed | 8, 8, 6 | 12, 11, 11 |
| Position transitions, summed | 826 | 884 |

The changed game trajectory feeds different screens back to the fly, so the
neural spike totals need not match. This is recovered access to a button, not
increased forward firing or demonstrated learning. Up remains rare; no battles,
captures or house exits occur. The extra tiles are measurements, not the goal.

An isolated emulator test moves to clear floor and lets each pulse settle.
Up, Up+A, and Up+B all move from `(map 38, x5, y6)` to `(38,5,4)`; Up+Start
leaves the position unchanged because Start opens the menu. The original
starting tile is directly below furniture, so blocked movement there is not
evidence of a broken key. These human-specified diagnostic pulses never enter
the autonomous trials.

### Why forward neural firing is weak

Frozen static-image audit: 768 neural steps (15.36 seconds), three seeds;
voltage/current means discard the first 128 steps. For hybrid bedroom seed 64:

- The two DNg100 forward cells average approximately 0.564/0.551 voltage,
  against firing threshold 1. Mean excitatory increments are +0.028/+0.028 per
  step, inhibitory increments -0.071/-0.073: about 2.5 times as large.
- DNge054 supplies the largest single typed inhibitory contribution, about
  21–22% of total inhibition. Its two cells fire at 9.8/16.3 Hz after startup.
  Their firing and Up's suppression persist with black and white input.
- Excitatory drive into DNge054 includes nonvisual bristle populations and
  AL-AST1. The latter is pinned at the 50 Hz timestep ceiling. With **no external
  touch or smell input**, 745 `BM_InOm` cells average 23.8 Hz, and 2,635 `ORN_`
  cells average 32.3 Hz. Black-screen values are almost the same.

`BM_InOm` identifies eye-bristle mechanosensory cells, not photoreceptors
([curated anatomy](https://www.virtualflybrain.org/term/bm-inom-fbbt_00047346/)).
The primary MaleCNS group's [DNge054 input/output atlas](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/DNge054.html)
independently lists bristle/AL-AST1 inputs and DNg100 outputs. Signed current
measurements above come from our local model, not inferred firing in that atlas.

The installed upstream `FlyBrain` uses incoming synapses on sensory cells by
default (`sensory_input=True`). Its own source warns of runaway olfactory
receptor recurrence in that setting. Our normal controller retained it. Thus
the current model is not a sensory blank slate outside vision: internally
generated nonvisual activity can recruit inhibitory pathways.

### Causal checks, not production fixes

Same static bedroom and seeds, fresh state per condition, no learning:

| Isolated diagnostic | Forward spikes by seed |
| --- | ---: |
| Unaltered hybrid | 1, 0, 0 |
| Remove only DNge054's negative input into Up | 3, 6, 2 |
| Remove all negative input into Up | 53, 55, 45 |
| Zero incoming network current to `BM_` sensory cells | 4, 5, 1 |
| Zero incoming network current to all nonvisual sensory cells | 7, 10, 6 |

In the last condition, photoreceptor input, original weights, tonic input and
neural noise remain unchanged. Nonvisual sensory firing falls to about 0.28 Hz;
DNge054 falls to 2.3/1.7 Hz in seed 64, and Up's mean inhibitory input falls
from about -0.071/-0.073 to -0.038/-0.039. Up still fires weakly. Disabling noise
alone does not restore Up, and removing tonic drive from just DNge054 helps
only slightly. The issue is distributed network dynamics, not one broken cell.

Interpretation: sensory recurrence is a demonstrated contributor to excessive
forward suppression in this simulator. It is **not** established that all those
biological synapses should be removed, that DNge054 is inherently inappropriate,
or that more spontaneous Up activity constitutes useful vision. Primary work
shows inhibitory halting pathways can legitimately suppress forward-driving
BDN2/DNg100 activity ([Sapkal et al., 2024](https://www.nature.com/articles/s41586-024-07854-7),
[DNg100 identity](https://www.virtualflybrain.org/term/dng100-fbbt_20007473/)).

The next modeling question is how to represent unused sensory modalities and
their recurrent/axon inputs without artificial saturation, while retaining
screen-sensitive walking recruitment. Any candidate needs blank-screen, visual
motion and matched-play controls before promotion. No such change is enabled
in the live controller, and no action-balancing or route reward was introduced.

### Reproduction and evidence

```powershell
.\.venv\Scripts\python.exe scripts/compare_motor_decoders.py --device cuda `
  --load-state runs/internal-learn-20260917T225448Z-83e5fb/start.state
.\.venv\Scripts\python.exe -m pokefly.forward_probe --device cuda `
  --frame runs/internal-learn-20260917T225448Z-83e5fb/start.png
# Add --sensory-followup for the separately labeled sensory-current controls.
```

- Matched gameplay and actual emulator pulses:
  `runs/motor-comparison-20260918T002323Z-791ffe/report.json`.
- Forward voltages, signed currents and causal controls:
  `runs/forward-audit-20260918T002623Z-80143c/report.json`.
- Sensory recurrence follow-up, including saturation measurements:
  `runs/forward-sensory-audit-20260918T003029Z-08fc77/report.json`.
- Pre-fix checkpoint compatibility:
  `runs/internal-learn-20260918T002833Z-0c4a68/legacy-resume-verification.json`.
- New baseline/hybrid exact resume:
  `runs/internal-smoke-20260918T002226Z-e19115/report.json` and
  `runs/internal-smoke-20260918T002246Z-2a40d6/report.json`.
- Final full verification: `runs/verify-4ced3d9a15ba4ae7a9d1b0a67c4104fb`;
  114 Python tests, 14 JavaScript tests, formatting/lint, and all five live
  ROM/CUDA checks pass, including combined-button stream/log consistency.

The ROM remains unchanged and Git-ignored. Both neural audits verify hashes of
the original connectome files after testing. Display logic tests recognize
combined delivered buttons; actual browser visual verification remains unverified.

## Hybrid fixes — September 17, 2026

Historical results below used the original `exclusive-v1` button decoder.

Goal: a pixel-driven fly participating in Pokemon, with learning inside the
neural model. No completion requirement, walkthrough, forced button, or new
game reward was added. The original baseline remains selectable.

### What improved

- Sustained KC saturation is eliminated in the tested hybrid configuration:
  black-screen post-startup means are 1.91/1.88 Hz for seeds 64/65, versus
  essentially 50 Hz in the baseline. No KC exceeds 45 Hz in these hybrid tests.
  About 3.8% of KCs are active on an average neural step, not all cells at once.
- Pixel changes still affect downstream activity and actions in all four
  spatial comparisons. The relay now represents early visual activity as graded
  signals, not fabricated spikes. This is not proof of biologically useful vision.
  After discarding startup, 173–263 of 335 visual-KC candidates change spike
  timing across hybrid spatial pairs; the saturated baseline changes none.
- All seven fixed readouts pass direct-neuron activation controls. The decoder
  does not need artificially balanced thresholds to expose Up or B.
- Hybrid checkpoints reproduce a 20-decision continuation exactly, including
  adaptive current and continuous release. Fractional GPU propagation agrees
  with an independent SciPy sparse reference. Frozen/no-reward weights stay fixed.
- The actual hybrid HTTP/SSE stream preserves frame-to-retina agreement, sends
  graded activity separately from spikes, and refuses human button injection.
  The supplied ROM hash is unchanged.

### Gameplay: more movement, not better exploration

Same bedroom state and seeds 64/65/66, 1,000 decisions per trial; values below
compare learning-enabled runs. All pulses are decoded from neural activity.
Tile transitions count changes between consecutive logged positions, including
revisits; they are **not** evidence of progress or intentional navigation.

| Measurement | Preserved baseline | Hybrid |
| --- | ---: | ---: |
| Mean KC firing during game | 49.93 Hz | 1.91 Hz |
| B pulses / 3,000 decisions | 12 | 213 |
| Right pulses / 3,000 decisions | 105 | 439 |
| Up pulses / 3,000 decisions | 6 | 0 |
| Tile transitions, summed | 217 | 827 |
| Distinct tiles per trial | 11, 13, 8 | 8, 8, 6 |

Hybrid learning, frozen, and no-reward conditions all visit 8/8/6 distinct tiles.
The latter two produce identical action trajectories. Learning changes some
actions in two seeds, but does not improve exploration in this test. The hybrid
uses six of seven buttons, waits less, and moves repeatedly around a smaller
part of the room. No natural battle or capture occurs. House exit, a secondary
measurement rather than a reward, remains zero in all conditions.

### Image association: still not working reliably

A separate ROM-free test uses raw left/right images, three seeds, 12 reward
pairings, frozen and balanced nonpredictive-reward controls, reward-free tests,
and reversal. Only existing visual-KC synapses learn; no classifier or decoder
is fitted to score the task.

Weights persist through resets, and repeated reward-free measurements are exact.
But rewarded-image-specific suppression does not consistently emerge: two of
three seeds suppress the unpaired image more, and reversal does not reverse that
ordering in any seed. MBON test spike counts are unchanged in two seeds; the
third changes by only three spikes for the unpaired image, also seen in the
balanced control. These are negative/weak results, not a learned association.

The DAN-targeted rule is a coarse anatomical-overlap proxy. True compartment
annotations, receptor-specific dopamine dynamics, useful image separation, and
screen-driven access to forward movement remain unresolved. The next scientific
work should target those gaps, not add route rewards or demand game mastery.

### Evidence

Ignored local artifacts are preserved, including configs, seeds and trajectories:

- Neural ablations, including post-startup comparisons:
  `runs/circuit-evaluation-20260917T235145Z-4808db/report.json`.
- Image association: `runs/visual-association-20260917T234152Z-98af1f/report.json`.
- Matched gameplay: `runs/evaluation-20260917T234336Z-0dbd1a/report.json` and
  `trials.json`, which point to the individual complete action logs.
- Exact hybrid resume: `runs/internal-smoke-20260917T235527Z-574245/report.json`.
- Hybrid stream: `runs/internal-learn-20260917T235522Z-a862cb/smoke.json`.
- Final verification: `runs/verify-498d20c620204506be95a052ea0039cd`;
  90 Python tests, 13 JavaScript tests, lint, and all five real-ROM/GPU smoke
  checks passed (manual/baseline/hybrid streams and baseline/hybrid resume).

The desktop UI now uses a viewport-height two-column showcase with compact
labels, screen/retina, delivered controls, neural activity, and metrics. Long
prose has been removed from the visible layout; the UI agent finished and stopped.
Actual browser
layout verification remains blocked: the browser skill reports the in-app
browser unavailable after retries. DOM/canvas and HTTP tests do not prove layout.
Model equations and display caveats are in [MODEL_VARIANTS.md](MODEL_VARIANTS.md).

## Earlier baseline evaluation (preserved)

September 17, 2026. RTX 4070 Ti; validated local English Red ROM. These are
negative/limited research results, not a claim that the fly has learned Pokemon.

## Controlled gameplay

Same bedroom emulator state, seeds 64/65/66, 1,000 neural decisions per trial.
All adapters and provisional general rewards were held fixed. No house-exit
bonus or gameplay script was used; the common starting state came from the
explicit intro-setup intervention. Every decoded button came from fly activity.

| Condition | House exits | Mean distinct tiles | Battle wins | Captures |
| --- | ---: | ---: | ---: | ---: |
| Internal learning | 0/3 | 10.67 | 0 | 0 |
| Frozen connections | 0/3 | 9.00 | 0 | 0 |
| Absent reward | 0/3 | 9.00 | 0 | 0 |
| Retained weights, frozen | 0/3 | 11.33 | 0 | 0 |

The retained checkpoint had only 70 training decisions on seed 64; one test seed
therefore overlaps training. This is a small integration/retention comparison,
not held-out statistical evidence. Tile differences alone do not demonstrate
learning or reliable control. The agreed first milestone has **not** been met.

Raw evidence: `runs/evaluation-20260917T225520Z-e5fb1f/report.json` and
`trials.json`, which link each preserved run and complete action/reward log.
These run artifacts are intentionally ignored by Git.

## Circuit capacity

Two matched-noise seeds, 384 neural steps per stimulus. Black-repeat traces
match exactly. All four left/right and top/bottom comparisons changed some
decoded actions. A separately labeled fourfold-KC→MBON-weight perturbation
also changed actions, demonstrating causal circuit access, not learned skill.

Every Kenyon cell averaged above 45 Hz; population mean was about 47.8 Hz,
near the 50 Hz timestep ceiling. Up and B each had a spatial pair with no
changed spikes. A/Left dominated gameplay output, while Up/B were rare.
Homogeneous upstream neural dynamics and motor excitability are substantial
limitations; more training alone has not been shown to resolve them.

Evidence: `runs/internal-probe-20260917T225804Z-14d192/report.json`.

## Engineering checks

### Live pacing and persistence checks (September 17, 2026)

Normal training/launcher defaults are now unlimited until Ctrl+C. The dashboard
can change the wall-clock cap during a run (including Max/uncapped); the requested
cap and measured actual speed are distinct. Neural equations/timesteps, frames
per decision, rewards, and button pulses are unchanged. Pacing/timing fields are
excluded from exact-resume comparisons, but actions, neural arrays, and rewards
still must match exactly.

Validation: 144 Python tests, 18 DOM/canvas UI tests, and all seven live GPU/ROM
checks passed. Live controls changed from 2.5 decisions/s to uncapped in each
autonomous profile without enabling human buttons. A 20-decision continuation
at a different wall-clock cap matched the original neural state and outcomes.
An additional default-unlimited continuation stopped after a test-generated
SIGINT and saved at a consistent decision boundary, preserving the source
checkpoint. The PowerShell launcher also accepted `-Resume -Hz 0` with a bounded
two-decision verification. Browser-based layout verification remains unavailable.

Evidence:

- `runs/verify-d2faa95fdf5c4300966e2a0744307489` (unit-test artifacts).
- `runs/internal-learn-20260918T011745Z-132c27/smoke.json` (isolated live speed edits).
- `runs/internal-smoke-20260918T011748Z-4d4d6a/report.json` (isolated exact resume).
- `runs/internal-learn-20260918T011909Z-83cdf9/summary.json` (unlimited/SIGINT check).

A separate in-memory timing diagnostic copied the user's step-1000 isolated
checkpoint, warmed up for 20 decisions, and timed 200 decisions without sleeps.
It averaged 31.65 ms per decision (~12.64x game speed at approximately 60 fps),
not the ~6x estimate inferred from the earlier capped run's 66 ms computation.
This excludes startup, checkpoints, trajectory-file writes, HTTP delivery, and
browser rendering, so it is not a guaranteed sustained dashboard rate.

| Work per decision | Mean wall time |
| --- | ---: |
| Brain observation, including learning traces | 27.62 ms |
| Emulator: 24 frames and reward hooks | 1.17 ms |
| Viewer activity, PNG and JSON preparation | 2.42 ms |

Inside the brain total, neural updates took 20.46 ms (including 6.77 ms for
CPU noise generation/upload) and learning eligibility bookkeeping took 5.83 ms.
These nested timings are not additional to the brain total. Twelve neural
updates visit about 307 million connections per decision. An isolated CUDA
propagation benchmark averaged 0.634 ms per neural step. CPU-side bookkeeping,
CPU/GPU transfers and repeated GPU operations are optimization candidates;
no numerical/learning optimization was applied in this pacing change.

### Existing checks

- Real neural input matches the streamed source image and retinal representation.
- All seven movement mappings are implemented, including MN9 proboscis extension.
- Existing internal synapses change and affect simulator current; no new learned
  external encoder or action-policy weights exist in `train`.
- Frozen and absent-reward controls preserve original weights.
- A 20-decision resume matches uninterrupted actions, rewards, images, and all
  saved neural arrays exactly on the tested CUDA backend.
- Autonomous HTTP/SSE display sends actual neural actions and refuses human
  button injection, even with a valid control token.
- ROM hash remains unchanged. Connectome/model/ROM/save/run files remain ignored.

The battle/capture observer has fixture tests and checked exact-ROM instruction
locations, but no natural autonomous battle/capture validation yet. Browser
layout/visual QA is also unverified because the browser tool repeatedly times
out. Node DOM/canvas unit tests and HTTP/SSE checks do not replace that check.

The implemented architecture and experimental assumptions are in
[EXPERIMENT.md](EXPERIMENT.md). Scientific model calibration, natural battle
tests, and reliable retained gameplay improvement remain unfinished research.
