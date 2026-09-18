# Follow-through: movement, internal learning and actual gameplay

Work is ongoing. This supersedes the earlier "no improvement" conclusion for
movement and basic motor learning, but does not claim Pokemon completion or
established screen-specific gameplay learning. Full protocol and failed
candidates remain in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).

## Latest evidence, 2026-09-18 10:04 UTC

Public source repository: https://github.com/Reldnahc/pokefly . ROM, model data,
saves, checkpoints and raw runs remain local and Git-ignored.

### Confirmed input-priority trap; serial-delivery results

The literal sustained/5-second-Start run remained in Bulbasaur's nickname
screen for thousands of decisions. The game already reports party count 1 at
this point, but the Pokemon's level is still zero. Measurements now separate
`first_party_count` from `first_starter` (requires a positive initialized level).
Earlier artifacts retain their original field; that timestamp may precede a
completed starter and must not alone be cited as success.

The pinned [naming-screen handler](https://github.com/pret/pokered/blob/a1a22aaf84d1675bcdbaeb194592379d586d838e/engine/menus/naming_screen.asm#L124-L187)
gives directions priority over Start, A and B. The sustained decoder nearly
always supplies a direction alongside a function button, masking that button.
On four exact copies of the recorded screen, 16 repeated simultaneous
direction+Start commands did not submit the name. The same commands delivered
serially initialized the party by decision 3 in all four directions. This is
a FORCED transport diagnostic, not autonomous play or learned behavior:
`forced-button-delivery-probe-20260918T090315Z-892e4e`.

Optional `sensorimotor-serial-v1` changes only delivery timing from the
sustained/5-second-Start model: direction held 11 frames, release 1, function
held 11, release 1. Same 24-frame budget, no added command, state detector,
learned adapter or reward. Single-button commands retain their full pulse.
Old checkpoints/default profiles retain simultaneous timing.

Actual fresh-bedroom seed 64 run `internal-learn-20260918T090356Z-7cfa7f`
completed 24,000 decisions: 478 sampled positions, six maps, a fully initialized
starter at 4,443, rival win at 5,092, one wild win, final level 6. It started
with original synapses and unchanged game rewards. There was still no new
town, capture or badge; this is not learned navigation. The same-seed
simultaneous control reached 501 positions but no wins. It eventually escaped
the name screen after >11,500 total decisions: a severe delay, not a permanent
lock. Exact recorded replay through 5,500 matches every state and reward
(`recorded-replay-proof-20260918T091426Z-1c786a`); that replay is not a new win.

Frozen Route-1 comparisons, 6,000 decisions each, are mixed:

| Seed | Simultaneous / serial positions | Northernmost Route-1 y | Wild wins |
| --- | --- | --- | --- |
| 1201 | 285 / 337 | 20 / 24 | 0 / 0 |
| 1202 | 303 / 414 | 20 / 24 | 2 / 1 |

Smaller y is farther north. Serial delivery improved command access and
coverage but not northward progress. The validated simultaneous controls are
explicitly reused, not additional trials. Artifact:
`button-timing-comparison-20260918T090536Z-f73dce`.

Initial serial trials were launched before a telemetry cleanup: their old
`pulse_frames` field still says 23 on paired commands, although delivery was
11+1+11+1. The recorded configuration identifies the exact timing; original
artifacts are not rewritten. New logs include phase schedules and correct
22-frame total held duration. UI hover no longer calls serial buttons simultaneous.

### Delayed-credit limitation remains open

Exact playback measured one rival faint-to-reward delay of32 decisions
(7.68 neural seconds) and two wild-battle delays of7/21 decisions (1.68/5.04 s).
No game reward timing/category/magnitude changed. Artifacts:
`recorded-outcome-latency-20260918T082827Z-7d13d1` / `...T082850Z-bfb366`.

The constant-image operant diagnostic was repeated with32-decision delayed
feedback,512 earning decisions per stage, plus32 delivery-only tail decisions.
All arms have matched length and shuffled reward counts; weights never enter
Pokemon. Before training: Up24.6%, Down22.7% on256 reward-free decisions.

| Local learning model | Retained paired Up / reversed Down | Shuffled Up / Down |
| --- | --- | --- |
| Short0.6s covariance | 23.4% /19.1% | 27.0% /21.9% |
| Original dual mean | 24.2% /23.8% | 23.0% /23.8% |
| Impulse-balanced dual covariance | 22.7% /28.1% | 21.1% /22.3% |
| Known-noise impulse dual | 25.0% /24.2% | 23.0% /30.9% |
| Event/Hebbian impulse dual | 21.9% /21.9% | 26.6% /20.7% |

None establishes acquisition plus reversal. Frozen controls reproduce the
pretest. At four times the training duration, the unchanged noise-dual model
also fails acquisition: Up 23.8% paired versus 24.6% before and 30.5% shuffled
(`operant-motor-probe-20260918T090913Z-b82659`).

The stochastic-spike score-v2 rule passes the immediate motor screen on one
seed: Up 16.4% -> 26.6% versus 17.2% shuffled; reversed Down 43.8% versus 32.0%
before and 25.4% shuffled (`...T091957Z-f8c78b`). Its 30-second delayed-score
variant, with an analytically variance-rescaled learning rate, does NOT pass
the delayed screen: Up 19.5% versus 16.4% before/shuffled; reversed Down 36.7%
versus 32.0% before and 33.6% shuffled (`...T092802Z-b87a7a`). Both gains are
below the preselected five-point requirement. Immediate learning does not
establish delayed credit. No synthetic assay-trained weights enter Pokemon.

### Earlier delivery of unchanged outcome rewards: implemented, being evaluated

`confirmed-outcome-v2` pays the same win at confirmed trainer victory, a wild
faint with a living active player, or a successful completed capture. The
stricter `last-faint-v3` can also pay at the final trainer faint, checking live
active HP, valid enemy party count/index and every other enemy's cached HP.
Double-KOs and uncertain cases fall back to the existing victory/end checks.
Both exclude demos/link battles and persist a once-paid flag. No HP reward,
waypoint, chosen-button feedback, category or amount was added. The [pinned
battle implementation](https://github.com/pret/pokered/blob/a1a22aaf84d1675bcdbaeb194592379d586d838e/engine/battle/core.asm)
is the basis for these execution-hook checks.

V3 exact replay: 88,000 recorded decisions, 120 completed encounters, identical
sampled states, legacy rewards, novelty timing and completed outcome totals.
Five rewarded encounters deliver 42/6/7/21/32 decisions earlier, respectively.
Artifact `reward-timing-audit-20260918T095419Z-6880e9`. These are old recorded
battles, not new successes; no actual capture occurred in this replay panel.
Capture and edge-case coverage is currently unit-fixture evidence.

Optional profiles `sensorimotor-outcome-v3` and `sensorimotor-serial-outcome-v3`
are being tested in matched repeated-battle learning and a fresh literal
24,000-decision bedroom run. Their learning benefit is not established yet.
Old profiles and checkpoints retain exact `encounter-end-v1` timing.

### Fresh images during commands: more battles, not solved navigation

With the corrected calibrated/serial controller and original frozen weights,
endpoint timing reproduces every snapshot action/state on both seeds. Streaming
fresh raw frames during commands changes only sensory timing:

| Seed | Snapshot / streamed positions | Northernmost Route-1 y | Wild wins |
| --- | --- | --- | --- |
| 1201 | 337 / 281 | 24 / 22 | 0 / 3 |
| 1202 | 414 / 401 | 24 / 24 | 1 / 3 |

Same recorded Route-1 reset, 6,000 decisions, no new town in either arm.
The snapshot controls are reused explicitly; duplicate endpoint trajectories
are validation, not independent successes. This is control/sensing evidence,
not learned battle strategy. Artifact:
`corrected-temporal-comparison-20260918T093145Z-5acca8`.

### Longer training and retained readout checks

The held-out whole-game tests after four 12,000-decision training attempts
do not show a better general player. Seed 1501: original / retained positions
444 / 408, wins 2 / 0. Seed 1502: positions 353 / 414, wins 1 / 0. Both original
brains won the rival battle; neither retained brain did. Retained B use grew
from 30.1% / 31.1% to 43.2% / 43.0%. No new town, capture or badge. This panel
does not isolate the cause of the B bias, but it rejects assuming that longer
training has helped. Saved synaptic changes are not general improvement. Artifact:
`retained-game-series-20260918T075051Z-705a0c`.

Re-reading the same actual motor spikes with a fixed one-second direction
trace barely changes overall cue accuracy for the acquired perturb-v3 brain:
52.7% -> 52.9%. The favorable late time bin is not a new trained result. Both
original and shuffled brains remain worse, but this is one training seed and
not an independent learning replication. Artifact:
`retained-readout-audit-20260918T092556Z-20ebae`.

Extending the existing score-v2 visual curve gives paired accuracy 52.0% at
8,192 then 46.2% at 16,384; its shuffled extension is still running. A separately
versioned presynaptically centered spike-innovation rule is now in ROM-free
testing. It is an approximation, not the exact score gradient of the physical
neuron, and has not been promoted to gameplay. Frozen full-neuron activity
matches score-v2 for 64 paired windows. No synthetic weights enter Pokemon.

### Longer-game plateau and fixed control comparisons

The dual-trace uninterrupted game now finishes at 40,000 decisions: 447
cumulative sampled positions, a rival win and two wild-battle wins, level 7,
but no new town, capture or badge. Route 1 eventually reached y=20 at decision
32,917. Earlier y=24 reports were interim measurements. This is still a plateau,
not evidence that enough passive runtime will complete the game. Source run
`internal-learn-20260918T070458Z-6a8fc0`, continuation
`internal-learn-20260918T071854Z-26737e`.

Measurement-only execution hooks replayed the entire 30,000-decision
continuation. All sampled states/rewards match, but **45.62% of frames were in
Start or its nested menus**, with 2,198 openings. Initial unknown context is
reported separately; timing is approximate to frame boundaries. Artifact:
`recorded-menu-audit-20260918T081409Z-908468`. These measurements never enter the
brain or decoder. Pokemon's [overworld handler](https://github.com/pret/pokered/blob/a1a22aaf84d1675bcdbaeb194592379d586d838e/home/overworld.asm)
checks Start before directions. This motivates a fixed-cooldown control test,
not a RAM-driven menu filter or removal of the Start mapping.

Frozen original-weight movement comparison, recorded Route-1 reset and 6,000
decisions per arm:

| Seed | Short / sustained mean direction bout | Short / sustained northernmost Route-1 y | Short / sustained positions |
| --- | --- | --- | --- |
| 1201 | 1.18 / 2.71 decisions | 24 / 22 | 245 / 230 |
| 1202 | 1.19 / 2.65 decisions | 32 / 20 | 222 / 225 |

No arm reached a town. One sustained arm won a wild battle; others did not.
This demonstrates coherent movement, not learned navigation. Artifact:
`movement-bout-comparison-20260918T072342Z-66b261`. The sustained-1201 control
spent 37.22% of frames in Start menus. The one-factor test changing only Start's
fixed cooldown from1 to5 neural seconds is complete. Positions increased
230->285 and225->303; menu-frame fraction fell37.22->14.07% and37.08->14.59%.
Wild wins changed1->0 and0->2, so there is no clear battle conclusion. Neither
reached a town. Its controls are reused explicitly, never counted as extra
trials. Artifact `start-cadence-comparison-20260918T081715Z-0fbcce`.
No default is changed by these tests.

### Partial visual learning and reversal

Perturbation-v3 uses existing neural-noise credit and centered presynaptic
history. After 16,384 synthetic diagnostic training decisions on seed 501,
the original retention probe gave 53.8% pooled conditional accuracy versus
37.1% shuffled and 43.1% before training. This is NOT a passed full assay.

Independent reward-free retention used 128 neutral warmup decisions, 128 test
decisions per cue, and two fresh noise seeds (512 tested decisions per arm):

| Test | Before training phase | Paired feedback | Shuffled feedback |
| --- | ---: | ---: | ---: |
| Acquisition, seeds 1311/1312 | 45.4% | 55.4% | 44.7% |
| Reversal, seeds 1313/1314 | 44.2% | 59.2% | 50.2% |

Values are pooled accuracy conditional on Left/Right choices, NOT accuracy
across all seven buttons. Acquisition paired and shuffled target-action rates
were both 26.95%; paired cue-specific accuracies were 62.2% and 48.4%. Thus two
clear associations were not established. The old criterion-2 screen, which
also requires target-rate gains, remains unchanged.

Reversal adds 2,048 training decisions. Its shuffled control starts from the
EXACT SAME acquired paired brain/dynamics, controlling for prior learning.
That control was designed after the initial reversal result and transparently
reuses the paired trial; it is not preregistered independent replication.
In the new-noise test, paired reversed-cue accuracies were 63.6% and 54.1%,
versus shuffled 67.3% and 32.4% (mostly one global button preference). Additional
descriptive cue-preference contrast is now reported to expose that distinction;
it does not retroactively change pass criteria. Independent training seed601
finished at50.0% paired versus40.2% shuffled at16,384, after56.0% paired at8,192.
Do not select only the best checkpoint: longer training lost the above-chance
result. Exact-input-normalized seed501 reached50.0% paired versus41.7% shuffled
at8,192, also not a robust two-cue solution. Samples within these time series
are correlated; these small results are not significance or Pokemon strategy proof.

Artifacts: acquisition `visual-learning-curve-20260918T063831Z-ffb363`,
reversal `...T073714Z-d4ad41`, matched control
`matched-visual-reversal-20260918T074442Z-fe16ab`, and independent retention
`visual-retention-probe-20260918T073238Z-46f5ed` / `...T075234Z-4d9e13`.
No diagnostic-trained synapses enter Pokemon. A fresh perturb-v3 MODEL, with
original weights and unchanged game rewards, finished24,000 actual decisions:
329 sampled positions, a level5 starter, no wins and no Route1 entry.
`internal-learn-20260918T081006Z-8f3f52`. This does not justify promoting it for
gameplay. Independent visual and normalization artifacts:
`visual-learning-curve-20260918T074556Z-02eb83` / `...T082252Z-7ebdac`.

### Active whole-game training

`scripts/train_game_series.py` carries GAME-trained internal weights through
four explicitly reset new-game attempts (seeds 1401..1404, 12,000 decisions
each), then freezes original and retained weights for matched seeds 1501/1502.
Only the intro is scripted. Game state, novelty ledger and fast neural state
reset; internal learned connections carry over. No waypoint curriculum or
synthetic button reward is used. This tests repeated practice, not one
uninterrupted playthrough. Results are pending, not presumed positive.

Other completed failures are preserved: dual-trace black/white-to-A/B and
bounded black/white-to-Up/Down do not pass the full counterbalanced assay.
The score-v1/v2/v3 and low-temperature escape candidates have not established
robust visual acquisition in their 2,048-decision screens. They remain optional
research models, not promoted replacements.

Latest completed verification:233 Python tests,23 JavaScript tests and lint
(`verify-9c4ea62ca9f84543a0f89e2c1ad22746`). Optional impulse/noise/event-dual and
serial models pass actual-ROM resume/frozen/no-reward checks. Serial live
HTTP/SSE pixels, phase delivery and speed controls pass
(`internal-learn-20260918T091204Z-6c4305/smoke.json`). A first smoke run correctly
failed its outdated23-held-frame assertion; the test now checks the configured
phase schedule. User's old checkpoint still reproduces its next20 recorded
decisions exactly (`legacy-resume-serial-20260918`).
Browser-rendered layout QA is still unavailable; emulator and HTTP/SSE checks
are not described as browser QA.

## What is implemented and usable

Fresh `scripts/start.ps1` launches now use `sensorimotor-bounded-v1`: fixed
neutral-image intrinsic calibration plus bounded learning on existing internal
motor/descending-neuron inputs. No button mapping, reward category/amount,
retinal features, route script or RAM-to-action input was added. The original
ROM, connectome and stopped user's checkpoints remain unchanged and ignored
where appropriate. Old checkpoints retain their own model on resume.

```powershell
.\scripts\start.ps1 -Intro -Hz 0
```

Default duration is still unlimited; Ctrl+C checkpoints. `-LoadState PATH`
can start a fresh brain from an existing game state, without scripting a route.
`-Weights CHECKPOINT -Intro` retains that checkpoint's weights/model for a new
game; `-Resume CHECKPOINT` continues its entire saved experiment. A saved old
model is not silently converted into the new model.

## The weak-Up defect: actual application comparison

Matched outdoor starting state, seed 401, 6,000 decisions, frozen weights.
The only changed model input is neutral-image excitability calibration.

| Measurement | Original isolated model | Calibrated model |
| --- | ---: | ---: |
| Up commands | 234 (3.9%) | 1,325 (22.1%) |
| Distinct sampled overworld positions | 25 | 232 |
| Maps visited | 1 | 4 |
| Lower-edge occupancy within nonbattle Pallet samples | 86.5% | 18.8% |
| Starter obtained | No | Yes |

Occupancy includes stationary menus/dialogue, so it is not a collision-time
measurement. This is control improvement, NOT evidence of reward learning.
Independent calibrated seed 402 produced 195 positions and 23.0% Up. Bedroom
seed 403 left the house at decision 447, producing 291 positions and 22.5% Up.
Neither of those two additional frozen runs obtained a starter within 6,000.

Sources in `runs/`:

- Original control: `internal-frozen-20260918T042436Z-c44df8`.
- Calibrated seed 401: `internal-frozen-20260918T041520Z-1f3501`.
- Calibrated seed 402: `internal-frozen-20260918T041942Z-17e21f`.
- Calibrated bedroom: `internal-frozen-20260918T043433Z-c5fa2f`.

The calibration applies one identical homeostatic equation to every nonsensory
spiking neuron, on uniform gray, with no game frames, rewards, motor labels or
button-frequency targets. Offsets are frozen before testing. The 1 Hz fitting
target is an engineering assumption, not measured cell-specific physiology.
It does NOT imply every neuron subsequently fires at 1 Hz: fresh-reset KC/MBON
activity is much lower, an important remaining memory-circuit limitation.
Regeneration with `scripts/probe_intrinsic.py --calibration-only --device cuda`
reproduced the deployed offset array bit-for-bit. See MODEL_VARIANTS for details.

## Behavioral learning, retention and reversal

ROM-free diagnostic: one constant raw gray image, 512 training decisions per
phase, then 256 reward-free decisions with fresh neural dynamics/noise. An
actual fixed-decoder button earns synthetic feedback in this assay only.
The learner receives neural activity and a scalar, never an action identity.
Reversal switches the rewarded direction. Slow reward expectation is preserved
between training phases; retention probes are separate branches.

| Neural seed | First rewarded button | Before / retained after training | Opposite button after reversal | Shuffled first / reversal target |
| --- | --- | --- | --- | --- |
| 501 (discovery) | Up | 24.6% / 38.3% | Down 46.1% | 22.7% / 23.8% |
| 601 (confirmation) | Up | 23.4% / 37.5% | Down 46.1% | 19.9% / 27.7% |
| 602 (confirmation) | Down | 23.8% / 53.5% | Up 37.1% | 29.7% / 23.0% |

Frozen controls reproduce pretraining behavior. These are descriptive,
correlated time-series measurements, not a significance claim. They establish
retained, reward-contingent motor preference and reversal in this experimental
neural model. They do NOT establish visual understanding, biological fly
learning fidelity, or useful learned Pokemon strategy. Diagnostic-trained
weights are NEVER used in the Pokemon runs.

Artifacts: `operant-motor-probe-20260918T043503Z-917333`,
`operant-motor-probe-20260918T043820Z-e303f5`, and
`operant-motor-probe-20260918T044641Z-b6b7aa`.

The first broader rule without tight limits learned Up but locked into it and
failed reversal. Exact per-neuron input normalization prevented lock-in but
showed weak acquisition. Both failed candidates remain available as opt-in
controls; neither is the launcher default.

## Actual Pokemon runs with learning enabled

Fresh internal weights; original general novelty and battle rewards only.
Same outdoor start as the frozen comparisons, 6,000 decisions per run.

| Seed | Starter decision | Rival outcome | Route 1 | Final level | Sampled positions |
| --- | ---: | --- | --- | ---: | ---: |
| 401 | 547 | Win at 1,491 | Entered at 1,882 | 6 | 221 |
| 402 | 2,887 | No recorded win | Not reached | 5 | 239 |

Artifacts: `internal-learn-20260918T043832Z-deaf73` and
`internal-learn-20260918T045003Z-7ea083`. The first is the preselected source for
held-out retention tests, not a cherry-picked best checkpoint.
Map 0x0C is Route 1 in the [pokered map constants](https://github.com/pret/pokered/blob/master/constants/map_constants.asm).
Battle wins are confirmed by the existing game's execution hooks, not guessed
from motion or inferred solely from level changes. No captures or badges have
been demonstrated in these runs.

The entire seed-401 recorded button sequence was independently replayed from
its saved initial emulator state. All 6,000 sampled states AND reward events
matched exactly. Unmodified emulator milestone images and the verification
report are in `runs/recorded-replay-proof-20260918T050823Z-fffce0/`.
These images are **recorded playback**, not additional autonomous successes.

Progress alone cannot attribute improvement to learning: the frozen calibrated
brain also sometimes gets a starter. The preselected seed-401 checkpoint was
tested on three held-out seeds, with learning disabled in both arms, the same
starting state and 6,000 decisions each:

| Seed | Original / retained positions | Original / retained starter decision | Original / retained battle wins |
| --- | --- | --- | --- |
| 701 | 257 / 333 | None / 2,123 | 0 / 0 |
| 702 | 264 / 244 | None / 1,996 | 0 / 0 |
| 703 | 208 / 254 | 5,025 / 1,081 | 1 / 0 |

Saved weights obtained a starter in 3/3 trials versus 1/3, but did not improve
battle wins and reduced exploration in one seed. This is limited positive
retention evidence for early interactions, not general learned strategy or a
statistical significance claim. Report: `retained-gameplay-20260918T045336Z-0fbf64`.
All six trials, including the regression, are retained.

The literal PowerShell launcher was also tested from the bedroom with its
default profile and default neural seed 64: house exit at decision 350, 257
positions and 1,325 Up commands in 6,000 decisions. It did not obtain a starter.
Artifact: `internal-learn-20260918T050029Z-0936da`.

## Controlled actual-battle learning

Each training episode explicitly resets the recorded first rival battle. This
is a diagnostic intervention, not autonomous whole-game progression. Eight
training episodes use only existing win/rival rewards; no action rewards or
forced moves. Original and retained evaluations freeze weights and match game
state and neural noise. The same baseline is reused across trace-duration
candidates because inactive eligibility cannot affect frozen forward dynamics;
reused results are identified, not counted as additional trials.

| Frozen evaluation panel | Original weights | Trained 0.6 s trace | Trained 30 s trace | Trained dual trace |
| --- | ---: | ---: | ---: | ---: |
| Seeds 1001--1008 | 4/8 wins | 6/8 | 8/8 | 6/8 |
| New seeds 1101--1108 | 3/8 wins | 5/8 | 5/8 | 6/8 |

This is positive, retained actual-battle evidence across two panels, with
regressions in individual seeds. It is not statistical proof of general battle
strategy: one starting matchup, small samples and multiple investigated models.
The long trace does NOT retain the default's strong immediate operant learning
within 512 training decisions. A two-timescale candidate is being evaluated to
retain both capabilities before changing the launcher default.

Artifacts: `controlled-battle-learning-20260918T053123Z-7acd46`,
`...T060231Z-564dc5`, `...T060731Z-eda3d8`, `...T062851Z-a7d336`.
The long-trace brain also ran in the actual launcher from its final training
checkpoint: `internal-learn-20260918T062745Z-ecfad9`. It finished at sample
19,276, with 425 sampled positions across six maps, but no further wins/towns
and Route 1 limited to y=28..35. The battle-training resets are an explicit
intervention; this continuation does not prove general learned exploration.

The dual-trace candidate (`controlled-battle-learning-20260918T064014Z-dd2d6d`,
confirmation `...T070446Z-946834`)
also retained operant acquisition/reversal on discovery 501 and confirmation
601/602, including initially rewarding Down. Corresponding paired target rates
were Up 35.2%, Up 29.7%, Down 47.7%; after reversal Down 44.5%, Down 44.9%,
Up 34.8%. Shuffled controls were 25.4/21.5%, 19.1/25.4%, 27.3/22.3%.
These assay-trained weights are never used in Pokemon. The default remains the
bounded single-trace profile while dual-trace longer-gameplay checks continue.

Fresh dual-trace seed 401, without battle pretraining, obtained Bulbasaur at
decision 4,243 and won the rival at 6,561. It reached 307 positions by 10,000.
Recorded playback matches every state/reward event (10,000 samples), artifact
`recorded-replay-proof-20260918T071938Z-2a9fff`. The unreset live continuation
entered Route 1 at 10,632 and recorded a wild-battle win at 11,967. It has not
yet demonstrated reaching another town. These are additional actual gameplay
outcomes, not a learned-strategy attribution; longer observation is ongoing.

## Longer gameplay exposes a remaining plateau

The original bounded seed-401 game was continued without resets from decision
6,000 through 24,000. It reached 399 cumulative positions but no further battle
wins or new towns. Route 1 observations stayed in its lower section (y=26..35).
Artifact: `internal-learn-20260918T055141Z-c23a10`. This is a real remaining
exploration failure; successful Up delivery is not the whole solution.

## What still needs evidence

- Useful learned gameplay across repeated independent trials, rather than
  changed weights, greater activity, or a lucky route.
- Screen-conditioned choices. Bounded, input-budget and perturbation-v1
  two-cue tests failed to establish both arbitrary associations and reversal.
  Increased Left/Right activity alone is not visual discrimination. A longer
  perturbation-v1 trial at 8,192 decisions still gave only 48.1% conditional
  accuracy versus 53.2% with shuffled feedback (192 held-out test decisions).
  Artifact: `visual-learning-curve-20260918T052012Z-af13f4`.
- Sustained progress beyond the early game; battle/capture competence and badges.
- Faithful biological dynamics. Connectome connectivity alone does not provide
  calibrated physiological parameters or a complete natural learning mechanism.

Two further opt-in hypotheses are being tested: bounded total synaptic input
with flexible individual connections (`sensorimotor-budget-v1`), and credit
from actual existing neural-noise perturbations (`sensorimotor-perturb-v1`).
Neither is promoted merely because it changes weights or passes a motor assay.
The separately versioned `sensorimotor-perturb-v2` fixes a demonstrated negative
bias caused by clipping an asymmetric, supposedly zero-mean neural-noise trace.
Eligibility is now linear; synaptic weight bounds and input budgets remain.
Its numerical and exact-resume tests pass, but the completed visual-choice
screen did not establish both associations/reversal. Version 3 centers local
presynaptic history as well. Its single-seed learning curve at 8,192 decisions
reaches 54.3% conditional accuracy versus 39.2% shuffled and 43.1% pretraining;
the target-action rates are 26.6%, 20.8% and 16.1%. This is an exploratory trend,
not a passed full counterbalanced/reversal test. Saved training continues to
16,384 decisions. No diagnostic-trained weights enter Pokemon.
Artifacts: `visual-learning-curve-20260918T060003Z-607331` and
`visual-learning-curve-20260918T061356Z-a8f1b1`.

Read-only neural diagnostics distinguish current cues at motor inputs across
independent noise seeds, including shortly after switches. A frozen-input
capacity bound finds that the default's tight individual edge bounds cannot
reverse the direct mean left/right preference; the wider input-budget profile
can. These are OFFLINE measurements, not classifiers or fitted weights deployed
in the game: `visual-latency-probe-20260918T061633Z-f2b2e1` and
`visual-capacity-audit-20260918T063501Z-0407b1`.
The launcher default remains unchanged while these candidates are tested.

## Verification so far

- Latest full suite: 263 Python tests, 23 JavaScript tests and lint pass
  (`verify-714b5668c68246fdaae6fd76ff49cc2a`). All ten protected source/ROM/
  model/user-run hashes still match at 10:06 UTC.
- Early-outcome checkpoint at decision 5,050 contains a paid but unfinished
  rival encounter. Resuming it reproduces all next 50 samples, final neural
  arrays and rewards exactly, without duplicate payment
  (`checkpoint-window-verification-20260918T101037Z-65eccc`). This is a
  verification branch, not another autonomous win.
- The original user checkpoint reproduces recorded samples 68,501..68,520,
  and its split continuation reproduces all final arrays/rewards
  (`checkpoint-window-verification-20260918T101118Z-9e8f3d`). Source unchanged.
- Centered-innovation and serial/outcome profiles pass real-ROM exact resume,
  frozen and no-reward smoke checks. Smoke runs are not gameplay evidence.
- Exact application checkpoint continuation, including neural arrays and rewards.
- Frozen and zero-reinforcement controls leave original weights unchanged.
- Real live stream matches input pixels, neural buttons and releases; live speed
  changes work with autonomous human controls disabled.
- 178 Python tests and 21 JavaScript tests passed before the v2 addition, plus
  27 live checks across 13 profiles. The v2 correction adds a zero-mean unit
  regression and passes its actual-ROM exact-resume/frozen/no-reward smoke.
  A fused eligibility update matches the preserved NumPy rule
  bit-for-bit; its isolated kernel benchmark was about 4x faster, not the game.
- Protected ROM/model/user-run hashes match the original manifest. ROM and
  generated artifacts remain Git-ignored.
- The original user's checkpoint also reproduced the next 20 recorded decisions
  exactly after the acceleration: `legacy-resume-followthrough-20260918`.
- In-app browser unavailable despite retries. No claim of browser-rendered
  visual layout verification; actual application/emulator/HTTP checks did run.
