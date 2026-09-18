# Pokefly: neural-control and learning improvement plan

Status: REOPENED. The user rejected stopping after the bounded comparison.
Continue diagnosis, implementation, and actual-application experiments until
there is credible evidence of repeatable progress and retained internal learning,
or a genuine external/authority blocker. Finishing a trial budget or finding a
promising hypothesis is not the stopping condition. Do not claim that a short
test proves eventual game completion.

## Active follow-through (supersedes the earlier stopping rule)

- [ ] Trace the remaining visual/memory/motor bottlenecks; test mechanisms, not
  merely repeat the original symptom. Inspect the simulator's assumptions.
- [ ] Explore intrinsic neural-dynamics calibration and internal credit assignment
  on existing pathways, keeping pixels-only input and the fixed button adapter.
  Global neuronal calibration must not be a disguised Up gain or button quota.
- [ ] Iterate on failed candidates. Require actual behavioral learning against
  frozen and reward-shuffled controls, including retained and reversed choices.
- [ ] Run promising implementations in the literal application from the bedroom
  and outdoor starts, then longer independent-seed gameplay. Measure sustained
  escape from boundary loops and meaningful interactions/progress without routes
  encoded in actions or rewards. Preserve all failed trials.
- [ ] Verify persistence, tests, unchanged protected data, and the live display;
  report a precise confidence limit on further-game potential. Do not substitute
  software test passes, increased Up counts, or one lucky route for learning.

### Follow-through evidence and live work

Latest checkpoint of work (older entries below are a chronological lab log):

- 08:23 UTC: independently trained perturb-v3 seed 601 reached 56.0% at 8,192,
  then fell to 50.0% at 16,384; shuffled control still running. Do NOT select
  only the best checkpoint or call prolonged training a reliable solution.
  One-factor mechanism probe: exact per-postsynaptic-neuron positive input
  normalization instead of the +/-25% total-input budget, same v3 rule and
  individual 0.25..4x limits. This constrains global excitability drift while
  permitting competition among EXISTING inputs. Offline frozen-input capacity
  audit still permits both cue-current signs (`visual-capacity-audit-20260918T082211Z-bb934f`);
  optimized weights are never exported/deployed. Preselect seed 501, snapshots
  2,048 and 8,192 with matched within-cue shuffled feedback; only an informative
  result advances to independent seed/reversal. No game reward or input changes.
- 202 Python, 22 JS and lint pass. Optional 5 s Start profile passes actual-ROM
  exact resume/frozen/no-reward checks (`internal-smoke-20260918T082021Z-bc9f8b`).

- 08:16 UTC: menu replay audit found 45.62% of frames in Start/nested menus,
  2,198 openings across the 30,000-decision dual continuation. All states and
  rewards exactly match; `recorded-menu-audit-20260918T081409Z-908468`.
  Mechanism test preselected BEFORE outcomes: keep the sustained decoder and
  original frozen neural weights, change only Start's existing fixed cooldown
  from 1 to 5 neural seconds. Same Route-1 reset, seeds 1201/1202, 6,000 each.
  Reuse the completed sustained-1s controls explicitly, not as new trials.
  No menu detection, action quota, learned adapter or reward change in policy.
  Five seconds is an engineering probe, NOT fitted/validated song physiology.
  Require actual travel/menu measurements before a learning run or promotion.

- 08:10 UTC: dual-trace uninterrupted continuation finished at 40,000: 447
  cumulative positions, rival plus two wild wins across the source/continuation,
  level 7, no new town/capture/badge. Read-only full-log audit corrects the earlier
  interim Route-1 bound: it reached y=20 at 32,917, not just y=24. It was still
  moving among Pallet/Route 1, with menu interruptions; final frame is the party
  menu, not evidence of a permanent emulator freeze. This plateau remains open.
- Frozen sustained-movement comparison completed both seeds. Minimum Route-1 y
  short/sustained: seed 1201 24/22, seed 1202 32/20; no town in any arm.
  More coherent movement is demonstrated, not useful learned navigation.
- Independent reversal retention completed: before 44.2%, paired 59.2%, matched
  shuffled 50.2% pooled conditional accuracy (512 decisions, new seeds 1313/1314).
  Paired reverse-mapping per-cue accuracies 63.6/54.1%; shuffled 67.3/32.4%.
  Add descriptive cue-preference contrast to distinguish global motor bias;
  DO NOT change the existing criterion-2 success screen after seeing results.
- Next actual-game test: run the visually more promising perturb-v3 MODEL with
  ORIGINAL synapses, common recorded outdoor start, preselected seed 401 and
  24,000 decisions. It receives only unchanged Pokemon rewards, no diagnostic
  weights or actions. Compare the first 6,000 with the existing matched frozen
  and bounded controls; compare later progress with their logged continuations.
  This is an explicit starting-state reset, not a new-game or uninterrupted
  continuation claim. Seed 402 confirmation is conditional on informative
  progress/retained benefit, not a search for a lucky seed.
- Literal sustained-model bedroom run and four-attempt retained-game series
  remain active. Escape-model actual-ROM exact resume/frozen/no-reward checks
  passed (`internal-smoke-20260918T080604Z-cea4f7`); its visual assay did not
  establish a solution and the default remains unchanged.

- 07:55 UTC: full new-game retained-training series running at dashboard port
  8779: `retained-game-series-20260918T075051Z-705a0c`. Four preselected training
  seeds 1401..1404, 12,000 decisions each, then frozen original/retained panels
  1501/1502. Every attempt has an explicitly scripted intro only; neural weights
  carry across attempts, game/novelty/fast state restart. These are labeled
  training resets, not one autonomous playthrough and not waypoint curricula.
  This addresses scarcity of fresh rewards after a single game saturates local
  novelty. Default application remains indefinite and unchanged.
- Same-acquired-state reversal control completed: before 46.2%, paired at
  +2,048 decisions 64.2%, shuffled from identical weights/dynamics 47.1%.
  Artifact `matched-visual-reversal-20260918T074442Z-fe16ab`; paired results are
  reused, NOT additional trials. New independent-noise retention 1313/1314
  is running, as is independently trained seed 601. This is partial visual
  learning evidence, not established arbitrary two-cue skill or Pokemon strategy.
- Actual dual-trace continuation replayed through 15,700, including both wild
  victories; all 5,700 recorded states/rewards matched. Unmodified playback
  frames show Bulbasaur using Tackle against Pidgey/Rattata. Artifact
  `recorded-replay-proof-20260918T075422Z-23e0d7`; not extra autonomous wins.
- Latest full verification: 199 Python, 22 JS, lint passed. Browser skill retried
  at 07:47, still cannot connect to iab; live HTTP/SSE checks are not layout QA.

- 07:45 UTC: source-only public snapshot `bebb493` pushed successfully. No
  private data/ROM/save/model files in Git. Current tests: 198 Python, 22 JS;
  sustained and signed-score actual-ROM resume/frozen/no-reward checks pass.
- Dual-trace live continuation additionally won wild battles at decisions
  11,967 and 15,697, reaching level 7. Route 1 reached y=24, then exploration
  slowed again; still no new town. Continue observing, not a solved policy.
- Frozen movement-bout seed 1201: short/sustained mean direction runs 1.18/2.71
  decisions; Route-1 minimum y 24/22, positions 245/230, wild wins 0/1. No town
  in either. Seed 1202 still running. Do NOT promote from this mixed result.
- Independent visual retention of perturb-v3 after 16,384 training decisions:
  neutral warmup 128, two new noise seeds, 512 total test decisions, no rewards.
  Conditional accuracies original 45.4%, paired 55.4%, shuffled 44.7%; paired
  and shuffled target rates both 26.95%. Paired per-cue accuracy Left 62.2%,
  Right 48.4%, so two clear associations are NOT established.
  Artifact `visual-retention-probe-20260918T073238Z-46f5ed`.
- Explicit v3 reversal at 2,048 further decisions gives paired conditional
  64.2% (before reversal 46.2%). Old shuffled-acquisition starting weights
  already favor the reversed contingency, so a new post-discovery shuffled
  reversal control starts from EXACTLY the same acquired paired brain;
  `probe_matched_reversal.py` transparently reuses paired results, not new trials.
  Independent acquisition seed 601 is now running to 16,384; no success claim.
- Score-v2 black/white -> A/B also failed (paired 34.6%, shuffled 32.7% at
  2,048). Signed score-v3 tests learning on existing inhibitory as well as
  excitatory inputs, preserving signs and separate per-sign resource budgets.
  Escape-v1 removes redundant background current noise and narrows stochastic
  firing temperature from 0.05 to 0.02, with fresh neutral-only calibration
  `intrinsic-probe-20260918T074218Z-01220f` and proportionally smaller update
  step. These remain diagnostic hypotheses, not promoted models or biology.

- 07:24 UTC: dual-trace independent battle confirmation completed 6/8 versus
  original 3/8 (`controlled-battle-learning-20260918T070446Z-946834`). Fresh
  autonomous dual-trace seed 401 obtained Bulbasaur at 4,243 and won the rival
  at 6,561, reaching 307 positions by 10,000. Entire recorded run replayed
  exactly (`recorded-replay-proof-20260918T071938Z-2a9fff`); images are playback,
  not extra successes. Literal launcher now continues that game to 40,000
  (`internal-learn-20260918T071854Z-26737e`), no diagnostic-trained weights.
- Score-v1 visual discrimination failed to improve over shuffled at 2,048;
  symmetric wrong-choice diagnostic feedback also failed. Score-v2 tests a
  synapse-size preconditioner, not an added policy; no success claimed.
- Separate fixed-adapter hypothesis: longer, decaying directional motor traces
  may permit coherent walking bouts without altering buttons/rewards or learning
  outside the brain. Opt-in sustained-v3 applies the same rule to all directions;
  functions retain the old immediate-spike rule. Frozen matched two-seed Route-1
  comparison is running via `evaluate_movement_bouts.py`; reset to recorded
  sample 1,882 is a labeled diagnostic intervention. Do not attribute this to
  learning or promote before useful results. Legacy decoder stays exact.

- 07:13 UTC: stochastic likelihood-score variant implemented as opt-in
  `sensorimotor-score-v1`; finite-difference/zero-mean-score tests pass.
  191 Python and 21 JS tests pass. Actual-ROM exact resume/frozen/no-reward
  passed (`internal-smoke-20260918T070537Z-cca273`), live pixels/buttons/speed
  passed, and original user's next 20 recorded decisions still match exactly
  (`legacy-resume-score-20260918`). All ten protected hashes match.
  Calibration `intrinsic-probe-20260918T070346Z-1ad5bd` is neutral-only.
  Initial visual curve: 46.6% pre, 49.1% at 512, 50.8% at 2,048; still NOT a
  behavioral solution. Matched shuffled controls running. Separately test
  equal +/-1 correct/wrong competing-choice feedback ONLY in the ROM-free
  diagnostic to distinguish discrimination from general Left/Right activation.
  No game reward or diagnostic-weight transfer is changed/allowed.
- Perturb-v3 16,384 curve completed: 53.8% paired versus 37.1% shuffled
  conditional accuracy, target rates 25.5%/24.0%. Still lacks repeatable
  acquisition/reversal. Bounded black/white -> Up/Down two-seed/full-reversal
  assay also completed without passing every seed/mapping; raw results retained
  in `visual-learning-up-down-bounded-20260918`.
- Dual-trace motor confirmation completed including initial Down seed 602:
  paired Down 47.7% then reversal Up 34.8%, shuffled 27.3% then 22.3%, frozen
  exactly pre. Initial actual-battle panel 6/8 versus reused baseline 4/8;
  independent 1101..1108 pending. Fresh literal application seed 401 is running
  from the common outdoor start (`internal-learn-20260918T070458Z-6a8fc0`),
  with no synthetic training and no route. No new profile promoted.

- 07:00 UTC: long-credit actual application continuation completed at sample
  19,276 (`internal-learn-20260918T062745Z-ecfad9`): six maps, 425 sampled
  positions; Route 1 remained y=28..35 and no further battle victories. This
  does NOT resolve long-run exploration. Both short and long-credit battle
  weights won 5/8 on confirmation seeds 1101..1108 versus the same original 3/8.
- Dual fast/slow eligibility retains operant acquisition/reversal on seeds
  501 and 601, with unchanged frozen controls; counterbalanced 602 and actual
  battle retention are running. Centered covariance and low-noise visual
  variants failed their exploratory learning curves. Perturb-v3 at 16,384
  training decisions plateaus at 53.8% conditional accuracy (control pending).
- Next bounded mechanism test: an OPTIONAL stochastic-spiking model with a
  local likelihood-score eligibility signal, instead of a heuristic firing
  covariance. Motivation: Florian 2007's stochastic-neuron derivation, not a
  claim of measured fly physiology. Keep original topology, pixel input,
  anatomy-only plastic targets, fixed decoder and general game rewards.
  Refit neutral excitability without images/actions/rewards from the task;
  verify the score by finite differences before visual/operant assays. Never
  transfer synthetic-assay weights to the game. Defaults and legacy checkpoints
  must stay reproducible. This tests a specific credit-assignment failure,
  not another arbitrary batch of unchanged gameplay trials.

- Full live continuation completed: `internal-learn-20260918T055141Z-c23a10`,
  source seed-401 checkpoint continued from 6,000 to 24,000 decisions. It reached
  399 cumulative sampled positions but no additional battle wins/new towns.
  Route 1 sampling stayed within y=26..35: the movement defect is improved,
  but useful exploration beyond the lower route remains unsolved. Up pulses
  actually move the player; do not describe this plateau as success.
- Controlled battle retention completed: original 4/8 wins, after eight actual
  battle-learning episodes 6/8 wins, same frozen evaluation seeds 1001..1008.
  Artifact `controlled-battle-learning-20260918T053123Z-7acd46`. Independent
  confirmation seeds 1101..1108 are running in `...T060731Z-eda3d8`, with no
  additional learning. Small-sample positive evidence, not robust mastery.
- Longer-credit candidate changes ONLY eligibility duration, 0.6 to 30 neural
  seconds. Same generic outcome rewards. Reuses the identical frozen baseline
  transparently (not counted as fresh trials); config/input hashes are checked.
  `controlled-battle-learning-20260918T060231Z-564dc5` is running. New battle
  assays export standard application checkpoints as well as raw neural arrays.
- `sensorimotor-perturb-v2` completed both seeds/mappings without reliable
  visual learning. `sensorimotor-perturb-v3` additionally centers presynaptic
  activity against its previous local 10 s mean. Same anatomy and input budget;
  no action/cue identity enters the rule. Seed 501 after 2,048 decisions:
  paired conditional accuracy 50.0%, shuffled 43.0%, pretest 43.1%. NOT a pass.
  Full-state continuation to 8,192 decisions is running from
  `visual-learning-curve-20260918T060003Z-607331`; no restart/cherry-picking.
- Reset-conditioned neutral calibration (`--reset-every 200`) removes most of
  the KC/MBON post-reset silence: three held-out seeds give KC 0.52--0.67 Hz,
  MBON 0.85--1.13 Hz. Artifact `intrinsic-probe-20260918T055315Z-45e2c3`.
  However, compartment learning with this calibration failed the full two-cue
  assay: `visual-learning-compartment-reset-v2-20260918`. Activity is not memory.
- Read-only latency probe `visual-latency-probe-20260918T061633Z-f2b2e1`: four
  independent noise seeds, constant and switched cues. Offline nearest-centroid
  measurements distinguish cues at visual projection, descending and motor-input
  populations (8/8 constant and 12/12 in each switch window), but not reliably
  at KC/MBON populations. This diagnostic classifier is NEVER a game policy.
  It rules out missing/delayed motor-pathway visual information as the whole
  explanation, not a proof of behavior or statistical significance.
- A read-only frozen-input capacity calculation indicates +/-25% individual
  excitatory bounds cannot reverse either DNa02 direct mean cue preference;
  the wider 0.25--4x/input-budget profile can. This is a first-order estimate
  with presynaptic activity fixed, not an optimized controller to deploy.
- Next signal-to-noise hypothesis: smaller neural perturbations (0.08 versus
  0.22), with the SAME neutral-reset calibration procedure refitted to retain
  activity. Separately versioned artifact/profile; no motor-label fitting.
  Original calibration/default remain unchanged. Require held-out/shuffled
  behavioral checks before any promotion.
- Verification now includes 181 Python and 21 JS tests; v2 and v3 actual-ROM
  exact resume/frozen/no-reward tests passed. Browser connection retried and
  still unavailable; HTTP/SSE was live through decision 18,367, not visual QA.

- Public repository created and audited source/docs pushed with `gh` outside
  the sandbox: https://github.com/Reldnahc/pokefly , initial commit `121f2a4`.
  No ROM, saves, model data, checkpoints or raw runs uploaded.
- Held-out retained-gameplay comparison completed: starters 3/3 retained versus
  1/3 original, but battle wins 0/3 versus 1/3. Per-seed positions were
  333/244/254 retained versus 257/264/208 original. Do not claim general strategy.
- Literal default launcher seed 64 left the bedroom/house at decision 350;
  257 positions, no starter in 6,000 decisions. All 10 protected hashes unchanged.
- Full pre-v2 validation passed: 178 Python, 21 JS, 27 live checks/13 profiles.
  Legacy user's checkpoint continuation exactly matched 20 original decisions.
- Longer visual perturb-v1 trial failed: paired conditional accuracy 48.1%
  versus shuffled 53.2% after 8,192 decisions. Raw cue contrasts at motor inputs
  persist into the second half of 256-decision probes (Left 22.9%, Right 24.1%);
  missing information is not simply an initial image-transient problem.
- Testing `sensorimotor-perturb-v2`: clipping asymmetric Bernoulli innovations
  creates negative expected eligibility (proved by enumerating outcomes).
  New version keeps a linear eligibility trace; v1 and default stay unchanged.
  Numerical regression and actual-ROM exact resume passed. Two seeds, both
  visual mappings, acquisition/reversal and shuffled/frozen controls running.
- Controlled actual-battle experiment is running: 8 frozen baseline seeds,
  8 learning episodes with explicit resets, then the same 8 held-out seeds.
  Original win count is 4/8; retained result pending. These resets are diagnostic
  interventions, not autonomous whole-game progression. Existing rewards only.

- Neutral intrinsic calibration is an explicit new modeling hypothesis: all
  non-sensory spiking neurons, never a button-specific group, adapt excitability
  using the same 1 Hz target on uniform gray for 10,000 neural steps (seed 707).
  Offsets are frozen before tests. This is not measured cell-specific physiology.
  Primary motivation: https://elifesciences.org/articles/45717 .
- `runs/intrinsic-probe-20260918T041050Z-7d0997/`: three probe seeds, four raw
  images, baseline versus fixed calibrated offsets. Town-screen Up increases
  from 9/13/14 to 60/63/63 per 256 decisions. Visual sensitivity remains.
  Calibration artifact: `fly-data/intrinsic-neutral-v1.npz`, neuron-ID checked
  and checksum identified by `configs/intrinsic-v1.json`. Original files unchanged.
- Literal application, frozen weights, original outdoor state, seed 401,
  6,000 decisions with live dashboard: `internal-frozen-20260918T041520Z-1f3501`.
  1,325 Up pulses, 232 positions, four maps. Starter at step 2,342; trainer battle
  at 2,533 through 4,782, no win reward. This is control/progression evidence,
  NOT reward learning. No script after loading the common starting state.
- Independent seed 402: `internal-frozen-20260918T041942Z-17e21f`, 6,000 decisions,
  195 positions and 1,377 Up pulses. Original-profile seed-401 matched long
  control is running in `internal-frozen-20260918T042436Z-c44df8`.
- Broader internal-learning hypothesis: reward-modulated neural covariance on
  495,962 existing excitatory inputs to ALL anatomically classified descending
  and motor neurons. No action label or trained adapter enters the learner.
  Motivation: https://florian.io/papers/2007_Florian_Modulated_STDP.pdf ; fly
  motor learning is not equivalent to mushroom-body-only plasticity (see
  https://pubmed.ncbi.nlm.nih.gov/38779314/), but this is not a molecularly
  validated fly model. First fixed-image operant test learns Up strongly but
  locks in and fails reversal. Testing local synaptic-input normalization as
  a mechanism to prevent this failure; do not promote the locked-in candidate.
- Ongoing isolated motor-learning controls: `operant-motor-probe-20260918T041911Z-9d88d7`
  (unconstrained, seed 501), plus normalized candidate. Synthetic button-contingent
  rewards exist ONLY in these ROM-free diagnostic assays, never in Pokemon.
- Browser skill has been read and connection retried; in-app browser remains
  unavailable. The literal application/live server runs, and actual emulator
  output and full logs are inspected; do not claim browser-rendered visual QA.
- Matched original outdoor seed-401 control completed: 25 positions, one map,
  234 Up pulses (3.9%) in 6,000 decisions, versus calibrated 232 positions,
  four maps and 1,325 Up pulses (22.1%). Calibration is a control fix, not learning.
- Bedroom seed 403, literal live application, frozen calibrated weights:
  `internal-frozen-20260918T043433Z-c5fa2f`. House exit at decision 447;
  291 visited positions and four maps by 6,000. No starter in this seed/budget.
- Unconstrained sensorimotor learning locks in; per-neuron input normalization
  avoids lock-in but gives weak acquisition (`operant-motor-probe-20260918T042424Z-0f0946`).
  Neither is promoted. Candidate `sensorimotor-bounded-v1` instead limits every
  eligible existing edge to 0.75--1.25 of its original weight. Reward expectation
  is preserved across diagnostic acquisition/reversal; reward-free tests branch
  separately from training. Previous assays that reset it remain recorded.
- Bounded diagnostic discovery seed 501 (`operant-motor-probe-20260918T043503Z-917333`):
  held-out Up 24.6% -> 38.3%; after reversal Down 46.1%, Up 16.8%.
  Shuffled Up 22.7%, shuffled post-reversal Down 23.8%. Frozen checks pending.
  Independent seed 601 has retained Up 37.5%, post-reversal Down 46.1%; controls
  still running. Must counterbalance the initial target and test visual cues.
- Fresh (NOT operant-pretrained) bounded learning in literal Pokemon application,
  outdoor seed 401: `internal-learn-20260918T043832Z-deaf73` (in progress).
  Starter at decision 547; confirmed first rival/battle win at 1,491, level 6.
  Same game rewards and starting state as frozen control; no route intervention.
  One run is NOT causal gameplay-learning proof. Next: independent seeds and
  frozen evaluations of saved weights against original calibrated weights.
- Broader learning spends substantial time updating 495,962 edge eligibilities.
  Fused CPU implementation matches NumPy bit-for-bit across randomized rounding
  tests and is ~4x faster for that kernel (not a claim of 4x overall game speed).
  Existing running processes retain their original implementation. Exact-resume
  and saved pre-optimization checkpoint comparisons still required.
- Next experiments are visual cue acquisition/reversal with within-cue shuffled
  feedback (two discovery seeds, both mappings), initial-Down motor confirmation,
  and same-checkpoint held-out gameplay. Do not select only successful runs.
- Bounded motor confirmation is complete: seed 601 Up 23.4% -> 37.5%, then
  Down 46.1% (shuffled 27.7%); seed 602 started with Down, 23.8% -> 53.5%,
  then Up 37.1% (shuffled 23.0%). All reward-free tests reset fast dynamics.
  Artifacts: `operant-motor-probe-20260918T043820Z-e303f5` and
  `operant-motor-probe-20260918T044641Z-b6b7aa`. Frozen arms unchanged.
- Completed bounded Pokemon seed 401: starter 547, rival win 1,491, Route 1
  1,882 (map 0x0C; pret/pokered map constants). 221 sampled positions, one
  rival win, level 6 in 6,000 decisions. Seed 402: starter 2,887, 239 positions,
  no battle win, level 5. Both enter play; repeatable learned benefit still pending.
- `scripts/evaluate_saved_gameplay.py` now evaluates the FIRST (preselected)
  Pokemon-trained seed-401 checkpoint against original calibrated weights,
  both frozen, 6,000 decisions each on held-out seeds 701/702/703. Record every
  result, not only wins. This is running; it is not yet a positive conclusion.
- Visual cue results remain mixed. Direct matched-noise input audit
  `motor-vision-probe-20260918T044831Z-55db98` finds screen-dependent changes
  at existing motor inputs, including strong AOTU025 contrast. Per-edge limits
  may restrict association. Testing `sensorimotor-budget-v1`: individual edges
  0.25--4x, but total incoming excitation at every eligible neuron stays within
  +/-25%. This is NOT a button-frequency target. Results are not yet sufficient.
- Additional credit-assignment hypothesis `sensorimotor-perturb-v1` uses the
  actual existing independent neural-noise events (centered against known noise
  probability), rather than treating all firing above a global mean as causal.
  Synapses use prior presynaptic activity and same-step perturbation, no action
  label or new input/connection/critic. Finite-noise approximation inspired by
  https://doi.org/10.1103/PhysRevLett.97.048104, not a physiological validation
  or exact reproduction. Isolated operant screen is running before gameplay use.
- The launcher now defaults to the movement-calibrated, bounded internal rule.
  It supports `-LoadState`; old checkpoints keep their original profiles.
  All later candidates stay opt-in. Fresh setup can reproduce neutral calibration
  without the ROM; it refuses to overwrite an existing calibration artifact.
- Application persistence and real live-stream/speed smoke checks passed for
  bounded mode. Unit verification: 178 Python, 20 JS, lint passing; full-profile
  live sweep pending. Browser connection remains unavailable despite retries.
- A literal launch THROUGH `scripts/start.ps1` is running from the bedroom with
  the new default, live dashboard 8778, 6,000 decisions (launcher default seed 64,
  so do not mislabel it as a matched seed-403 comparison).

## Objective

Improve the pixel-driven fly's ability to participate in Pokemon, especially its
persistent weak forward/Up activity, and determine experimentally whether internal
reward-driven plasticity produces useful, retained changes in behavior. Playing,
not mastering or following a prescribed route, remains the goal.

The user has stopped their run and authorized independent implementation and
experiments using the available compute. Their existing run/checkpoints must be
preserved. A negative result is a valid result, not permission to fabricate success.

## Non-negotiable boundaries

- Legal root ROM remains unchanged and Git-ignored; never distribute it.
- Raw screen pixels only for sensory/action input. RAM only detects rewards and
  supplies evaluation measurements. No semantic vision or RAM-driven decisions.
- Learning stays on existing internal fly connections. No trained encoder,
  external policy/critic/readout, forced button, or random button policy.
- Keep the fixed movement-to-button mapping and current general game rewards.
  No Up bonus, route/house-exit reward, revisit penalty, or new reward categories.
- Preserve original connectome files and the stopped user's run. New versions
  and experiments have explicit identities; old checkpoints resume unchanged.
- Any stimulation, ablation, synthetic reward, scripted reset or visual test is
  labeled as an isolated diagnostic, never passed off as autonomous gameplay.
- A changed weight, a changed action, and improved learned behavior are different
  claims. Use matched controls and held-out seeds before claiming improvement.

## Work sequence

### 0. Preserve and establish the baseline

- [x] Check that no user simulation is still running; do not kill user processes.
- [x] Record protected ROM/connectome/checkpoint hashes and the stopped run's
  final summary. Keep the completed generation immutable.
- [x] Inspect current code/tests and record baseline motor/learning evidence.

### 1. Versioned continuous raw-vision timing

- [x] Add an opt-in temporal-vision mode that interleaves existing neural updates
  with fresh raw screenshots during the same button pulse. Define causal ordering,
  reward timing, initialization and checkpoint boundaries explicitly first.
- [x] Keep neural timestep, neural updates per decision, game frames per decision,
  buttons, reward amounts and base model fixed for the timing comparison.
- [x] Preserve the old snapshot mode and exact legacy resume. Display/log the
  actual input timing honestly; do not label a video window as one static input.
- [x] Unit-test stepping/release/error paths and exact stream-mode resume; run
  matched frozen snapshot/stream trials before interpreting a learning effect.

### 2. Forward-circuit diagnosis and evidence-based candidate changes

- [x] Measure signed excitation/inhibition, voltages, spikes and decoder losses
  for all four directions, not only Up, under blank, static and moving raw input.
- [x] Compare original versus retained learned weights with matched input/noise.
- [x] Isolate the responsible drive with narrowly scoped diagnostic ablations.
  Test model changes only when those measurements motivate them; no arbitrary
  Up gain, equal-button quota or removal of biologically valid braking by fiat.
- [x] Separate discovery seeds from held-out seeds; report unchanged/negative
  effects and impacts on visual sensitivity and other motor populations.

### 3. Internal dopamine/plasticity improvement and behavioral test

- [x] Inspect available anatomical annotations and primary learning literature.
  Identify which compartment/feedback assignments can actually be supported;
  do not invent missing anatomy or call an approximation a complete fly model.
- [x] Implement a separately versioned, bounded internal-learning candidate if
  supported, including appropriate reward-expectation feedback where feasible.
  Preserve original weights/signs/sparsity and the current rule as a control.
- [x] Run a separate controlled visual-choice/conditioning assay using the fixed
  neural decoder: pretest, learning, reward-free retention, and reversal.
- [x] Include frozen and reward-timing controls, matched starting states, fresh
  held-out noise, save/reload checks and explicit sample sizes. Behavioral choice
  specificity, not merely weight magnitude, is the relevant learning result.
- [x] If learning does not reliably alter choices, report that limitation and do
  not promote the candidate or compensate with a hidden external learner.

### 4. Matched Pokemon experiments and handoff

- [x] Compare baseline and justified candidates one factor at a time, then any
  supported combination, from identical game states with multiple matched seeds.
- [x] Include bedroom and the user's stopped outdoor state, with reset/setup
  explicitly recorded as an intervention. Never reward a test location or route.
- [x] Measure directional access, corner/boundary dwell, image-sensitive activity,
  unique exploration, interactions/battles if encountered, and retained behavior.
  More Up or tiles alone does not establish reward learning.
- [x] Carry learned weights across explicit practice/retention trials where useful;
  do not erase the user's brain or silently convert its checkpoint identity.
- [x] Run Python/JS tests, lint/format and proportionate ROM/CUDA/resume smoke
  checks. Recheck protected hashes and ROM ignore status. Stop owned diagnostics.
- [x] Update RESULTS.md, MODEL_VARIANTS.md and usage docs with exact commands,
  artifacts, what improved, what failed, and remaining uncertainty. Only promote
  defaults with evidence; leave inconclusive candidates opt-in.

## Initial experiment budget and decision rules

Use short circuit screens first (at least three seeds), followed by independently
seeded confirmation for promising candidates. Start matched gameplay comparisons
at 1,000 decisions per condition/seed and extend only informative comparisons.
Do not search parameters until a favorable game route appears. Record protocol
changes before rerunning; distinguish exploratory from held-out results.

Success for this work is a reproducible implementation and an honest experimental
answer about useful vision, forward recruitment and retained internal learning.
Do not claim the fly learned Pokemon unless controlled behavioral evidence exists.

## Progress log

- Plan written before implementation. Existing evidence: Up about 4.1% versus
  Down 25.7% in a recent 10,000-decision window; forward silent in 92.3% of those
  windows. Existing sensory isolation improved access but did not resolve bias.
  Previous image-association tests did not establish reliable learning.
- Next: preserve the stopped checkpoint, define temporal input ordering, and
  inspect tests and available neural metadata before choosing model candidates.
- Protected manifest and experiment protocol:
  `runs/neural-improvement-20260918T020816Z-3605f1/`. User stopped cleanly at
  decision 68,811; final checkpoint `step-00068811-adf1b9de`. No user Python
  simulation remained. ROM ignore rule confirmed.
- Temporal protocol: new stream/endpoint modes share a one-window static
  initialization, choose from the preceding neural window, and integrate 12
  neural steps per 24-frame pulse. Stream observes offsets 2,4,...,24; endpoint
  repeats the final frame. Both apply accumulated rewards after the new neural
  window. The endpoint control isolates fresh-image effects from changed
  eligibility timing. Snapshot-v1 is unchanged. Pending counts/retinal drive and
  input-window hashes are checkpointed for exact resume. UI shows the last actual
  input image and explicitly labels activity as spanning a window.
- Temporal tests: 18 frozen gameplay trials completed (three seeds, two starting
  states, three input modes), 1,000 decisions each. Snapshot and endpoint choices
  match exactly. Stream supplies changing frames but does not consistently raise
  Up (270 versus 260 pulses across 6,000 decisions); remains opt-in.
- Forward audit: 39 frozen conditions, 1,536 steps each. User's learned weights
  produce exactly the same motor-count windows and buttons as original weights
  on the fixed outdoor screen for three seeds. Remaining Up inhibition is
  distributed; CB0890 is the largest typed contributor in the inspected case.
  Removing all Up inhibition is a diagnostic capacity control, not a fix.
- Quiescent unused-sensory boundary is implemented only as an explicit candidate.
  It does not consistently improve forward activity in circuit screening and is
  not promoted. No user data or original connection was changed.
- Learning candidate: `compartment-ema-v1`, partial type-level gamma5/alpha1/
  gamma1pedc mapping from Li et al. (2020), existing shared DAN targets only.
  Fine synaptic compartment positions and DAN subtypes are unavailable locally;
  this is not a resolved biological model. Internal reward EMA uses a fixed 30 s
  neural-time constant; no external critic or action features are introduced.
- Controlled choice assay is running: 256 training decisions, two raw visual
  cues, two counterbalanced Left/Right contingencies, three seeds, paired,
  within-cue reward-permuted and frozen arms, two held-out test noise streams,
  reward-free retention, reversal and on-disk weight reload. Current rule and
  candidate tested separately. The >=5 percentage-point all-case screen is
  exploratory, not a significance test. No synthetic assay reward enters Pokemon.
- Snapshot API hardened: snapshots now own every plasticity array. A conservative
  assay restart followed discovery of shared references. Further review shows the
  harness's pre-arm restore already detached its baseline arrays, so contamination
  was a potential API hazard, not an established defect in those completed arms.
  Superseded partial artifacts are retained in `invalid-shared-snapshot-assay/`
  (initial conservative label) and `superseded-assay-tail/`; do not report them as
  final results. The authoritative rerun is in `learning-v2/`. Frozen controls
  assert original weights at each arm. Disk reload repeats the first 48-decision
  held-out choice-count probe; all behavioral scores still use both held-out noise
  streams. The assay's reload equality checks action histograms; the separate
  live resume tests compare full trajectories and saved neural arrays.
- Retained-weight gameplay completed: same outdoor state, seeds 101/102/103,
  1,000 frozen decisions each. Original versus retained distinct positions are
  11/14, 15/13, 13/12. Actions differ in 580/559/359 windows, establishing a
  trajectory effect without consistent exploration improvement. Retained runs
  spend 86.0-88.0% of decisions in the bottom two town rows. A real old-checkpoint
  continuation matches the user's recorded decisions 68,501-68,520 exactly
  except run ID and wall-clock timings.
- No forward candidate justified a combined-profile test or held-out confirmation;
  negative screens are retained, not parameter-tuned until a favorable route appears.
- Integration verification passed: 159 Python tests, 20 JavaScript tests, lint,
  formatting, seven existing live CUDA/ROM checks and five additional candidate
  checks. Stream, quiescent and partial-compartment exact resume pass. The new
  -Weights launcher passed a two-decision frozen new-game test using the user's
  saved connections. Full source-file hash recheck remains for final handoff.
- Browser skill connection retried twice; in-app browser remains unavailable.
  UI logic and actual HTTP/pixel checks pass, but rendered visual QA is unverified.
- All 36 learning arms completed. Both rules pass 0/6 seed/mapping cases; neither
  advances to independent confirmation. Mean acquisition change is +0.09 points
  versus pretest for both, but advantage over unpaired is +0.52 points for the
  old rule and -0.17 for the candidate. Reversal does not improve over unpaired
  in any case. All 72 saved stage histograms reproduce after reload, and frozen
  weights remain unchanged. Weight updates work; useful learned choices are not
  established. No new default or game reward is promoted.
- Final protected-file recheck: all 10 manifest entries match, including the
  source trajectory, final checkpoint, ROM and original connectome. ROM, model
  and research artifacts remain Git-ignored. The live verification script now
  includes all four candidate profiles, with final expanded-suite cleanup pending.
- Final expanded suite completed successfully: 159 Python tests, 20 JavaScript
  tests, lint/format and 15 live ROM/CUDA checks. All owned experiment processes
  exited; a read-only process check found none remaining. All protected hashes
  were rechecked after the last live test. Consolidated evidence and result-file
  hashes are in `runs/neural-improvement-20260918T020816Z-3605f1/verification.json`.
  README, RESULTS, MODEL_VARIANTS, EXPERIMENT and PROJECT_BRIEF are updated.

## Handoff

Implemented: opt-in temporal raw input with exact checkpoints, quiescent sensory
and partial-compartment learning candidates, reproducible controlled assays,
independent in-memory snapshots, explicit launcher weights reuse, and regression
coverage. Completed 21 matched 1,000-decision gameplay trials, 39 circuit
conditions, and 36 learning arms. No route shaping or external learner was added.

Working: pixel-sensitive activity, delivered buttons, real internal weight
changes, saved-brain trajectory effects, exact continuation and weight reload.
Not established: consistently stronger Up, beneficial retained exploration,
reward-specific choices, or reversal. No candidate met the promotion screen.

Recommended next research: calibrate the visual-memory-to-motor pathway and
validate retained cue-specific behavior before treating longer Pokemon runs as
evidence of learning. This recommendation is not an unperformed requirement of
the completed experiment budget. Browser-rendered visual QA remains unavailable.
