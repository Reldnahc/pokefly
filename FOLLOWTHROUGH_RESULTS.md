# Follow-through: movement, internal learning and actual gameplay

## Reward delivery and evaluation checks, 2026-09-19 00:23 UTC

A replay from the beginning of v7 learning 402 matches every recorded state
and reward across all 18,000 decisions. Its victory reward is delivered at
the final knockout (decision 12,523), 26 decisions before encounter closure.
This verifies delivery timing, not move-selection credit or useful learning.
It is playback of the existing victory, not a new win or a battle-start
practice episode. Evidence: `recorded-outcome-latency-20260919T000938Z-c898dc`.

The next retained-weight panel's audit includes time to first sampled battle
and distinguishes paid from completed encounters. It excludes unfinished
encounters from aggregate win counts and rejects non-fly actions or reward
ledger mismatches. These measurements never enter the controller or rewards.
The two ongoing whole-game practice runs and their frozen tests are not yet
complete. Defaults remain unchanged.

CPU verification passes 406 Python tests, 26 JavaScript tests and lint:
`verify-6d09e3b1a6724df39d729b0af4608008`. CUDA tests were omitted to preserve
the two-job compute limit; the earlier full CUDA-inclusive result is below.

## Longer whole-game comparisons, 2026-09-19 00:02 UTC

Both earlier-outcome v7 pairs completed18,000 decisions per arm. All begin
with original synapses and the same explicit intro; no stage reset or trained
adapter. Results are mixed, not a demonstrated consistent learning advantage:

| Seed / mode | Positions | First starter | Completed rival wins | Route1 minimum y |
| --- | --- | --- | --- | --- |
|401 learning |346|3122|0|34|
|401 original frozen |395|6284|0|26|
|402 learning |413|10936|1 (decision12523)|not reached|
|402 original frozen |313|7780|1 (decision8955)|not reached|

The earlier starter in401 does not extend to farther exploration; the original
reaches Route1 earlier and goes farther north.402 learning explores more and
spends less of its town time at the lower edge (3.26% vs26.43%), but acquires
its starter and victory later than its original control. Both learning and
original controls therefore have2/2 starters and one rival win across two
runs. No captures, badges, or paid-but-unfinished encounters. Progress and
one actually delivered early training victory reward are real; reliable useful
whole-game learning still requires retained tests. This does not isolate
reward-timing improvement against late-timing learning at18k.

Audit: `full-game-pair-audit-20260919T000100Z-980a15`, verifying complete raw
trajectories, checkpoints, original control weights, configuration, ROM and
matching start states. Pair reports `visual-model-gameplay-20260918T230017Z-c8f22a`
and `...T230049Z-fade7a`. Each original6k development prefix exactly repeats
its earlier records apart from wall timing/run ID; those are not new trials.

Both final trained brains now continue through one more18k whole-game attempt,
retaining learned synapses with unchanged v7 parameters (noise4101/4102).
The registered subsequent frozen3801/3802 panels use both final brains and one
shared original per seed,18k per arm. Training and evaluation are not complete
yet. No stage-specific or battle-start episodes will be used.

Software verification:408 Python/26JavaScript/lint pass
(`verify-287e071c85d64e1b9cd0857b23a0b57c`). Optional homeostaticv8 and unchanged
showcasewide-v3 both pass actual-ROM/CUDA exact-resume/frozen/no-reward smoke
(`internal-smoke-20260919T000017Z-24f7dd`, `...T000050Z-6bd19f`). The new
constraint remains experimental; it has numerical checks, NOT a behavioral
learning result. No default promotion and no rendered browser QA are claimed.

## Learning-dose result and next full-game tests, 2026-09-18 23:19 UTC

Both projected10x-rate trials completed6,000 decisions and FAILED the declared
retention-priority criterion: no starter, battle victory or Route1 on either
401 or402. Positions220/251, with B in81.52%/43.20% of windows. The slower
projected parent acquired a starter on401 and covered185/290 positions. These
are not successful improvements, and the candidate is not promoted. Reports
`visual-learning-rate-gameplay-20260918T224350Z-e6e62f` and
`visual-learning-rate-gameplay-20260918T224450Z-a47070`.

The timing-only wide-v7 model is now in two18,000-decision learning/frozen
full-game comparisons, reports `visual-model-gameplay-20260918T230017Z-c8f22a`
and `...T230049Z-fade7a`. Same original synapses, intro, physical brain, pixels,
buttons and reward values; only the existing battle-outcome delivery hook
differs from the showcase default. No final improvement result yet. Shared
6k development prefixes are not extra independent trials. A new audit reports
paid early outcomes separately from fully completed encounters; it does not
count a rival bonus as a second battle or claim that open dialogue was cleared.

The user confirmed whole-game-only practice with learned synapses carried
between games. No future battle-start episodes: learning to reach the battle
again is part of the task. Historical battle-reset results remain archived,
not a proposal for the current curriculum. Original/frozen arms are controls,
not the normal training mode. The default showcase remains wide-v3.

CPU-only verification passes373 Python/26 JavaScript/lint, explicitly omitting
ten CUDA kernel tests while both GPU slots are occupied:
`verify-aaae4a787817449fa2f7256f97ad09c8`.
## Independent practice follow-up, 2026-09-18 22:44 UTC

The promising starter-acquisition advantage below **did not replicate** on
the two prospectively selected new seeds. All six arms completed 6,000
decisions with weights frozen and identical fresh-game starts:

| Final internal weights | Seed 3701 | Seed 3702 |
| --- | --- | --- |
| Original shared control | 278 positions; starter 2114 | 246 positions; no starter |
| Practice source 401 (final episode 1402) | 292 positions; no starter | 265 positions; no starter |
| Practice source 402 (final episode 1404) | 254 positions; no starter | 295 positions; no starter |

No wins or Route 1 in any arm. Both practice brains are included; no selected
checkpoint or training source is substituted. Each original control counts
once. The higher coverage of some trained arms is not better Pokemon play:
retained starter acquisition is 0/4 versus original 1/2 in this independent
follow-up. The earlier rival victory is genuine, but these completed results
do not establish reliable useful gameplay learning. Practice brains are NOT
promoted to the showcase default.

Audit: `game-retention-panel-audit-20260918T224333Z-7384d6`. Raw panels:
`game-retention-panel-20260918T221429Z-8b795c` and
`game-retention-panel-20260918T221440Z-d732f1`. Audit verifies actual frozen
synapses, every outcome, matching model/ROM/start state and both final sources.
Recorded playback of the earlier victory independently matches all 6,000
states/rewards: `recorded-replay-proof-20260918T222416Z-7bb39d`; frames 4206
and 4216 show level 6 and the defeated rival. This is playback, not another win.

Experiments continue with the preregistered projected 10x-rate candidate.
Separately, replay verifies that the existing earlier-outcome hook moves the
same rewards 32/45 decisions earlier on the two actual victories, without new
rewards or changed novelty (`reward-timing-audit-20260918T222844Z-22372c`).
The wide-v7 timing-only profile is prepared for a longer full-game comparison;
neither candidate is the showcase default. Full verification now passes
377 Python/26 JavaScript tests and lint (`verify-b0c0236d5d7d450b8cfc958d95886b0f`).

## Learning follow-through, 2026-09-18 22:14 UTC

After three explicitly reset actual-game learning episodes (18,000 decisions
per independent brain), freeze ALL learning and start fresh games from the
same scripted intro. The complete audited results are:

| Final internal weights | Seed 1601 | Seed 1602 |
| --- | --- | --- |
| Original shared control | 250 positions; no starter | 349 positions; no starter |
| Practice source 401 | 221 positions; no starter | 275 positions; starter 3238, no win |
| Practice source 402 | 173 positions; starter 4598, no win | 251 positions; starter 3533, one rival win, level 6 |

Thus retained brains obtained a starter in 3/4 tests versus 0/2 original
controls, and one retained brain won one encounter. The duplicated original
controls reproduce ALL 6,000 records exactly apart from run ID/wall timing
and are counted once per seed. No arm reached Route 1. This is promising
retained behavior, **not yet a replicated general gameplay-learning result**:
only two evaluation seeds, lower coverage in every retained arm, and source 402
had no training battle victory. Its test victory cannot be attributed
specifically to having received battle-win reinforcement during training.

Audit: `runs/retained-game-series-audit-20260918T221314Z-fe4d56`.
Practice panels: `resumed-game-series-20260918T212058Z-14a067` and
`resumed-game-series-20260918T212058Z-5e8d18`. Source weights, exact model,
matched start states, frozen zero-update records and every evaluation outcome
were verified. No synthetic weights or action features entered these games.
Independent follow-up 3701/3702 is now preregistered for BOTH final brains,
with one original control per seed; the showcase default remains unchanged.

The neutral probe (`game-motor-drift-probe-20260918T221128Z-a4b169`) records
B in 219/384 windows (57.03%) for source 401, 91/384 (23.70%) for source 402,
versus original 110/384 (28.65%). All seven anatomical groups were observed;
weights/rewards off, same 3201..3203 noise, 128 warmup + 128 scored windows.
The repeated original exactly matches previous probes and is shared, not a
new independent control. Tonic changes are heterogeneous, and are not by
themselves proof of contextual improvement or universally harmful drift.

Verification: 369 Python/26 JavaScript tests and lint pass, artifact
`runs/verify-23623b86a8cd4d43802a810829911a36`. The queued fast projected
candidate also passes real-ROM learn/resume/frozen/no-reward smoke
`internal-smoke-20260918T221315Z-c31c77`. These are implementation checks,
not behavioral evidence or rendered-browser QA.

## Learning follow-through, 2026-09-18 21:21 UTC

Projected-wide's final independent visual-retention gate PASSES for both
training seeds after all shuffled controls completed:

| Training seed | Original balanced | Reward-paired balanced | Shuffled balanced | Paired left/right |
| --- | --- | --- | --- | --- |
| 501 | 40.70% | 71.45% | 36.44% | 72.11% / 70.78% |
| 601 | 40.70% shared | 68.62% | 39.17% | 70.28% / 66.97% |

This uses the final 8,192-decision synapses, fresh noise 3001..3004, a 128-window
neutral warmup and 128 scored windows per cue/seed. Learning and rewards are
off. The unchanged gate requires balanced accuracy >=60%, each cue >=55%,
and >=5 points over BOTH controls for BOTH training seeds. These are descriptive
engineering criteria, not a significance test. Accuracy is conditional on the
competing Left/Right choices; correct targets occurred in 38.96%/41.21% of ALL
windows. Both original panels match exactly and count as one shared control.
This supports retained two-cue association while reducing tonic motor drift,
NOT learned Pokemon navigation. Synthetic-trained weights never enter the game.
Audit: `visual-retention-gate-20260918T212057Z-3b0724`; raw panels
`visual-retention-probe-20260918T211130Z-b17f16` / `...-ccbda7`.

Both causal-credit runs completed 6,000 decisions: **210/128 sampled positions,
no starter**, first house exits 359/495. Both underperformed wide-v3 learning
(287/255 positions, starter at 3,122 in seed 401) and original frozen controls
(268/188 positions, no starters). The candidate is NOT promoted.
Artifacts `visual-credit-timing-gameplay-20260918T204817Z-b4a1c9` and
`visual-credit-timing-gameplay-20260918T205918Z-821dab`.

The original repeated-practice studies can now resume without discarding their
training. `resume_game_series.py` validates the complete weight chain, model,
ROM and original seed/budget schedule; copies previous evidence as explicitly
shared, pins the interrupted checkpoint, checks overlapping recorded decisions,
and then finishes the original fresh-start/frozen-weight test schedule. Both
actual step-4,500 sources passed read-only preflight and are now running:
`resumed-game-series-20260918T212058Z-14a067` / `...-5e8d18`. These are the only
two GPU jobs. They retain the original practice/evaluation schedule; results
are pending, not positive evidence yet.

An exploratory read-only dose check illustrates a possible transfer gap, NOT
a causal finding. Wide-v3's two successful 8,192-decision synthetic paired
assays received sums of `tanh(reward)` of 2,800.38/2,567.33; its two 6,000-decision
actual-game training runs received 17.16/15.57. These are about 163/165 times
different in total, or 119/121 times per decision. Cue structure, contingency,
reward frequency and magnitude all differ, so this does not isolate a gain
effect or establish that stronger feedback will help. No reward has changed.

Verification: 359 Python tests, 26 JS tests and lint pass, artifact
`verify-996b60e4725e46da89472a486488b83a`; all 10 protected file hashes last
matched at 21:05. One earlier full-suite attempt hit a Windows HTTP connection
abort in an unchanged rejected-control request. All 30 dashboard tests then
passed, followed by the full rerun; no guard or assertion was bypassed.
The ROM, brain data and run artifacts remain Git-ignored and untracked.

Work is ongoing. This supersedes the earlier "no improvement" conclusion for
movement and basic motor learning, but does not claim Pokemon completion or
established screen-specific gameplay learning. Full protocol and failed
candidates remain in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).

## Learning follow-through, 2026-09-18 20:52 UTC

Both interrupted projected-wide actual-game candidates were completed from
their exact step1000 checkpoints to total6000. All20 overlapping decisions
match each original log exactly; old prefix plus new suffix count as ONE trial.
Artifacts `resumed-rule-gameplay-20260918T203139Z-ce3fff` (401) and `...-c0dd8a`
(402). Previously completed controls are reused explicitly, not new trials.

| Seed | Original frozen tiles | Default wide-v3 learning | Projected-v4 learning |
|---|---:|---:|---:|
| 401 | 268 | 287; starter3,122 | 185; starter5,767 |
| 402 | 188 | 255; no starter | 290; no starter |

No condition wins a battle or visits Route1 in these6000-decision comparisons.
Projection is not a demonstrated gameplay improvement: coverage changes in
opposite directions, and there is one starter in each learning condition.

The same candidate DOES reduce retained neutral-screen B bias. Frozen saved
game synapses, uniform RGB128, all three preselected3201..3203 noise streams,
128 warmup+128 scored decisions per stream, all seven motor groups measured:

| Source | Default-wide B selections / Hz | Projected-wide B selections / Hz |
|---|---:|---:|
| Original shared control | 28.65% / 0.928 | identical observations, counted once |
| Game401 final6000 | 52.34% / 1.752 | 32.03% / 0.939 |
| Game402 final6000 | 36.20% / 1.090 | 29.17% / 1.009 |

Artifact `game-motor-drift-probe-20260918T204118Z-6cdcbb`; original observations
match the earlier control across ALL recorded fields. The original count is
110/384=28.65%; the earlier28.13% transcription has been corrected. This is a
mechanism diagnostic, not a claim of useful visual strategy. The CPU-only
`credit-update-audit-20260918T204115Z-a0e2b4` finds projected mean-input update
residuals at rounding scale and no active weight bounds in these final game
snapshots. Hypothetical updates do not modify or export the source weights.

A SEPARATE timing candidate is now being tested. `visual-causal-credit-v5`
changes only which internal eligibility trace receives the unchanged reward:
the trace present when the button was chosen, before the next streamed neural
window. That next window still runs normally and drives the next choice. It
cannot change the button already being held, so excluding its new innovations
is a causal-alignment/variance hypothesis, not a proven learning improvement.
No action identity, RAM state or fitted decoder enters this neural trace.
The two actual-game seeds401/402 and6000 budget were preselected;401 is running.
Projected visual-acquisition controls are also being recovered from saved full
states; final independent3001..3004 retention is still required.

Implementation verification:345 Python/26 JS tests and lint pass
(`verify-0f8757f21894491899bc7ebb301f1ec5`), with real-ROM learn, exact-resume,
frozen and no-reward checks (`internal-smoke-20260918T204550Z-141776`). The
causal candidate's frozen70-decision run exactly matches the older default's
pixels, neurons, actions, game/rewards and ALL final neural arrays; only the
intended learning-timing label differs. These are implementation checks, not
additional learning trials. The showcase default remains wide-v3. At most
two GPU jobs are allowed; the prior eleven-job queue is not relaunched.

## Showcase default update, 2026-09-18 17:59 UTC

At the user's explicit request, fresh `scripts/start.ps1 -Intro` now selects
`visual-release-wide-v3`: the stronger frozen visual circuit, streaming raw
frames, perturbation-based internal learning and serial button delivery. This
is the model that passed the independent retained two-cue test below, not a
claim of established Pokemon progression. The unfinished projected-credit
candidate remains opt-in. Rewards, fixed anatomical button mapping, original
weights/data and existing saved profiles are unchanged. No assay-trained or
selected successful gameplay weights are silently loaded into fresh runs.

Bounded implementation verification (not new gameplay-performance evidence):

- Full suite:332 Python tests,26 JavaScript tests and lint pass;
  `verify-52d08aeba83c45ffaa4f7cf7c9afa7e1`.
- Literal default PowerShell launcher:80 decisions, correct recorded model,
  streamed/serial timing and consistent final checkpoint;
  `internal-learn-20260918T175722Z-1562c7`.
- Real-game dashboard stream:50 decisions; pixels/retina/actions agree,
  graded activity remains separate from spikes, reward totals match logs,
  live speed changes work and human buttons remain disabled;
  `internal-learn-20260918T175752Z-e3bf38/smoke.json`.
- Learning, exact20-decision resume, frozen and no-reward controls all pass;
  `internal-smoke-20260918T175833Z-78448d`.
- All10 protected ROM/model/checkpoint hashes match. All verification workers
  exited; no Pokefly Python job was left running. The browser connection is
  still unavailable, so rendered desktop-layout QA remains unverified.

## Research evidence recorded at 2026-09-18 17:30 UTC

The following "running" references describe the historical state before the
17:38 process cleanup; interrupted experiment panels are not completed results.

The completed neutral-screen diagnostic finds a retained GENERAL B bias after
Pokemon training, especially in source401. These are frozen saved game weights,
RGB128 only, three fresh noise streams,128 warmup+128 scored decisions each:

| Synapses | Mean B-neuron rate | B selections over384 decisions |
|---|---:|---:|
| Original shared control | 0.928Hz | 28.65% |
| Final6k game401 | 1.752Hz | 52.34% |
| Final6k game402 | 1.090Hz | 36.20% |

All seven motor groups were measured. No cell was retuned and no weights
changed in this diagnostic (`game-motor-drift-probe-20260918T172152Z-e9332d`).
This separates tonic drift from a purely menu-evoked response; it does not
prove drift is the sole gameplay bottleneck. The existing mean-input-preserving
rule is now being tested in matched6k actual games on the stronger circuit,
both401/402, using only original synapses and unchanged rewards. The unchanged
frozen controls are explicitly reused after exact64-decision prefix checks.
It is an exploratory rule test, not model promotion or new independent controls.

Exact replay of the four wide-v3 game arms also measures Start/nested-menu
fractions: frozen/learning36.09/25.55% (401),37.86/34.93% (402). All24,000
recorded states/rewards match. Less menu time is not automatically screen-aware
menu competence; increased B can also cause it. These replays are not new play.

The mean-preserving rule on the WEAKER visual-v1 circuit failed its completed
retention gate: paired56.61/58.28%, original48.84%, shuffled52.35/51.11%;
right-cue42.93/50.25%. Artifact `visual-retention-gate-20260918T172144Z-93513c`.
This negative result is retained; the separately registered stronger-signal
combination has not yet completed its independent controls.

### Actual-game pairs recorded at17:13 UTC

The stronger visual model's two6,000-decision actual-game pairs are complete:

| Seed | Frozen / learning tiles | Frozen / learning Up | Starter, frozen / learning | Wins |
|---|---:|---:|---:|---:|
| 401 | 268 / 287 | 21.55% / 22.05% | no / yes, at3,122 | none |
| 402 | 188 / 255 | 17.17% / 23.87% | no / no | none |

Both learning runs explore more than their matched frozen controls, but none
visits Route1 or wins a battle. This is encouraging short-budget exploration,
not yet learned navigation or game progression. Both final actual-game brains
are undergoing the registered repeated-practice/frozen-transfer protocol.
The weaker v1 source402's completed fresh-start test does NOT show improvement:
original/retained tiles199/159 and266/185, one starter in each condition over
two seeds, no wins. Its favorable first starter result was not a general gain.

### Confirmed visual-association result at16:59 UTC

**The stronger visual circuit passes the preselected retained two-cue learning
test for both independently trained brains.** Final8,192 synapses are tested
with learning/rewards off after neutral warmup and on fresh noise seeds:

| Training seed | Original | Paired reward | Shuffled reward | Paired left / right cue |
|---|---:|---:|---:|---:|
| 501 | 44.72% | 69.10% | 36.37% | 56.43% / 81.77% |
| 601 | 44.72% | 68.80% | 36.82% | 59.34% / 78.26% |

All preselected conditions pass: balanced accuracy>=60%, BOTH cues>=55%, and
at least5 percentage points above EACH control in BOTH training seeds.
These are descriptive engineering thresholds, not a statistical significance
test. Each panel has1,024 scored decisions over noise seeds2801..2804; repeated
original controls are a reproducibility check, not extra independent samples.
The raw-choice/provenance audit is `visual-retention-gate-20260918T165919Z-e1b7c7`.
It validates the final synaptic hashes and includes every scored time window.

This is evidence of a retained association between the two tested pixel cues
and fixed neural motor choices. It does NOT establish general visual reasoning,
learned Pokemon navigation or biological fidelity. No assay-trained weight is
loaded into Pokemon. Registered same-acquired-start reversal is now running
for both sources, with independent3101..3104 tests afterward. Actual-game
practice/evaluation continues separately; no experimental model is promoted yet.
Full verification after the gate-audit addition:328 Python/26 JS plus lint
(`verify-294136a72b69487a87c68540cf2cafea`).

### Actual-game and numerical results recorded at16:50 UTC

Both longer v1 pairs finished24,000 decisions per arm:

| Seed | Frozen / learned tiles | Starter, frozen / learned | Battle wins, frozen / learned | Northernmost Route1 y, frozen / learned |
|---|---:|---:|---:|---:|
| 401 | 352 / 351 | 11,593 / 8,789 | 1 / 0 | 29 / no visit |
| 402 | 348 / 297 | 7,262 / 3,748 | 0 / 0 | 25 / 28 |

Earlier starters did not translate into better overall progress. Do not promote
v1 as a learned-navigation fix. Its final saved brains are both undergoing
the registered frozen fresh-start comparisons, not selected best checkpoints.
The single frozen victory is the rival encounter: `battle_win` includes
`rival_win`; these counters must not be added as separate wins.

Wide-v3's first6,000-decision pair completed: learning287tiles/starter3,122
versus frozen268tiles/no starter. Neither won or visited Route1. Source402's
pair is still running. The registered repeated-game practice from source401
has begun; source402 will follow, each with two fresh attempts and a final
frozen original/retained panel. Only actual-game synapses carry across resets.

Numerical speedup: calibrated-visual CUDA steps now skip current sums that
the existing equations discard. Original graph/weights and diagnostic current
measurements remain unchanged. Two64-decision ABBA comparisons match ALL
arrays, counts, actions and metadata, with7.6..14% less time under shared GPU
load. Actual v1/wide-v3 resumes reproduce original recorded decisions5001..5020
exactly and pass split-resume checks. Artifacts
`internal-propagation-benchmark-20260918T164408Z-657c5e` / `...T164726Z-ba9aae`,
`checkpoint-window-verification-20260918T164752Z-ef7a24` / `...T164837Z-c489f3`.
Full suite:323 Python/26 JS and lint pass (`verify-2fbcdadd28be44ae9adfa5d42798323e`).

### Evidence recorded at16:36 UTC

The stronger visual circuit has a promising independent retention result:
wide-v3 seed501's saved final synapses score **69.10% balanced**, versus
44.72% original, after a fresh neutral warmup with rewards and learning off.
Both cues exceed the preselected 55% threshold (56.43% left, 81.77% right).
This completes only the original/paired portion of the test; the shuffled
reward control and second training seed must also pass. It is not yet a full
gate pass or proof of learned Pokemon progress. Artifact:
`visual-retention-probe-20260918T162138Z-e22d73`, independent seeds2801..2804,
1,024 scored decisions per arm. Second-source panel is now running.

The verified serial command-delivery fix is now the fresh-launch default via
`sensorimotor-bounded-serial-v2`. This changes ONLY transport, not the default
brain, rewards, decoder or frame budget. The old profile and all saved-run
settings remain unchanged. A real PowerShell default launch delivered all
paired requests in the expected phases (`internal-learn-20260918T162141Z-dc6a70`);
learn/exact-resume/frozen/no-reward smoke also passes
(`internal-smoke-20260918T162148Z-7f47ac`). No experimental visual/learning model
has been promoted. This is a delivery fix, not a learning-performance claim.

Both independent v1 reversal panels are complete on noise seeds 2701..2704:
paired 56.30% / 56.48%, acquired baseline 41.35% / 41.91%, shuffled 44.22% /
47.81%. There is retained contingency-following change, but both still FAIL
the unchanged balanced 60% / each-cue 55% gate. One cue is 45.68% / 38.41%.
Artifacts `visual-retention-probe-20260918T155705Z-25c8d5` and `...T155735Z-203df4`.
The v1 final 32,768 small paired probes are 59.18% / 59.20%; more unchanged
training has not produced a clear improvement. Final shuffled/independent
controls remain in progress; no positive longer-training claim is made.

Wide-v3 capacity diagnostic: original 43.05%, temporary oracle forward 72.71%,
reverse 88.88%, both cues above 55%. Those supervised weights were discarded,
never exported or used in Pokemon (`oracle-synaptic-capacity-20260918T154733Z-45c072`).
The actual reward-trained 8,192 small probes reach 66.09% / 72.39%, but one
cue fails in seed501's small probe. The larger independent retention panel
above is the registered gate; shuffled and second-source checks are pending.
Its literal seed401 game obtained a starter at 3,122
and entered the rival battle, without synthetic training weights or scripted
gameplay beyond the disclosed intro. That first learning arm finished with
287 tiles, no battle win and no Route 1 visit. Full frozen/learning comparisons
on BOTH 401/402 are still running; no treatment-effect claim yet. The first
v1 retained-weight game pair (source402, held-out1501) gets a starter at5,416
versus none in the original control, but fewer tiles159 versus199. Other
registered seeds/sources remain pending; this is not a selected success report.

Projected-credit v1 final 8,192 small probes are 59.11% / 59.40%; both cues
exceed 55% but balanced scores remain below 60%. Shuffled and independent
tests are still needed. The separately registered wide+projected combination
changes only credit relative to wide-v3, with all physics/rewards fixed;
it is queued, not yet run or promoted.

Current full verification: 313 Python tests,26 JavaScript tests and lint pass
(`verify-08437da4630c48888a804fbea731fd11`). The default launch uses the same
protected ROM/data and does not migrate any existing checkpoint.

### Display fix and earlier evidence

Fixed a remaining display-only reward issue: the metric previously showed the
instantaneous sample, usually zero. It now displays the server's cumulative
reward ledger (including resumed history), with current and delivered feedback
in a tooltip. It never sums browser packets, so reconnects/skipped frames cannot
lose reward. Old already-running servers are explicitly labeled STEP REWARD.
The brain, reward definitions, and archived trajectory schema are unchanged.
Live real-game pixel/brain/button/speed checks also verify displayed totals
against logged reward prefixes (`internal-learn-20260918T155247Z-05a8eb`).
26 JS tests pass. Browser-skill connection retry at15:53 still fails; rendered
layout remains unverified, independently of passing stream/logic checks.

Two separate hypotheses are now being tested, not combined or promoted:

- Stronger frozen visual release (`visual-release-wide-v3`): both2,048-decision
  interim paired checks show both-cue preference, balanced61.13%/66.36%.
  Final8,192/shuffled controls and independent2801..2804 retention are pending.
  Static/motion untrained choices remain45.0%/46.67%; offline neural stimulus
  information alone is not visual control. Static direct-current contrast/std
  is-0.50/+0.20 at the two steering cells, versus near-zero in v1. No manual
  correction is made to those signs. Actual-game401/402 comparisons are
  preregistered for6,000 decisions per frozen/learning arm, not yet started.
- Mean-input-preserving credit (`visual-projected-learning-v4`): unchanged
  visual-v1 neural dynamics/rewards/button adapter, with a local constraint
  on internal synaptic updates. It tests generic motor-drive drift seen in the
  retained-current audit. Both501/601 paired/shuffled curves are running;
  independent final-weight retention is preselected on2901..2904. Full real-ROM
  smoke passes (`internal-smoke-20260918T154218Z-5d82ee`). Its70-decision frozen
  trajectory AND all checkpoint arrays exactly reproduce the original v1
  frozen control; only the rule label differs. This is implementation control,
  not positive learning evidence.

The v1 seed402 FINAL24,000 synapses are now being tested in frozen fresh starts,
original versus retained on1501/1502 for6,000 decisions each. The seed401 final
brain will undergo the same panel when finished; this is not best-run selection.
Artifact `retained-gameplay-20260918T153624Z-4e6ad6` is still running.

Verification at15:48:310 Python/24 JavaScript tests and lint pass
(`verify-0e8ccfb18008408e920df9295bd3f56e`), plus the subsequent profile-continuation
guard test. All10 protected-file hashes and the original retina still match.
Source/documentation through15:39 were pushed as public commit `e7068cb`.
An additional bit-exact CUDA eligibility benchmark gives only~2% end-to-end
benefit on the current shared GPU; small-graph production dispatch is unchanged
(`internal-eligibility-benchmark-20260918T154527Z-89b51a`).

### Completed negative tests and longer-game controls

The 10x learning-rate candidate completed and FAILED both 6,000-decision
comparisons: 175/111 tiles, no starter, versus v1 learning 213/233 tiles
(one starter) and frozen 188/193. In seed401, B rises to 42.9% and lower-town
dwell to 25.2%. Do not promote it. Artifacts:
`visual-learning-rate-gameplay-20260918T145757Z-5084e4` / `...-ffee91`.

The first 24,000-decision v1 continuation is complete: FROZEN seed401 gets
352 tiles, a starter, ONE rival win, level6, Route1 minimum y29. Correction:
the earlier wording incorrectly also counted a wild win; `battle_win=1` and
`rival_win=1` describe the SAME encounter, not two victories.
This is untrained control capacity, NOT learning evidence. Its matched learning
arm and both seed402 controls are still running. Do not compare different
seeds as a paired treatment effect. The independent menu replay of seed402
learning decisions6001..12000 reproduces all recorded states/rewards and finds
41.89% of emulator frames inside Start/nested menus. Remaining weak contextual
control is measurable (`recorded-menu-audit-20260918T151409Z-603e0d`).

The stronger visual-release calibration failure was traced to its uniform
bias floor: one escape cell hit -0.14. The separately versioned, ROM-free
wide-range calibration uses [-0.5,0.5] for EVERY eligible neuron, with the same
neutral-gray 1 Hz equation, no button/game/reward labels. It reaches stable
0.75..1.20 Hz mapped populations over three independent resets (B0.90..1.16).
Artifact `intrinsic-probe-20260918T152313Z-f5a2bf`, frozen before any evaluation.
Static/motion and paired/shuffled learning tests are registered, not a promotion.
Its actual-ROM learn/resume/frozen/no-reward smoke passes:
`internal-smoke-20260918T153208Z-5db90f`.

A separate standalone fallback-edge unit-scale diagnostic was NEGATIVE/MIXED.
Scaling only unfitted visual edges to0.05 does not reduce overall saturation
and raises LC4 gray activity from0.093/3.94 to2.55/4.78 (left/right), with mixed
motion selectivity. Production remains scale1; no game model was changed by
this probe. Artifacts `calibrated-visual-probe-20260918T151830Z-5ad577` / `...-7dcd60`.

Full suite:307 Python,24 JavaScript tests and lint pass
(`verify-3a89e46df41a4db58008b6902ff2457f`). After the versioned intrinsic loader
refactor, both original-user and new-visual checkpoints match their ORIGINAL
recorded20-decision trajectories as well as independent split resumes:
`checkpoint-window-verification-20260918T152359Z-89da19` / `...-dde349`.

### Acquisition and matched-play evidence through 15:12 UTC

The first calibrated-model paired/shuffled visual curves are complete. At
8,192 decisions, the small interim retention probes score 65.47% / 65.38%
balanced for paired rewards, versus 49.59% / 48.64% for shuffled rewards
(training seeds 501/601). However, the stricter independent test after 128
neutral-gray warmup decisions gives only **58.49% / 59.56%**, versus 49.16%
original. Right-cue accuracy is 44.76% / 47.39%. BOTH fail the preselected
>=60%, each-cue >=55% gate. Completed independent shuffled controls score
48.81% / 49.83%; all four individual noise streams improve with paired
weights versus original in both training seeds, but this is still modest.
These are descriptive small-sample results, not a statistical claim.

Artifacts: `visual-learning-curve-20260918T140900Z-5b30a6` and
`...T140910Z-f1ebf1`; independent 2501..2504 reset tests
`visual-retention-probe-20260918T142820Z-179f1f` and `...T142831Z-7f96e5`.
The identical original control in the two reports is a reproducibility check,
not two independent controls. Preselected unchanged training continuations to
16,384/32,768 and final-weight independent seeds 2601..2604 are ongoing. No
assay weights enter Pokemon. Reversal must start both arms from the SAME
acquired state; protocol unit tests now verify that intervention and shuffling.
Shuffled artifacts: `visual-retention-probe-20260918T144721Z-359cfe` and
`...T144721Z-92caab`. The acquired-weight direct-current audit also finds the
correct cue contrast at BOTH steering cells in both seeds (~0.45..0.57 sample
standard deviations, versus ~0.001..0.015 original and ~0.05..0.07 shuffled).
It reuses original presynaptic samples and therefore excludes recurrent
changes: `retained-current-audit-20260918T150116Z-225a19`.

Protocol extension, explicitly exploratory despite the failed acquisition
gate: same-start reversal of the 8,192 acquired brains is now running at
512/2,048/8,192 additional decisions, with independent 2701..2704 retention
preselected. Artifacts `visual-learning-curve-20260918T150238Z-ab054c` (501)
and `...T150238Z-718867` (601). This does not retroactively pass acquisition.

The literal-application 6,000-decision pairs also completed:

| Seed | Frozen / learning tiles | Starter and Route 1 | Up, frozen / learning |
|---|---:|---|---:|
| 401 | 188 / 213 | Neither arm | 24.45% / 25.33% |
| 402 | 193 / 233 | Learning only; starter at 3,748 | 24.87% / 24.70% |

All four leave the house; none wins a battle in this interval. This does not
establish retained gameplay improvement. All four are now explicitly resumed
to 24,000 total decisions without a reset or changing reward/model parameters,
including the less-successful seed401. Artifacts:
`visual-model-gameplay-20260918T141410Z-cda52c` and `...T141421Z-31e87a`;
continuations `...T144408Z-70369b` and `...T144408Z-e4332d`.

The neutral-only stronger-release v2 sensitivity test failed: escape/B activity
stays 9.0..9.3 Hz across reset probes despite a uniform 1 Hz calibration target.
It is NOT used in gameplay (`intrinsic-probe-20260918T142524Z-2de05f`). The
calibrated-v1 supervised capacity diagnostic reaches 61.38% forward / 65.65%
reverse versus 50.01% original (`oracle-synaptic-capacity-20260918T141731Z-f42512`).
Those temporary oracle fits were discarded, never exported: capacity, not learning.

Current verification: 297 Python tests, 24 JS tests and lint pass
(`verify-ea76cc06ad2a4f27846085cc2d288de6`). The desktop model label identifies
frozen internal visual-rate calibration without mislabeling rates as spikes.
The launcher accepts opt-in `-Profile visual-rate-v1`; default is unchanged.
Browser-rendered verification remains unavailable; actual HTTP/stream checks
and prior exact neural/emulator resume tests pass as detailed below.

One additional GAME learning-dose candidate is registered, not promoted:
`visual-fast-learning-v2`, same brain/rewards but internal rate 0.2 instead of
0.02. The 8,192 assay had 1,488.92 total tanh reward versus 14.56 in the
6,000-decision seed402 game. This is not a gradient estimate, but motivates
testing learning rate without adding denser/shaped rewards. Both fresh 401/402
6,000-decision candidates were registered (negative results above). Frozen 64-decision prefixes reproduce
the original controls exactly, including raw input hashes; those existing
controls are reused, not counted as new trials.

The visual Euler update now fuses CUDA dispatch while explicitly preserving
each float32 rounding operation. All neural arrays, actions and spike counts
match over four 64-decision learning branches. Shared-GPU timing benefit is
only ~2..7%, not a large speedup. Artifacts `visual-update-benchmark-20260918T145259Z-73f582`
and `...T145409Z-e7fa9e`. Real-game exact resume/frozen/no-reward checks pass
(`internal-smoke-20260918T145416Z-f513e0`), as do live pixel/brain/button/speed
checks (`internal-learn-20260918T145811Z-42f1e9`). The full suite at 15:01 passes
299 Python / 24 JS tests and lint (`verify-6da5246c70af41f4bdffa89e39b10917`).
Subsequent protocol guards have separate passing unit checks. Browser retry
at 15:03 remains unavailable; no rendered-layout verification is claimed.

## Implementation evidence archived at 14:17 UTC

The user authorized offline calibration of INTERNAL visual neurons on generic
stimuli/published responses, frozen before Pokemon learning. The fly still
chooses every button through the same motor cells. No external visual policy,
virtual photoreceptor, game-aware feature or route controller is introduced.

### Visual calibration: implementation and early evidence, not a solved task

The pinned reference graph maps exactly to 65,799 existing cells and 1,967,771
existing edges. Its count scaling agrees with the original graph; 24,365
reference edge signs differ, so Pokefly keeps ORIGINAL signs. Audit:
`visual-reference-audit-20260918T134644Z-5ec14f`. Data/license provenance and
full limitations are in `THIRD_PARTY_NOTICES.md` and `MODEL_VARIANTS.md`.

The isolated visual prototype responds to moving gratings, but T4/T5 group
direction selectivity is only 0.025..0.242 and differs between eyes. About
3.1..4.1% of cells reach the chosen ceiling; this is not validated physiology.
Artifact `calibrated-visual-probe-20260918T135707Z-d52749`. The unfitted flyvis
parameter transfer is weaker (0..0.023), with zero-strength reference edges
retained at original magnitude rather than deleted:
`calibrated-visual-probe-20260918T140047Z-01e298`. Neither is an exact reproduction
of the upstream graph/photoreceptor convention. Whole-fly integration adds
one missing original L4, retains all cross-boundary connections, and freezes
the visual model before gameplay. New neutral spiking calibration is stable
across three reset seeds (mapped motor means 0.65..1.32 Hz):
`intrinsic-probe-20260918T140237Z-11ac48`.

In the full fly, an OFFLINE held-out diagnostic now distinguishes opposite
motion at descending cells and motor inputs on all eight constant samples and
all tested switching windows; the old motor-input measurement was 50%. These
are small correlated diagnostic samples, not a significance test or deployed
classifier. Actual untrained button choices remain weak: 49.4% balanced for
motion and 43.7% for static cues. Artifacts:
`visual-latency-probe-20260918T140540Z-de7dbb` (motion),
`visual-latency-probe-20260918T140551Z-5ff069` (static).
Static steering-current contrast/std is still only 0.0009 / 0.015; distributed
input information is not automatically routed into motor choices. Audits:
`motor-signal-audit-20260918T141620Z-869982` and `...T141621Z-7bfcc7`.

Perturbation learning now accepts actual continuous presynaptic release for
the new model; previously it would silently omit graded visual activity.
Prior-history timing is maintained, with no fictitious spikes or labels.
Both registered paired/shuffled visual curves and two matched actual-game
learning/frozen comparisons are ongoing. There is no retained visual-learning
or improved-progression claim yet.

### Exact-resume defect found and repaired

The first real-ROM visual-model smoke failed at decision 51. Neural-only
snapshots were exact, but the first post-load sampled image differed in 3,470
pixels. PyBoy 2.7 does not serialize its renderer's window-line counter.
A temporary render followed by reloading the exact saved bytes restores it;
serialized game state stays identical, and the temporary image/time/rewards
never reach the fly. Diagnostic:
`emulator-resume-probe-20260918T141055Z-39e8ce`. The failed runs are retained.

After repair, real-ROM learning/resume/frozen/no-reward tests pass exactly:
`internal-smoke-20260918T141157Z-f22c68`. The older stopped-run branch test also
passes with unchanged sources: `checkpoint-window-verification-20260918T141218Z-f57c61`.
Live HTTP pixel/graded-activity/button/speed tests pass on
`internal-learn-20260918T141211Z-c9b96e`. Browser connection retried twice;
rendered visual QA is still unavailable. Final integration checks: 294 Python
tests, 23 JavaScript tests and lint pass (`verify-47f32e82a6564df69623a42009b77177`).
All ten protected original files still match at 14:18 UTC, with unchanged
retina hash (`visual-integration-protection-20260918T141840Z-6e415c`).

### Completed negative/mixed learning controls

Projected-score v5 final 8,192-decision balanced cue accuracy is 52.4% / 56.8%
for training seeds 501/601; matched shuffled controls 47.7% / 49.4%. Neither
passes the preselected >=60% screen. Not promoted. Artifacts:
`visual-learning-curve-20260918T132756Z-69204e` and `...T132756Z-39c426`.

The old-circuit supervised capacity control with wider 0.1..10 synaptic bounds
(same +/-25% budgets and preserved predicted mean input) obtains 64.1% forward
and 82.4% reversed balanced cue accuracy on independent noise streams. This
is a temporary OFFLINE oracle, not reward learning; no fit was exported or used
in the game. Original control reproduces the prior report exactly. Artifact:
`oracle-synaptic-capacity-20260918T133928Z-9a8dfe`.

## Evidence archived at 13:27 UTC

Public source repository: https://github.com/Reldnahc/pokefly . ROM, model data,
saves, checkpoints and raw runs remain local and Git-ignored.

### Completed before the interrupted turn was resumed

The three outstanding processes exited normally, with their complete reports
saved. Broader anatomical plasticity is NOT promoted. At 8,192 visual-training
decisions, its paired balanced cue accuracy is 51.6%, shuffled 46.0%. Pooled
accuracy (54.7% versus 45.2%) overstates the cue distinction: paired Left/Right
counts are 54/10 on the left image and 43/10 on the right image. The result is
mostly a generic Left preference, not reliable context learning. Artifact:
`visual-learning-curve-20260918T105829Z-52ad54`.

Actual Route-1 runs, 6,000 decisions each, original weights at the start:

| Seed | Reused frozen positions / wins | Motor-only learning | One-hop premotor learning |
| --- | --- | --- | --- |
| 1201 | 337 / 0 | 208 / 3 | 249 / 1 |
| 1202 | 414 / 1 | 327 / 1 | 430 / 1 |

Motor-only uses the same perturb-v3 rule as the one-hop candidate, not the
older bounded covariance rule. Both validate exact original-weight frozen
64-decision prefixes against their explicitly reused controls. Northernmost
Route-1 y is 24 in every arm except broader seed 1202 (25). No new town,
capture or badge. More changed connections is not evidence of better learning.
Artifacts: `anatomical-scope-gameplay-20260918T113017Z-f591b6` and
`anatomical-scope-gameplay-20260918T113028Z-d26821`.

Frozen transfer of the FINAL brains from the two 24,000-decision whole-game
training runs also completed. Same Route-1 reset and two 6,000-decision seeds:

| Seed | Original positions / wins | Legacy-outcome trained | Earlier-outcome trained |
| --- | --- | --- | --- |
| 1201 | 337 / 0 | 338 / 5 | 387 / 0 |
| 1202 | 414 / 1 | 312 / 0 | 442 / 3 |

The earlier-outcome brain increases coverage on this small panel, but wins are
mixed and there is still no new town/capture/badge. Northernmost y remains 24
except legacy seed 1202 (23). Learning is off throughout evaluation; original
controls are reused, not additional trials. Final training checkpoints were
preselected, not chosen from peaks. Artifact:
`final-game-weight-transfer-20260918T110148Z-e7bea9`. An earlier driver failed
before gameplay because it compared pre-load and post-prime emulator hashes;
that failed report is preserved and the successful protocol checks both states
at their appropriate boundaries.

### Local credit and signal-to-noise diagnostics

Two frozen score-v2 credit collections (4,096 decisions each) have inter-seed
update cosine 0.027. A bounded temporary step along the averaged direction
raises target frequency 18.5% -> 20.8%, but conditional cue accuracy only
44.9% -> 45.5%; the negative direction gives 16.5% / 44.7%. These are diagnostic
temporary synapses, not online learning or deployed game weights. All originals
were restored. Artifact: `local-credit-direction-audit-20260918T111051Z-6af0e0`.

Projection of already recorded stochastic presynaptic counts through original
signed weights estimates the steering left-minus-right mean-current contrast
at 0.10 / 0.16 of within-cue standard deviation. There are no missing graded
inputs to these motor cells. This is a decision-averaged current measurement,
not the full membrane dynamics or proof of a causal bottleneck. Artifact:
`motor-signal-audit-20260918T111942Z-4607d0`.

The broader learning kernel now has optional bit-exact CUDA acceleration for
at least one million eligible edges (the default smaller scope is unchanged).
At 4.3 million edges, including host/device transfers, single-trace updates
take about 3.2 ms versus 13 ms on CPU; dual traces about 5.6 versus 19 ms.
This is a kernel comparison under concurrent load, NOT a fourfold whole-app
speedup. CPU-threaded variants were not faster and are not activated.
64 complete learning windows match CPU exactly in counts, actions, weights
and all final neural state. Actual-ROM learning/resume/frozen/no-reward smoke
also passes: `internal-smoke-20260918T112726Z-a7a496`.

Two new opt-in hypotheses were preselected at 13:23 UTC. Projected-v5 preserves
the score-v2 forward circuit and constrains the local update against estimated
mean input. Exact frozen forward and real-ROM checkpoint smoke pass; the two
paired/shuffled learning curves are running, with no success claim yet.
Whole-circuit gain 12 with the existing bounded neutral calibration fails its
stability check: reset A rates 5.5..9.3 Hz, B 15.2..15.8 Hz, and MBON mean
21.5..21.8 Hz, versus target 1 Hz. Frozen held-out cue measurement falls to
50% at motor inputs / 37.5% descending, versus 100% in the original measured
model. Steering current contrasts remain only 0.14 / 0.13 of within-cue
variation. This candidate is NOT promoted and will not receive gameplay
training. Artifacts: `intrinsic-probe-20260918T132602Z-f862ec`,
`visual-latency-probe-20260918T132756Z-42f162`,
`motor-signal-audit-20260918T133209Z-661196`.

Current checks: 286 Python tests / 23 JS tests / lint pass. The new mean-input
rule has exact 64-window original forward equivalence and actual-ROM
learn/resume/frozen/no-reward smoke. All 20 original recorded samples
68,501..68,520 still reproduce exactly after the inactive gain-field addition;
all ten protected ROM/model/stopped-run hashes are unchanged (13:30 UTC).
This is software verification, not evidence of learned game progress.

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

### Earlier delivery of unchanged outcome rewards: mixed results

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

The literal earlier-outcome serial run finished 24,000 decisions with 497
sampled positions and five wins (rival plus four wild), final level 7, versus
478 positions/two wins/level 6 in the same-seed legacy-timing control. Both
trajectories match in actions, telemetry, rewards and learning through decision
5,049, immediately before the first changed reward. Neither reached another
town, captured a Pokemon or earned a badge. Artifact:
`internal-learn-20260918T095634Z-3c6661`. This one matched seed is an exploratory
positive gameplay observation, not retained-learning proof. Frozen transfer
of BOTH final 24,000-decision brains is now being tested, not selected peaks.

The separate repeated-rival panel did NOT improve: original 4/8 wins, retained
4/8 after eight practice episodes (training itself won 5/8). Individual seeds
changed in both directions. Artifact:
`controlled-battle-learning-20260918T095632Z-8e9695`. Earlier reward delivery is
not automatically a stronger learning rule; the old delayed-outcome dual model
had 6/8 on that same held-out panel. No independent confirmation was launched
for the failed early-outcome result.

Preselected before that panel finished, its final battle-practice weights also
transferred to two fresh 12,000-decision games. Positions original/retained:
444/348 and 353/331; wins 2/1 and 1/1. Neither retained run reached Route 1.
The second retained win was confirmed at the last faint, but its encounter was
still unfinished at the evaluation boundary; the original controls pay later.
Do not hide this boundary difference or interpret it as a completed additional
battle. Artifact `battle-to-game-transfer-20260918T103109Z-dd59c2`. Original
controls are validated reused trials, not new evidence. Practice resets remain
explicit interventions. Overall this is negative transfer evidence, not a
general-player improvement. Old profiles/checkpoints keep `encounter-end-v1`.

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

Extending score-v2 gives paired accuracy 52.0% at 8,192 then 46.2% at 16,384;
shuffled controls are 46.5% and 46.4%. Artifact
`visual-learning-curve-20260918T093705Z-f817f9`. The separately versioned
presynaptically centered spike-innovation rule also fails: 48.4% paired versus
48.7% shuffled at 8,192 (`...T100345Z-350e0c`). Its paired left-cue choices are
31 Left/33 Right; right-cue choices 33 Left/31 Right. Higher target-action rate
is not cue discrimination. This rule is an approximation, not the exact score
gradient of the physical neuron. Frozen activity matches score-v2 over 64
paired windows. Neither candidate nor its synthetic weights is promoted.

### Further motor and sensory-capacity diagnostics

From the same Route-1 state, original frozen weights, 6,000 decisions per seed:

| Seed | Original / 3-second direction trace / motor adaptation positions | Northernmost Route-1 y | Wild wins |
| --- | --- | --- | --- |
| 1201 | 337 / 308 / 353 | 24 / 20 / 18 | 0 / 6 / 3 |
| 1202 | 414 / 168 / 252 | 24 / 24 / 28 | 1 / 0 / 3 |

Neither candidate reaches a new town. Original controls are reused explicitly.
Longer movement bouts are mixed, not a solution. The adaptation candidate also
has its own neutral calibration, so this tests those two coupled settings,
not adaptation alone. Artifacts `direction-duration-comparison-20260918T095952Z-d4bc0c`
and `motor-adaptation-comparison-20260918T102359Z-61e321`. Exact replay of the
3-second arm at decisions 3,406..3,413 shows repeated Up into a north-facing
ledge (`recorded-replay-proof-20260918T101515Z-a1360c`): this later failure is
not missing Up, and no wall detector is fed back to the controller.

A separate ROM-free capacity control temporarily fits existing DNa02 synapses
with supervised cue labels, then evaluates the ACTUAL recurrent fly and fixed
decoder on four fresh noise seeds. No fitted weights are exported or used in
Pokemon. This is deliberately NOT learning proof. With positive-edge factors
0.25..4, +/-25% local input budgets and fixed predicted mean cue input, accuracy
is 55.1% forward and 79.8% reversed (original forward 40.7%). Allowing both
original signs, with separate E/I budgets, gives 56.9% / 76.3%. Fixed inhibition
alone is not established as the bottleneck. These first-order fits do not
prove a general capacity limit. Artifacts `oracle-synaptic-capacity-20260918T102612Z-f7f970`
and `...T103600Z-6d3690`; source hashes unchanged and original weights restored.
The repeated original arm matches exactly and is validation, not replication.

The earlier successful sensory audit used deterministic thresholds. A matched
audit of the stochastic score-v2 model also distinguishes cues at visual
projection and motor-input populations in all tested windows, using a
measurement-only held-out-noise classifier. No classifier is deployed. Artifact
`visual-latency-probe-20260918T104309Z-8665db`; correlated small samples, not a
significance test. Cue information exists, but effective learned control remains
unestablished. An optional one-hop premotor learning scope now tests the prior
restriction to motor inputs; frozen activity is unchanged and a paired/shuffled
visual assay is running. This is another hypothesis, not a successful solution.

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

- Latest full suite: 269 Python tests, 23 JavaScript tests and lint pass
  (`verify-179558ee31f54b3a8c26a815061c088f`). All ten protected source/ROM/
  model/user-run hashes still match at 11:03 UTC.
- Early-outcome checkpoint at decision 5,050 contains a paid but unfinished
  rival encounter. Resuming it reproduces all next 50 samples, final neural
  arrays and rewards exactly, without duplicate payment
  (`checkpoint-window-verification-20260918T101037Z-65eccc`). This is a
  verification branch, not another autonomous win.
- The original user checkpoint reproduces recorded samples 68,501..68,520,
  and its split continuation reproduces all final arrays/rewards
  (`checkpoint-window-verification-20260918T101118Z-9e8f3d`). Source unchanged.
- After adding inactive motor adaptation, the optional premotor scope and
  faster sparse-index lookup, the user's next 20 original recorded samples
  still match all non-walltime fields exactly. Split continuation reproduces
  final arrays/rewards (`...T105752Z-68ca30`). The broader-scope real-ROM smoke
  test passes exact resume, frozen and zero-reward checks
  (`internal-smoke-20260918T105620Z-1d9798`); these are not gameplay wins.
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
