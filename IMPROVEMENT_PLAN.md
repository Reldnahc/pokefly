# Pokefly: neural-control and learning improvement plan

2026-09-19 07:31 UTC: v11 recall COMPLETE and raw-audited in
visual-retention-gate-20260919T073022Z-acc0fd.501 PASSES: paired64.2579%,
individual62.9268%/65.5889%, shuffle44.6062%.601 FAILS each-cue threshold:
paired61.4037%, individual69.1038%/53.7037%, shuffle46.0069%. Shared original
44.9132%, all original raw actions identical. OVERALL required two-seed gate
FAILS; not zero learning, but no v11 full-game108k study or promotion.31785/
45632 exited/drained. All sources/final8192 checkpoints retained unselected.

CURRENT ONLY TWO GPU JOBS: registered v10 longer WHOLE-game practice,4 extra
fresh18k attempts each, final96k total/SIX games per brain. No model/reward/
decoder change.4501 family session90734, series
retained-game-series-20260919T073051Z-db71a2, seeds61001..61004, dashboard59320;
current61001 actual internal-learn-20260919T073054Z-9944ac.
4502 family session44677, series retained-game-series-20260919T073051Z-f7b4fc,
seeds62001..62004, dashboard59322;current62001 actual
internal-learn-20260919T073054Z-896c7b. BOTH use --defer-evaluation and their
own preserved24k parent checkpoints, no synthetic weights or selected stages.
Allow ALL FOUR attempts to finish in each existing process; no extra GPU job.

After BOTH series complete, read each final training row's actual run path.
Evaluate ALL FOUR brains (original24k parents4501/4502 and final96k61004/62004)
with one shared original, whole-fresh/frozen18k panels63001/63002. Then run
screen_training_extension.py --reports <both> --families 4501:61004 4502:62004.
Use the07:09 registered parent-relative and original-relative gate below;
reserve63003/63004 SAME-child confirmation if triggered. The normal default
and user history stay unchanged. Current full verification575Python/26JS/lint.

2026-09-19 07:24 UTC: BOTH v11 paired/shuffled curves COMPLETE8192;65056/24345
exited/drained. Short final probes501 paired54.83%,shuffle42.66%;601 paired57.54%,
shuffle46.26%. These do NOT replace the registered independent recall gate.
CURRENT ONLY TWO GPU JOBS are final frozen recall, ALL arms9201..9204:
501 session31785, visual-retention-probe-20260919T072111Z-d645fa;
601 session45632, visual-retention-probe-20260919T072210Z-0b161a.
Both use128 neutral+128 decisions/cue; original first arm for501 is44.9132%
balanced. No qualification conclusion until BOTH complete and raw-audited with
summarize_visual_retention.py --reports <both paneldirs>. No original brain,
weights, rewards, profile or gate modified during these experiments.

Full CUDA-inclusive regression NOW575Python/26JS/lint PASS,
verify-49fcff341c944c128408dbb22e713cc7;39454 exited/drained. Latest source-only
commit7e407f1 pushed, includes extension observer and its21 tests. If BOTH v11
lineages pass, use the06:59 amended SIX-game/108k whole-game protocol, not the
old36k plan. If v11 fails, use the07:09 unchanged-v10 FOUR-additional-game/96k
parent-versus-child study, after all current recall arms finish. BOTH training
families retained in either branch. Never promote on a small positive probe.

2026-09-19 07:17 UTC: conditional extension observer IMPLEMENTED/CPU-verified,
not run as a neural study. scripts/screen_training_extension.py re-audits ALL
five-arm panels, scores only the assigned children for extension success,
requires original AND own-parent comparisons, and verifies exact ancestry to
the pinned parent brain. Use --reports <both completed paneldirs> --families
4501:61004 4502:62004. These are2 related families, not4 independent replicas.
21 new fixture cases pass; real existing401->4501 and402->4502 checksum links
pass, and the deliberately wrong cross-family relation is rejected (read-only,
not a new training result).565 CPU Python/26JS/lint passes,
verify-6127db69ebcb48dfa01d467f5c21ac7c;95550 exited. Prior60003 exited after
565Python passed but lint found2 formatting issues, which are now corrected.
Latest FULL CUDA suite remains554Python before these21 observer-only tests;
rerun full suite when a GPU slot is available. No gameplay/controller changes.

CURRENT ONLY TWO GPU JOBS remain v11 curves65056(seed501)/24345(seed601),
both now SHUFFLED arms; paired8192 short balanced probes54.83%/57.54%.
Do NOT use those small probes to replace the registered final9201..9204 recall.
Paths are the06:50 entry below. No extended v10 game or v11 game study started.

2026-09-19 07:09 UTC: prospective FALLBACK after a v11 gate failure, NOT RUN.
Keep v11's existing priority and complete all its registered arms first. If it
fails, test MORE whole-game experience in the unchanged, memory-capable v10
model instead of discarding its two saved gameplay histories. Carry BOTH final
24k sources4501/4502 through FOUR additional whole fresh18k games each:
4501 (internal-learn-20260919T042416Z-ad59e3) ->61001,61002,61003,61004;
4502 (internal-learn-20260919T042415Z-c28307) ->62001,62002,62003,62004.
Final total96k training per brain (6k+18k+4*18k), SIX whole-game attempts.
Use train_game_series.py --initial-game-run <parent> --training-seeds <four>
--steps 18000 --evaluation-steps 18000 --eval-seeds 63001 63002
--defer-evaluation, two separate ports, maximum TWO GPU jobs. No model, reward,
mapping, stage reset or original-source overwrite. No intermediate selection.

Dose test: ALL FOUR source brains (24k parents4501/4502 and FINAL96k children
61004/62004) plus ONE shared original per seed, whole fresh/frozen18k63001/63002.
These are TWO related training families, NOT four independent lineages. Each
child must pass the existing original-comparison whole-game-progress-v1 gate;
also preserve every matched parent's starter/Route1/completed battle/rival/
capture/gym milestone on both seeds, and EITHER add parent-relative progression
OR reach the SAME starter/battle endpoint >=10% earlier on BOTH seeds relative
to that parent. Include every parent and child, never let a parent-only pass
qualify the extension. Verify the actual checkpoint ancestry from child to
assigned parent. The observer-only extension screen will be unit-tested before
these games launch. If triggered, repeat ALL five arms on63003/63004; the SAME
child family must pass confirmation. This is a conservative development gate,
not significance or an endgame guarantee. Current v10 failed result stands.

Motivation is the user's whole-game learning requirement and v10's mixed
two-attempt result, including faster starters for4501. More practice is not
assumed to fix the missing long-horizon predictor. No extension result exists;
the currently running v11 cue curves65056/24345 are UNCHANGED.

2026-09-19 06:59 UTC: EXPLICIT PROSPECTIVE v11 practice-budget amendment.
Its70-decision software smoke has completed, but NO full-game training study
has begun. The TWO current ROM-free curves65056/24345 and ALL their registered
memory/holdout criteria are UNCHANGED. V11 still needs BOTH final cue-memory
panels to pass before the game study. Smoke/assay weights never seed that study.

If qualified, keep the initial18k learn/frozen pairs401 learn-first and402
frozen-first as registered. Then give EACH final learner FIVE whole fresh18k
practice games, not one:401 lineage uses4701,4703,4705,4707,4709;
402 lineage uses4702,4704,4706,4708,4710. Every attempt keeps its own previous
FINAL synapses, resets the whole game/novelty/transient state, and chooses every
post-intro button itself. This is SIX actual whole-game attempts and108,000
training decisions per lineage (18k+5*18k), replacing the03:24 total36k plan.
No early-game gate skips a lineage and no intermediate successful brain is
selected. Use train_game_series.py --initial-game-run <initial learner>
--training-seeds <its five seeds> --steps 18000 --evaluation-steps 18000
--eval-seeds 4801 4802 --defer-evaluation; separate ports for the two workers.

Then ALL final4709/4710 brains plus shared originals get unchanged whole-fresh
frozen18k tests4801/4802, whole-game-progress-v1, and4803/4804 confirmation if
triggered for the SAME lineage. Model, reward values/timing, decoder and all
success criteria remain fixed. Normal-user defaults/history are unchanged.
Motivation: v10's two-attempt result shows faster starters but mixed later
progress, and game reward exposure is much sparser than the cue assay. This is
a larger-training-budget hypothesis, not evidence that time fixes delayed
credit and not an amendment to v10's FAILED completed screen. It directly tests
sustained learning across entire games. User informed before this amendment.
Maximum TWO GPU jobs still applies. No battle/stage starts or new source model.

2026-09-19 06:50 UTC: v11 neutral calibration COMPLETE and sanity PASS:
reset301/302/303 motor rates0.8626..1.2044Hz, finite/KC/MBON limits pass.
AssetSHA6e899a4add8da3781a232eb100e847a508bd169d5965f8637da14c82067b9e43.
Calibration99164 exited. Actual-ROM smoke internal-smoke-20260919T064635Z-d52285
passes exact20-decision resume/all arrays and rewards, frozen/no-reward weight
invariance, retained-weight current effect, sparse-reference propagation and
unchanged ROM;63845 exited. This is software stability, not gameplay learning.

CURRENT ONLY TWO GPU JOBS: v11 paired/shuffled visual curves, sessions65056
and24345. Seed501=visual-learning-curve-20260919T064743Z-94f579;
seed601=visual-learning-curve-20260919T064743Z-c975ab. Registered512/2048/8192,
unit correct/zero incorrect rewards, default left/right cues/buttons unchanged.
Finish BOTH ALL arms, then final8192 recall panels on9201..9204,128neutral+
128/cue; summarize_visual_retention.py applies the same registered two-seed
gate. Do not select an intermediate peak or import any assay weight into a
game. If qualified follow the03:24 v11 whole-game36k/4801/4802 protocol below.
Process audit06:49 confirms only these2 workers plus venv launchers, no agents.
Latest full suite554Python/26JS/lint. Normal default and user history unchanged.

2026-09-19 06:44 UTC: BOTH v10 frozen18k panels COMPLETE;53942/88470
exited/drained. Raw checkpoint/trajectory/ancestry audit and registered screen:
game-retention-progress-screen-20260919T064322Z-3b8eec, BOTH lineages FAIL.
4501 obtains starters >=10% sooner on BOTH seeds, and adds Route1/wild victory
on4602, but loses4601 rival victory versus original.4502 misses4601 starter/win.
No4603/4604 confirmation or default promotion is triggered. This rejects the
candidate at24k training, not all longer training; keep every source/result.
Final third arms:4601 retained4502=364 positions, no starter/win/Route1,
step-00018000-affbbaf1;4602 original=360 positions, no starter/win/Route1,
step-00018000-e21e1692. See completed six-arm table in FOLLOWTHROUGH_RESULTS.

CURRENT ONLY GPU JOB: registered v11 neutral calibration99164,
intrinsic-probe-20260919T064348Z-ed09ca, immutable export
fly-data/intrinsic-neutral-visual-wide-score-v11.npz. No game/reward/button
labels; original synapses. Apply the03:24 sanity thresholds BEFORE any behavior
testing. Then actual-ROM smoke, paired/shuffled501/601 curves512/2048/8192,
ALL final arms9201..9204 recall and same two-seed gate. Full CUDA regression
already passes554Python/26JS/lint, verify-03cfe167d3be47e5b30cb66b7bd7afed;
84650 exited/drained. Process audit06:44 shows only calibration worker+launcher.
No prototype synapses enter the normal user's brain/history. Default wide-v3.

2026-09-19 06:39 UTC: ONLY GPU panels53942/88470 continue, third/final arms
around16k/18k. CPU-only ancestry guard now excludes test seeds used by ANY
ancestral whole-game launch, not just the final practice. Exact resumes retain
their parent's RNG and do not count an unused launch-seed option. Panel launch,
whole-game practice scheduling, panel audit and neutral motor audit use the
guard. Source-provenance/report formats and actual controller are unchanged.
Both CURRENT source lineages pass4601..4604 exclusion, with unchanged brain
hashes.544 CPU Python/26JS/lint passes, verify-fb9e784f57e045d49c51cf748fce21d5;
11951 and prior34051 exited/drained. Full CUDA regression awaits a free slot.
At06:29 all10 protected ROM/model/old-user-run hashes matched. At06:31 browser
skill retry still unavailable; no rendered UI validation or replacement browser.
No v11 run yet. Finish/audit BOTH panels before the conditional next candidate.

2026-09-19 06:20 UTC: second arms COMPLETE18k; ONLY existing GPU panels
53942/88470 now run their THIRD/final arms.4601 retained4502 actual
internal-frozen-20260919T061144Z-d41ea6;4602 original actual
internal-frozen-20260919T061224Z-ebd29f. Both whole fresh openings, no updates.
Completed4601 retained4501:331 positions, starter3462, battle4252, zero wins,
no Route1/capture/badge; final step-00018000-2255acc4. It regresses against
original4601's rival win, so cannot satisfy the registered preservation gate.
Completed4602 retained4501:410 positions, starter3559, battle3716, Route1=6667,
45 Route1 positions/northernmost y24, rival win1 and wild win1, no capture/badge;
final step-00018000-ea49f120. Its original comparison is still pending.
CPU replay audit recorded-outcome-latency-20260919T061840Z-a161f8 reproduces
ALL18k sampled states/rewards: rival win4111, wild win16645, actual wild loss
17171 (playerHP0/no living party). Return home is NOT evidence of planned
healing. Audit session12632 exited/drained; no new autonomous trial/training.
Finish BOTH panels and screen ALL sources/seeds as registered below; no default
promotion, no battle-start practice, no intermediate-checkpoint selection.
V11 remains conditional/unrun. Only documentation and observer analysis since
last code verification. Latest current third arms have no starter at~4.5k.

2026-09-19 05:57 UTC: same TWO GPU panels53942/88470, still second arms
(retained4501 on both seeds). Provisional4601: starter3462 vs original3918
(11.64% sooner), firstbattle4252 vs4001 (later); no victory so far after returning
to overworld.4602: starter3559, firstbattle3716, rival win1 and firstRoute1=6667,
then returned to Pallet Town. Its matching original is the pending third arm.
Do not call this retained progress proof or compare across mismatched seeds.
All arms must still finish. Source/reference/defaults unchanged. Read-only
review confirms older delayed-credit failures used different weak-vision/tight-
bound models, not v10 anchored strong vision. No extra candidate or GPU run.

2026-09-19 05:41 UTC: first arms in BOTH frozen panels COMPLETE18k, panels
still running ONLY53942/88470.4601 original:379 positions, starter3918,
battle4001, completed rival win1, no Route1/capture/badge; final
internal-frozen-20260919T050809Z-e685e2/step-00018000-26e50153.
4602 retained4502:351 positions, starter9115, battle9756, rival win1,
no Route1/capture/badge; final
internal-frozen-20260919T050832Z-80a208/step-00018000-3455b0a7.
Do NOT compare these different seeds as a matched pair or claim a panel pass.
Auto checks verify zero updates, exact assigned weights and identical starts.
CURRENT SECOND ARMS: retained4501 on BOTH held-out seeds:
4601 actual internal-frozen-20260919T053942Z-b279af (53942);
4602 actual internal-frozen-20260919T054013Z-14b076 (88470).
Then automatically retained4502 on4601 and original on4602. Complete ALL before
screen_game_retention.py on the two registered paneldirs below. No new GPU job,
no default promotion, no protocol change; v11 still conditional/unrun.
Process audit05:36: ONLY these2 project Python workers plus their venv launchers;
no CPU audit/test leftovers, no agents. Latest implementation commit a86bf4c pushed.

2026-09-19 05:20 UTC: ONLY GPU panels53942/88470 continue from05:09 below.
Observer-only end-state audit added;534 CPU tests/26JS/lint passes,51674 exited.
CPU whole-opening recorded-button playbacks both verified18k exact states and
rewards;66759/55797 exited.4502 actual loss: playerHP0/enemy5/no living party,
end11209, no victory evidence.4501 completed win4387 confirmed. Winning last
accepted move4337->reward4387:50 decisions/12 neural seconds, isolated.6s
trace passive fraction2.061e-9 (NOT net trace/credit). Audit artifacts
recorded-outcome-latency-20260919T051643Z-aaf9e2 and ...T051803Z-9142dd.
No new neural trial, policy input, reward shaping, battle-stage start or model
change. Current/fallback registered protocols remain unchanged. Useful
retained whole-game learning is still pending4601/4602, not established.

2026-09-19 05:09 UTC: BOTH v10 WHOLE fresh-game practices COMPLETE18k;
51191/41934 exited/drained. Both final lineages now24k total training (6k+18k).
4501:340 positions, starter3020, battle3481, completed rival win/reward4387,
final Pallet Town level6.4502:353 positions, starter9040, battle10315, no win,
final Pallet Town level5. BOTH no Route1/capture/badge. Do not call different
practice seeds a controlled before/after improvement. Raw samples1..18000,
all-fly action sources, whole-game ancestry and completed encounters verified.
4501 final step-00018000-cb010e60, brainSHA
5c74c71c73e449ea35007f231308f4ae45fc362761f4b2a134712bf32ce5c443;
trajectorySHA10f44a520343c4f7a3f81f9f97fba909ffe866009e6cad5e262658e7d37be015.
4502 final step-00018000-8eddedc7, brainSHA
c0f8acac73c777357b89f7117e798f0cdfcf874efcf8da4fb5faf9a45d0e9228;
trajectorySHAe46ba2887d4750c3e8f62230d48a4df72e58a4291b648442777020d4da9b30fe.
Both startsSHAffba33e31392541f192b608b461461e3821d54e3ebad480bd11b7ab81c3118e4.
Full CUDA-inclusive verification540Python/26JS/lint passes,
verify-44596fe3472241c5af558a01a67d38fd;37638 exited/drained.

CURRENT ONLY TWO GPU JOBS: frozen18k WHOLE-new-game retained panels.
4601 session53942, game-retention-panel-20260919T050807Z-df5abf;
order original -> retained4501 -> retained4502, first actual original
internal-frozen-20260919T050809Z-e685e2.
4602 session88470, game-retention-panel-20260919T050829Z-32a439;
order retained4502 -> retained4501 -> original, first actual retained4502
internal-frozen-20260919T050832Z-80a208.
Both source-runs are internal-learn-20260919T042416Z-ad59e3 and
internal-learn-20260919T042415Z-c28307, ALL final practice brains, no selection.
After BOTH complete run screen_game_retention.py --reports on BOTH paneldirs;
apply the04:11 whole-game-progress-v1 gate. Reserve4603/4604 confirmation
only if triggered, for the SAME lineage, with all arms still included.
V11 remains unrun; normal default remains wide-v3. No small-reward assay run.
At04:59 all10 protected hashes match; no protected assets tracked by Git.
Browser retries04:46 still report unavailable; no rendered UI verification.

2026-09-19 04:39 UTC: WHOLE-game practice continues ONLY51191/41934;
~6,100/18,000 each, no resets.4501 has starter3020, battle3481 and completed
rival win, back outside level6.4502 still no starter. No Route1 yet. These
different-seed practice observations do NOT establish improvement against
their initial attempts. Finish both, then unchanged4601/4602 held-out panels.
Added an assay-only --correct-reward option and reward-dose provenance checks;
default1.0 and all registered current/fallback tests UNCHANGED. No small-dose
neural assay run. CPU verification530Python/26JS/lint passes;36437 exited.
Re-audit of existing v10 cue data still passes, ...T043929Z-b1c63b; not new data.
At04:44 corrected the provisional starter note3012->3020:3012 was the party
counter before completed initialization. The registered measure() already uses
party AND nonzero levels; no raw data or evaluation measurement code changed.

2026-09-19 04:24 UTC: v10 initial6k matched pairs COMPLETE and audited,
full-game-pair-audit-20260919T042349Z-a5210f. Learners BOTH starters and
completed rival wins; originals BOTH no starter/win. No Route1/capture/badge.
401:297 vs268 positions; starter3588,battle4246,victory4572.
402:191 vs188 positions; starter2363,battle3072,victory3793.
Both new original controls reproduce ALL6000 physical/action/reward/input
records and18 common final arrays exactly against historical originals. The
extra fixed_release_reference is learning-only. Count SHARED controls once.
Original pair sessions31236/34878 exited/drained. Gate PASSED, not retention
proof. Source401 final step-00006000-21ab3bc4 brainSHA
e6bb6fad5e25f7a698627ebfc3c8635e5ae5b979b28213afb0d7161137650669;
source402 step-00006000-4899389f brainSHA
b411f6f4167dc48eb340bd6bff51acec94b4ddcaffa4edc69c0f7c7a2fc003a9.

CURRENT ONLY TWO GPU JOBS: longer WHOLE fresh-game practice, each18k,
using the explicit04:17 amendment below. Final brains will have24k total
training (initial6k+practice18k), not12k or36k.
401->4501 session51191, retained-game-series-20260919T042413Z-96b924,
actual internal-learn-20260919T042416Z-ad59e3, dashboard59220.
402->4502 session41934, retained-game-series-20260919T042412Z-111c02,
actual internal-learn-20260919T042415Z-c28307, dashboard59222.
Both use --defer-evaluation and preserve original sources/checksums; scripts
validated whole-game ancestry before starting. After BOTH complete, run two
shared-control panels with ALL final practice brains, one each4601/4602 at18k.
Then screen_game_retention.py re-audits/computes the predeclared progress gate.
Reserve4603/4604 confirmation for the SAME winning lineage, keeping all arms.
Do not promote merely because the initial games succeeded. V11 remains unrun.

2026-09-19 04:17 UTC: EXPLICIT prospective v10 practice-budget amendment,
BEFORE either4501/4502 practice or4601/4602 retention has started. After the
registered initial6k pair screen/audit, BOTH saved learning brains will get
18,000 decisions in one WHOLE fresh game4501/4502, not the earlier6,000 plan.
Thus each retained lineage has24,000 total training decisions (6k+18k).
All initial6k comparisons remain intact and complete. All later18k frozen
evaluation budgets/seeds/controls/progress criteria are unchanged. This is
an exploratory training-budget revision after observing the first opening-game
win, NOT independent confirmation and not a claim that longer training works.

Reason:6k is an opening-game screen, but full-game training needs time beyond
the lab/rival; read-only logs also document much sparser/smaller reinforcement
than the unit-reward cue assay. The signal totals do not prove a bottleneck.
Keep BOTH lineages, final checkpoints only, no battle reset, no selected-stage
continuation and no reward/parameter change. New practice still starts from the
beginning with each fly's own existing synapses; exact own-game resume remains
available only to recover interruptions. User informed of this explicit change.
Use train_game_series.py --steps 18000 --training-seeds 4501 (or4502),
--evaluation-steps 18000 --eval-seeds 4601 4602 --defer-evaluation,
then the shared evaluate_game_retention_panel.py on ALL final practice brains.

2026-09-19 04:11 UTC: v10 seed401 learning arm COMPLETE6000, saved unchanged:
internal-learn-20260919T035428Z-c404bf.297 positions, starter3588, battle4246,
one completed rival win, final level6, no Route1. Its process31236 now runs
the separate matching original control internal-frozen-20260919T040951Z-2775e0.
Seed402 learning remains active in34878, internal-learn-20260919T040756Z-1ff349.
Only these two GPU jobs. The old59210 learning dashboard closed with its arm;
the second learner is at59212. Seed402 original's18 common final neural arrays
all reproduce historical data exactly; only fixed_release_reference is added.

Before ANY retained whole-game test outcomes, explicitly use the already coded
whole-game-progress-v1 screen for v10's4601/4602 panel too. Same lineage must
preserve completed milestones, add progression, and reach the SAME starter OR
battle endpoint at least10% earlier on BOTH seeds (censored originals use the
budget lower bound). If triggered, confirmation4603/4604 must support the SAME
lineage, not a different winner. ALL final brains remain included. This adds
no game reward, intervention or policy input; no default promotion on the first
positive6k run. Existing6k practice and18k evaluation budgets are unchanged.

2026-09-19 04:08 UTC: v10 first learning arm is still running but has achieved
starter3588, battle4246 and a rival victory, returned outside at level6 by5228.
This is actual gameplay, NOT yet retained-learning proof or a completed paired
comparison. Do not stop after this observation; finish BOTH registered pairs.
The seed402 frozen arm completed6000:188 positions, no starter/win/Route1.
ALL6000 actions/buttons/spikes/groups/motor rates/telemetry/rewards/input windows
match historical internal-frozen-20260918T162143Z-74e099 exactly. Shared control,
not new independent evidence. Its final checkpoint is step-00006000-88be37f0.

CURRENT TWO GPU JOBS remain31236 (401 learning, then original frozen) and34878
(402 NOW learning internal-learn-20260919T040756Z-1ff349, dashboard59212).
After completion audit both reports with audit_game_pair.py, validate seed401's
repeated original too, then apply the registered gate. If qualified carry BOTH
final gameplay brains through6k fresh whole games4501/4502, followed by ALL
final brains/shared original frozen18k4601/4602; no stage resets or assay weights.
Latest source-only commit72e8b25 pushed. No v11 run or default promotion.

2026-09-19 03:55 UTC: v10 independent memory gate PASSED BOTH training seeds.
Audit visual-retention-gate-20260919T035407Z-fc016f, ALL final8192 arms on
9101..9104. Original41.93%; paired50164.47%,60164.77%; shuffled36.60%/40.46%.
Paired cue accuracies65.22%/63.72% and65.91%/63.64%; all preset thresholds met.
This establishes generic retained association, NOT Pokemon learning. Retention
sessions38475/26864 exited/drained. Full CUDA regression527 Python/26JS/lint
passes (verify-007236faea4e4bc6bd7130f2d263d1f2),46583 exited/drained.

CURRENT ONLY TWO GPU JOBS: registered v10 full-new-game6k learn/frozen pairs.
401 learn-first session31236, visual-model-gameplay-20260919T035425Z-9d7a07;
actual learning internal-learn-20260919T035428Z-c404bf, dashboard59210.
402 frozen-first session34878, visual-model-gameplay-20260919T035447Z-db11a0;
first original control internal-frozen-20260919T035450Z-5c0d58, later dashboard59212.
ALL begin with original synapses and ordinary scripted intro, no loaded game
state or assay weights. Validate repeated originals against historical401/402
and label them SHARED controls, not independent replications. Follow the
predeclared6k development gate, then4501/4502 practice and4601/4602 retention
if passed. V11 fallback remains UNRUN unless a v10 gameplay/retention gate fails.

2026-09-19 03:42 UTC: v10 BOTH curves COMPLETE;16990/52362 exited/drained.
Final small paired/shuffled balanced scores:50165.30%/38.15%,60166.22%/39.04%.
These are preliminary probes, not the independent recall gate.
CURRENT ONLY TWO GPU JOBS: ALL final original/paired/shuffled frozen recall
on9101..9104.501 session38475, visual-retention-probe-20260919T034213Z-71e281;
601 session26864, visual-retention-probe-20260919T034227Z-f3a85a.
After BOTH finish, run summarize_visual_retention.py over both reports.
No v10 game run yet; if passed follow the registered6k401/402 queue below.
V11 remains conditional/unrun. Source-only progress-screen commited/pushed
ed014fe. Browser bootstrap now responds but reports the in-app browser absent;
troubleshooting guidance read, no page interaction/rendered verification possible.

2026-09-19 03:34 UTC: v10 paired8192 COMPLETE on both curves; small final
probes65.30%/66.22% balanced. Shuffled arms continue in16990/52362, still
the ONLY TWO GPU jobs. Finish all final arms before9101..9104 retention.
The complete physical model/default and game queue remain unchanged.

Observer-only prospective progress screen implemented/tested:517 CPU Python,
26JS/lint pass (verify-a522cee39ad64af9b81d05538e5771a7). CUDA unit module
was explicitly skipped while both GPU slots are occupied; no live neural
math changed. Read-only application to the old v7 raw panels rejects BOTH
lineages (game-retention-progress-screen-20260919T033241Z-c1953b); that is
audit regression evidence, not new gameplay trials or a prospective v7 test.
CPU audit27485 exited/drained. Conditional v11 profile/source and v9 results
were source-only committed/pushed8845514. V11 still has NO calibration/run.

2026-09-19 03:26 UTC: CPU regression after v11 profile preparation passes
501 Python/26JavaScript/lint, verify-a8f6084f249844d09cf27786265882dc.
Only GPU jobs remain the two v10 curves16990/52362. All ten protected
ROM/model/old-user-run hashes still match; no protected or generated artifact
is tracked by Git. The v11 asset has not been generated or behavior tested.

2026-09-19 03:24 UTC: prospective fallback v11 prepared, NOT RUN. Keep v10
first, including its registered gameplay/retention if its memory gate passes.
Only after a v10 gate failure, evaluate configs/visual-wide-score-v11.json.
This combines the EXISTING score-v2 equations/rate0.002/temperature0.05 with
the approved strong visual model and a separately frozen neutral calibration.
Its earlier weak-vision failure is preserved; this new combination is a
hypothesis, not a newly invented rule or evidence that credit is fixed.

V11 calibration: original synapses, seed707, gray128,10k steps, uniform1Hz,
wide[-.5,.5], no resets, reset probes301/302/303 for256 decisions. Use the
exact command in MODEL_VARIANTS.md; never overwrite another asset. Before
behavioral testing require finite reset-probe rates, each mapped anatomical
motor group in[.1,10]Hz and KC/MBON means<10Hz. These are coarse engineering
sanity checks, not a target button distribution or biological validation.
Then actual-ROM exact-resume/frozen/no-reward smoke and full regression.

V11 memory: paired/shuffled501/601 curves512/2048/8192, ALL final arms on
9201..9204,128 neutral+128 decisions/cue, SAME balanced>=.60,each>=.55,
advantages>=.05 vs original AND shuffled, BOTH training seeds. No assay
weights enter Pokemon. Physical firing/calibration differ, so old original
game controls cannot substitute; do not call this a one-factor physics test.

If BOTH pass, two full-new-game18k learn/frozen pairs401 learn-first,402
frozen-first. Carry BOTH final learned brains through another whole18k game
4701/4702, without selecting a better intermediate checkpoint or using a
battle reset. Unlike prior6k development screens, these18k budgets are fixed
up front so each lineage has36k full-game learning before testing retention.
No early-game success gate skips an unfavorable lineage. Then ALL final brains
and shared original on frozen whole-new-game18k panels4801/4802. Reserve
4803/4804 confirmation. Require no lost starter/completed-win/Route1 milestones
against the matched original, at least one additional completed progression
milestone, and >=10% earlier first-starter OR first-battle on BOTH held-out
seeds for at least one SAME lineage before confirmation. Report every arm.
This is a development threshold, not a significance test or endgame guarantee.
No default promotion without independent confirmation and fresh safety checks.

03:31 UTC prospective scoring clarification BEFORE any v11 run: require the
SAME faster arrival endpoint on both seeds, not starter on one and battle on
the other. If the original never arrives, use the evaluation budget as a
conservative lower bound on its time; retained must arrive by90% of that budget.
Missing arrivals never count as time zero. Preserve completed rival/capture/gym
outcomes too. The observer-only screen_game_retention.py re-audits all source
panels, scores every lineage, and never feeds measurements back to the brain.

2026-09-19 03:18 UTC: v10 software/live smoke checks PASSED, not gameplay
learning proof. Full CUDA regression510 Python/26JavaScript/lint:
verify-1ec9bc8cf1c044668492f1e5edb7a24a. Actual-ROM smoke
internal-smoke-20260919T030819Z-2b6041 verifies exact split/uninterrupted
arrays, rewards and actions, with frozen/no-reward controls unchanged.
Smoke37817 and regression85617 exited/drained.

CURRENT ONLY TWO GPU JOBS: v10 paired/shuffled ROM-free curves, final8192.
501 session16990, visual-learning-curve-20260919T031016Z-36197e;
601 session52362, visual-learning-curve-20260919T031016Z-c436d2.
Small paired2048 probes are59.34%/55.54% balanced; neither is the gate.
Finish BOTH complete curves, then ALL final arms on9101..9104 with the
previously registered thresholds. No synthetic weights enter Pokemon.
If passed, use game401 learn-first and402 frozen-first for the registered
6k whole-new-game comparisons. No battle-reset practice, no default promotion.

2026-09-19 03:08 UTC: v9 independent memory gate COMPLETE, FAILED both seeds.
Audit visual-retention-gate-20260919T030755Z-8f8eae; all final8192 weights,
full four-seed controls, no cherry-picked checkpoint.501 paired54.37%,
original41.50%, shuffled46.68%; cue accuracies70.31%/38.43%.601 paired60.95%,
original41.50%, shuffled38.26%; cues69.19%/52.71%. Both miss each-cue>=55%;
501 also misses balanced>=60%. Original raw choices reproduce exactly across
the two panels and count once. Do NOT run v9 gameplay4301/4401 queue or promote.
Panels visual-retention-probe-20260919T025755Z-58ec07 and ...T025756Z-3f54cf;
sessions79422/89487 exited/drained.

Read-only final-array check: fixed-reference mean-input residuals <5.24e-8
relative for all four paired/shuffled brains. Paired501/601 have6780/5241 of
495962 edges at the lower bound and106/91 at the upper bound. This verifies
the constraint, NOT behavioral stability or a proven cause of poor recall.

V10 fallback is triggered under its already registered protocol. CURRENT ONLY
TWO GPU JOBS: real-ROM CUDA smoke37817, full CUDA-inclusive regression85617.
No v10 learning-curve run or gameplay progress claim yet. After smoke succeeds,
use freed slots for the two501/601 v10 paired/shuffled curves. All thresholds,
reserved9101..9104 retention and later full-game rules remain unchanged.

2026-09-19 02:58 UTC: v9 curves COMPLETE; sessions40118/20407 exited/drained.
Final small-probe balanced paired/shuffled:50148.45%/45.07%,60166.77%/32.62%.
Neither intermediate score is the independent gate; no checkpoint selection.
CURRENT ONLY TWO GPU JOBS: full frozen9001..9004 retention panels79422 (501
curvef4889e),89487 (601 curve297315). All original/paired/shuffled arms,
neutral128 and128 cue decisions. Config/model/physical reference unchanged.
Source-only ancestry/audit/conditional-v10 commitd292221 was pushed; no v10
neural or game run yet. If v9 gate fails, execute the registered v10 fallback.

2026-09-19 02:51 UTC: final CPU regression after ancestry/one-reference audit
and conditional v10 profile passes500 Python/26JavaScript/lint:
verify-b0f2a68d230e4ddaa7cb02c64be61167. Reference-factor baseline composition
now verifies the exact copied intrinsic payload before accepting a shared
original control with a different asset filename; other physical settings
still must match. No live neural dynamics changed. Two v9 shuffled controls
continue; no other GPU jobs. Browser skill reread and connection retried after
the earlier pause, but bootstrap still times out. No tab or rendered UI claim.

2026-09-19 02:48 UTC: v9 BOTH paired8192 arms finished, shuffled controls
still running in the same ONLY TWO GPU sessions40118/20407. Small paired
probes are48.45%/66.77% balanced on501/601. These are NOT the held-out gate;
finish all controls and final9001..9004 retention. Never pick501's better2048
checkpoint. No v9 gameplay or final memory conclusion yet.

Register conditional one-rate fallback v10 BEFORE its behavior: if v9 fails
its memory or later registered gameplay/retention gate, test
configs/visual-wide-anchored-slow-v10.json. It changes ONLY rate0.2->0.02,
the established wide-v3/projected-v4 rate, with the SAME frozen reference and
all other parameters. Motivation: possible overwriting/high-variance updates;
the interim decline does not prove this cause. This is exploratory model
development, not an independent confirmation or a fitted best checkpoint.

V10 queue: CUDA exact-resume/frozen/no-reward smoke, then two paired/shuffled
ROM-free curves501/601 at512/2048/8192. ALL final arms test9101..9104 with
neutral128 and128 decisions/cue, existing balanced>=.60/each>=.55/advantages
>=.05 against original AND shuffled, both training seeds. No synthetic game
initialization. If passed, two full-new-game learn/frozen6k pairs401/402 using
evaluate_visual_gameplay.py (counterbalance arm order). Repeated originals
must reproduce historical physical trajectories and count as SHARED controls,
not extra independent trials. This tests v10 learning against its own frozen
model; it is NOT a v9-v10 gameplay dose comparison if v9 never enters gameplay.

Use the same retention-priority gate: both starters OR completed victory/Route1
without losing a matched original's starter/win/Route1. If passed, BOTH final
brains get another6k WHOLE new game4501/4502, then frozen18k panels4601/4602
with shared originals and ALL final brains. Reserve4603/4604 confirmation.
Keep the current v9 queue first; no third GPU job, no default promotion.

Whole-game source guard implemented: practice/retention checks original run
ancestry through checksummed parent checkpoints. Rejects stage resets anywhere
in the chain, cycles, ambiguous sources, ROM/model mismatches or lost provenance;
allows fresh power-on, explicit intro and exact own-game resume. Four actual
v7/v8 source lineages verify unchanged. CPU regression496 Python/26JS/lint
passes, verify-8697b1c7e8554d5caee2086e91ac5ccd. The preceding fixture-only
failure and then lint-only failure are preserved; checks were not weakened.
No neural runtime math or user checkpoint was changed by this guard.

2026-09-19 02:29 UTC: v8 6k development games COMPLETE, criterion FAILED.
401:239 positions, no starter/battle/Route1.402:208 positions, starter2333,
no victory/Route1. Neither matches the registered both-starters OR completed
victory/Route1 gate. Do not spend4201/4202 practice or3901/3902 retention on
this failed screen. No promotion or best-checkpoint selection. Raw starts,
all-fly actions, configurations, final checkpoints and original-control
prefixes revalidated by derived-game-baseline-20260919T022738Z-51cd29 (401)
and ...T022740Z-07a33c (402); these are audited compositions, not new trials.
Final brain hashes07b959a9a6ce49bd41b9b184ea2d0fccded014b8e3f739900ecd2faf709ad2ea
and6a597e0a4326d9f90835f2529032ba0b9ced82f93d02987cfaff4cdec6eadefc.
Both GPU sessions55468/78633 exited and were drained.

V9 generic collection COMPLETE, exactly the registered7201 protocol:
generic-release-reference-20260919T022608Z-c74a2a.16,384 measured neural steps,
no game/reward/buttons, original synapses and exact intrinsic payload unchanged.
New asset SHA256 b9c6c6718c6dd9cb3fa63df70dd5415ba5d7586638ff02ec297bebafc82717f6.
Real-ROM CUDA smoke PASSED for v9 (internal-smoke-20260919T022723Z-ddfd48)
and unchanged default (...T022635Z-3a0bbb), exact resume and both no-learning
controls included. Collection24079 and smoke40776/83506 exited/drained.

CURRENT ONLY TWO GPU JOBS: registered ROM-free v9 paired/shuffled curves.
501 session40118, visual-learning-curve-20260919T022816Z-f4889e;
601 session20407, visual-learning-curve-20260919T022817Z-297315.
Checkpoints512/2048/8192; all final arms subsequently held-out9001..9004,
same existing two-seed gate. Synthetic learned weights NEVER enter gameplay.
If passed, use the two newly derived v8 baselines for the registered v9
one-reference-factor full-game tests; all later4301/4302 and4401/4402 rules
remain unchanged. No third GPU job. Normal-history source is committed and
pushed as826c1ef; no user checkpoint/history was adopted by verification.

2026-09-19 02:15 UTC: CURRENT TWO GPU JOBS, registered v8 full-new-game6k
rule tests.401 session55468, report visual-plasticity-rule-gameplay-20260919T021258Z-d86466,
actual internal-learn-20260919T021311Z-beeb25, dashboard127.0.0.1:59200.
402 session78633, report visual-plasticity-rule-gameplay-20260919T021258Z-dd13aa,
actual internal-learn-20260919T021311Z-7004f6, dashboard127.0.0.1:59202.
BOTH fresh64-step frozen prefixes exactly matched original controls, including
actions/neural activity/rewards/telemetry/start state. Same predeclared gate and
queue; do not start a third GPU job. V9 collection/behavior still NOT started.

The v7 36k-trained held-out whole-game panels are COMPLETE and audited:
game-retention-panel-audit-20260919T020815Z-04563f. Originals2/2 starters and
2/2 completed rival wins; retained4101 1/2 starters,0 wins; retained4102 2/2
starters,1 win. No Route1/captures/badges/paid-unfinished outcomes in any arm.
Original positions370/404; retained4101 322/359; retained4102 351/332.4102
reached battle slightly earlier on3801 and later on3802. No consistent retained
advantage; no promotion or3803/3804 confirmation priority. Preserve negatives.
Panels64038/67732 and audit62215 finished/drained.

Normal-history CUDA application smoke PASSED:
normal-history-smoke-20260919T020700Z-1ef1bb;86434 exited/drained. Exact
auto-resume vs uninterrupted and retained whole-new-game learning vs explicit
weights, ALL arrays/rewards/actions matched. Actual user history unchanged.
Full CUDA regression491 Python/26 JavaScript/lint passed:
verify-e593caaceda54b2489b6e84f95bf5170;80222 exited. CPU and CUDA persistence
are verified, not a claim of improved gameplay. Source-only commit pending.

Tool recovery: standard apply_patch stalled twice before writing/launching;
waited, then terminated orchestration cells396/398.02:11 process check found
NO Python jobs, so only the two intended v8 launches were retried. Native
installed apply_patch helper works through approved elevated command calls;
use that for scoped edits while the standard sandbox launcher is stalled.
Do not terminate unrelated Codex/security services. Browser connection remains
unavailable after a waited retry; no real-browser layout verification claimed.

2026-09-19 02:06 UTC: panel3801 COMPLETE; session64038 exited/drained.
Retained4102:351 positions, starter6923, one completed-looking rival win
(final battle0, level6), no Route1; exact full audit awaits3802. Original3801
also won; retained4101 had no starter. Panel3802's final original arm is at17k.
One free GPU slot is now used for the full CUDA-inclusive regression, then
private-history CUDA smoke. CPU history smoke is finished; no third GPU job.
Browser skill read; bootstrap timed out, a simple connection retry after30s
also timed out. No DOM/layout claim, no alternate browser bypass or open tab.

2026-09-19 01:59 UTC: normal-history real CPU application smoke PASSED,
`normal-history-smoke-20260919T015516Z-b17e44`; session79864 exited/drained.
Auto-resume70->90 exactly matches a90-step uninterrupted run, and automatic
retained whole-new-game learning exactly matches an explicit weights-loaded
new game. All neural arrays, reward state and action logs compared; starts
match; frozen control preserves source weights and does not advance history.
Previous sources and the actual user's normal-history directory are unchanged.
This is persistence proof, NOT learned gameplay. CUDA confirmation queued in
the first freed GPU slot; currently only sessions64038/67732 on GPU.

Full CPU regression481 Python /26 JavaScript/lint clean, artifact
`verify-a02bce6a259d434b882278e65752db69`; PowerShell launcher syntax valid.
Optional v9 source-only commit f7cfd2f was pushed. Normal-history changes are
not yet committed; complete CUDA confirmation before its source-only push.

2026-09-19 01:52 UTC: found and fixing NORMAL launcher persistence gap. The
registered practice experiments retain synapses, but start.ps1 -Intro still
needed explicit -Weights. New opt-in CLI --remember (enabled by start.ps1)
tracks only its own model/ROM history, never the newest research directory.
Bare launcher resumes brain+game; -Intro retains weights into a whole new game;
-FreshBrain deliberately starts original weights. Old runs require explicit
Resume/Weights adoption because their user/research origin is not identifiable.
Frozen/no-reward runs do not advance this history. Corrupt/missing history
fails closed; an OS-owned lock prevents concurrent same-model normal sessions.
Checkpoint callbacks update only after a complete checksummed generation.
Existing research train() calls, model identities, rewards and neural math stay
unchanged. The normal launcher rejects stage-specific LoadState.

478 CPU Python tests passed before additional malformed/CLI cases; lint-only
formatting was fixed. Added private-history real-ROM smoke (NOT launched yet),
which compares auto-resume with uninterrupted gameplay and auto-retained new
game with explicit -Weights, including ALL saved neural arrays/reward state.
It must not create/adopt a brain in runs/user-history for the user. Run this
short check in a free GPU slot after the v7 panel; v8 paired trials follow.
No third GPU job, no battle reset, no promotion based on persistence plumbing.

2026-09-19 01:40 UTC: final v9 source regression clean: 463 Python tests,
26 JavaScript tests and lint, `verify-7b3dba6f197a45f7823bac8c3a299ffb`.
CUDA/live tests and generic-reference collection remain queued, not passed.
Default launcher unchanged. Both GPU panels are now on their third/final arms.

2026-09-19 01:38 UTC: four of six v7 frozen arms complete. Retained4101 fails
to acquire a starter on3801 (322 positions), acquires one at5106 on3802
(359 positions), zero wins/Route1 in either. B fractions53.08%/59.97%; source
weights stayed exactly frozen. Original3801 already had a starter and victory,
so this lineage does not show a consistent retained benefit. Remaining third
arms are retained4102/3801 `internal-frozen-20260919T013556Z-c16699` and
original3802 `internal-frozen-20260919T013700Z-1dff5c`. Sessions64038/67732;
no other GPU jobs. Complete/audit the full panel before any queue decision.

Protected ROM/connectome/user-stopped-run manifest: all10 hashes still match
at01:36UTC. Git has no tracked ROM, saves, neural assets or runs. No files were
deleted. The v9 reference now also binds retina and all five pinned visual
reference files, not only the connectome. CPU suite463 passed; last lint-only
long line corrected, final wrapper rerun pending. No v9 calibration collected.

2026-09-19 01:33 UTC: full CPU regression after v9 core/collector: 460 Python,
26 JavaScript and lint pass (`verify-55a8cfae963f465899bc89165f73413b`). CUDA was
explicitly skipped because both slots remain occupied. This is implementation
verification only. Added an explicit `release_reference` one-factor comparison
mode: moving v8 -> fixed v9 only, requires bit-exact original intrinsic payload
and recorded original-file checksum; rejects changes in every other setting.
It still requires a new exact frozen 64-decision prefix before actual training.

Prospective v9 protocol (no collection/behavior seen yet): after the registered
v7/v8 priority jobs, collect the generic reference once with fixed7201 defaults;
run the real-ROM exact-resume/frozen/no-reward smoke and old-default regression.
Then ROM-free paired/shuffled cue curves on501/601 at512/2048/8192, followed by
ALL final-weight original/paired/shuffled retention on unused9001..9004 using
the EXISTING gate (balanced>=.60, each cue>=.55, advantage>=.05 vs both controls,
both training seeds). No best-checkpoint selection or synthetic-to-game weights.

If both visual gates pass, compare v9 against completed v8 6k development games
with the explicit reference factor, seeds401/402, same whole fresh game and
original synapses. The same retention-priority criterion applies: both starters,
OR a completed victory/Route1 without losing matched-original starter/win/Route1.
If passed, both final brains get another6k fresh-game practice (4301/4302) and
then frozen18k whole-game panels4401/4402, sharing one original per test seed.
Reserve4403/4404 confirmation. Named new-model lineages persist across practice;
normal user runs never discard weights silently. Whole-game progression,
not cue success, numeric constraints or prettier button counts, is the target.

2026-09-19 01:28 UTC: optional v9 implementation prepared, NOT calibrated or
run on the GPU/game. New collector performs only generic-image neural steps
on original weights, checks unchanged synapses/biases, and creates a new asset
with the exact old intrinsic payload. The reference loader checks physical
configuration, connectome hashes, neuron IDs, payload hash and no-game/reward/
button provenance. Reference arrays are immutable and checked on restore.
No new config fields/identity keys are added to legacy models.

First CPU regression: 459 passed, one NEW test failed because its expected
float32 values were accidentally promoted to float64. Corrected the expected
value to the exact saved float32 array; no tolerance relaxed. Full rerun
pending. This is not a behavioral result and no default is changed.
GPU sessions64038/67732 continue their second arms; do not exceed two jobs.

2026-09-19 01:13 UTC: first two frozen arms completed, NOT the full panel.
3801 original: 370 positions, starter 7,759, one rival win, no Route 1.
3802 retained_4102: 332 positions, starter 3,035, no wins/Route 1, level 5.
Both next arms are retained_4101, still sessions 64038/67732, actual runs
`internal-frozen-20260919T010647Z-72a217` (3801) and
`internal-frozen-20260919T010649Z-3f4725` (3802). Finish all registered arms.

CPU-only capacity warning for optional v8: a two-input counterexample shows
that repeatedly constraining original mean input in two different contexts
can erase a hand-set response contrast while satisfying every numeric bound.
This is NOT observed fly behavior or a trained checkpoint. A fixed mixed-context
reference preserves that toy contrast. Reproducible in
test_moving_context_constraint_can_erase_selectivity_unlike_a_fixed_reference;
all 19 pure numeric tests pass. V8 remains unpromoted and behaviorally untested.

Prepare an OPTIONAL anchored-reference v9 alternative, not a mutation of v8:
same existing synapses, bounds, rate and frozen physical model. Collect a fixed
release mean with original synapses on generic visual patterns ONLY, no ROM,
reward, button optimization or game images. Store it alongside an EXACT copy
of the original intrinsic biases in a NEW versioned calibration artifact;
never overwrite the old asset. The new rule constrains against this fixed
reference, not the changing gameplay context. Preserve old identities and
all old algorithms. No learned external readout/critic and no trained assay
weights may enter gameplay. This alternative is not yet implemented, calibrated
or behaviorally validated. Do not launch its collection or CUDA checks while
the two frozen panels occupy both slots. Existing v7/v8 schedules remain;
register any v9 behavioral comparison before seeing its outcomes.

2026-09-19 00:55 UTC: final clean CPU-only regression passes 420 Python /
26 JavaScript / lint, `verify-57f958c6c8eb4791a21fa5f5d120acc0`.
No active child agents. Only GPU sessions 64038/67732 remain; first panel arms
are around 10,000/18,000. CPU replays and test processes have exited. Original
3801 has a rival victory; retained_4102 on 3802 has a level-5 starter. These
different-seed interim arms are NOT a matched comparison; finish all six arms.

2026-09-19 00:49 UTC: CPU move-confirmation replays COMPLETE; session 37997
exited/drained. Both source 402 victories reproduce all 18,000 sampled states
and rewards from the whole-game start. Last confirmed move to final-faint
reward: 8 decisions / 180 game frames / 1.92 neural seconds in its initial
game; 9 / 209 / 2.16 in practice 4102. At the active 0.6 s eligibility constant,
an isolated trace impulse decays to 4.08% / 2.73%. This is a passive-decay
calculation, NOT measured remaining eligibility, causal move attribution or
proof that extending the trace will help. Earlier long/dual-trace failures
remain relevant. No new rewards or training episodes were generated.
Artifacts `recorded-outcome-latency-20260919T004440Z-8c5c7f` and
`...T004507Z-0776d6`. The accepted move hook is signature-checked against the
pinned disassembly; the cursor-hover RAM field alone is not a confirmation.
420 CPU Python tests pass; two line-length lint issues were then fixed, and
lint plus 26 JavaScript tests pass separately. No CUDA tests were launched.

Practice 4101's first sampled battle is 15,271. Frozen panels 64038/67732
continue unchanged; do not launch a third GPU job. No new predictive-memory
model is implemented. Such a proposal needs verified internal anatomy and
controlled tests, not a trained external critic or invented MBON button labels.

2026-09-19 00:45 UTC: both whole-game practice runs FINISHED at 18,000 decisions.
Sessions 50606/39030 exited and were drained. Each lineage now has 36,000
actual-game training decisions across two fresh starts, with unchanged v7.

- Source 401 -> 4101: 204 positions, starter 13,640, reaches rival battle but
  it is unfinished at cutoff; no victory/Route 1. B in 12,112/18,000 windows.
  Final `internal-learn-20260919T000217Z-c17cf7/checkpoints/step-00018000-9e259df9`,
  brain SHA256 `90493985dc1f1bcf9a1d96d13277ebbc1b7ac78645433e8752fe9a425b7455a0`.
- Source 402 -> 4102: 351 positions, starter 4,822, first sampled battle 4,880,
  rival victory reward 5,962, final level 6; no Route 1. B in 5,562/18,000 windows.
  Final `internal-learn-20260919T000218Z-e62826/checkpoints/step-00018000-0af9bc24`,
  brain SHA256 `457f6ecbfec2ebd8b4942cdf9b44e5c2b5b09781e7df88877ffa0502c906711c`.

These are practice observations on new noise seeds, NOT matched improvements.
Both registered frozen panels NOW RUNNING, ONLY TWO GPU jobs:
- 3801: session 64038, `game-retention-panel-20260919T003820Z-25c5e2`.
  Arm order original, retained_4101, retained_4102. First actual run
  `internal-frozen-20260919T003823Z-68739b`.
- 3802: session 67732, `game-retention-panel-20260919T003821Z-6e893d`.
  Arm order retained_4102, retained_4101, original. First actual run
  `internal-frozen-20260919T003823Z-30d20e`.
Each arm gets 18,000 decisions from the same whole fresh game, all weights
frozen. Do not call a single seed or a partial arm a successful panel. Audit
BOTH reports afterward with summarize_game_retention_panel.py; reserved
3803/3804 confirmation if the complete result looks beneficial. V8 stays queued.

CPU-only full-game move-to-outcome replay is session 37997: source 402's two
recorded victories, never battle-start practice or new wins. It is measuring
accepted move selection, not the RAM field that also changes on cursor hover.
Pending; no new timing result yet. The earlier v7 pair audit was recomputed
with battle-access timings: `full-game-pair-audit-20260919T003615Z-fd7c0e`.
No new trials: first battle learn/original is 3,622/6,400 (401),
11,818/7,915 (402). The snapshot state becomes battle-active slightly before
the game's battle-start execution hook; distinguish those measurements.

2026-09-19 00:30 UTC: queued v8 behavioral protocol, AFTER the already registered
v7 practice/3801+3802 retained tests (and reserved confirmation if promising).
Use the existing one-factor runner, factor=rule, 6,000 whole-game decisions on
development seeds 401/402. Original synapses initialize this DIFFERENT model's
named lineages; no v7 checkpoint migration or synthetic weights. Same fast-v6
rate, visual model, physical dynamics, buttons and rewards. No battle starts.

The failed fast-v6 parents and their original controls are explicitly shared,
not new replications. New CPU provenance compositions:
`derived-game-baseline-20260919T002854Z-427dea` (401) and
`...T002855Z-663e36` (402). They recheck complete logs, actual original frozen
weights, final candidate checkpoints, matching starts and the recorded 64-step
frozen prefix. The runner must also reproduce a NEW candidate frozen prefix
before training. No v8 behavioral trial has started yet.

Retained-test priority gate, declared before v8 behavior: both development
games acquire starters, OR at least one completes a victory/reaches Route 1
without losing a starter/win/Route-1 outcome achieved by its matched original.
Coverage or a more balanced button histogram alone does not pass. If it passes,
continue BOTH final v8 brains through another 6,000-decision whole new game
(noise 4201/4202), then freeze BOTH final lineages on held-out 3901/3902,
18,000 decisions per arm with one shared original per seed. Those longer
tests—not the development gate—judge useful retained gameplay learning.
If the gate fails, keep the failure and diagnose before another variant;
do not select an earlier checkpoint or silently replace the current default.

CPU baseline-composition regression now passes 416 Python / 26 JavaScript /
lint, artifact `verify-a64fb057c4994468a61837afc832c85c`. A test import-order
lint error was corrected before this complete pass. No additional GPU job.
Read-only mechanism investigation is recorded in [LEARNING_MECHANISMS.md](LEARNING_MECHANISMS.md):
the active reward average is not a learned state-value predictor. Primary
fly-memory studies motivate investigating internal feedback, not an external
critic or another unvalidated circuit change during the registered tests.

2026-09-19 00:23 UTC: both registered whole-game practice jobs remain active,
near 10,000 of 18,000 decisions each. Do not select intermediate weights or
replace either source. Source 401 is still moving but has stalled at 203
positions, with B in 362/500 recent windows; source 402 has a level-6 starter
and 327+ positions. These are interim observations in different contexts, not
a retained-learning comparison. Frozen panels 3801/3802 remain pending.

Before those panels begin, their observer-only audit now reports first sampled
battle access, first victory-reward delivery and completed versus still-open
paid encounters. Aggregate battle wins exclude unfinished encounters. No
reward/model change; no timing bonus or battle reset. Missing/non-fly action
provenance and inconsistent final reward ledgers are rejected.

Full-game recorded replay `recorded-outcome-latency-20260919T000938Z-c898dc`
verifies all 18,000 states/rewards from v7 learning 402. Its existing victory
reward arrives at final faint, decision 12,523; encounter end is 26 decisions
later. This measures outcome-hook to reward delivery, NOT move-selection
credit latency, and is not another autonomous win or training episode.

CPU-only regression: 406 Python / 26 JavaScript / lint pass,
`verify-6d09e3b1a6724df39d729b0af4608008`. Ten CUDA tests deliberately omitted
while both GPU slots are occupied. A direct pytest attempt hit the known
Windows temp-directory permission issue; the isolated project test wrapper
then passed. Use scripts/test.ps1, not the shared system pytest temp directory.

2026-09-19 00:02 UTC: both18k v7 pairs COMPLETE and audited, mixed/negative
for consistent improvement. Audit `full-game-pair-audit-20260919T000100Z-980a15`.
401 learn/original:346/395 positions, starter3122/6284, no wins; Route1 first
8913/7902 and minimum y34/26.402 learn/original:413/313 positions, starter
10936/7780, one completed rival win each (12523/8955), neitherRoute1.
No paid-but-unfinished encounter, capture or badge in any arm. Earlier-timing
learning delivered an actual victory reward, but did not demonstrate a reliable
whole-game advantage. These are not retained-transfer results or a comparison
against late-reward learning at18k. All original6k prefixes are shared history.

Full408 Python/26JS/lint pass, artifact
`verify-287e071c85d64e1b9cd0857b23a0b57c`. Real-ROM/CUDA70/20-resume/frozen/
no-reward smoke passes for optionalv8 (`internal-smoke-20260919T000017Z-24f7dd`)
and unchanged showcasewide-v3 (`internal-smoke-20260919T000050Z-6bd19f`).
Neither smoke is a gameplay-learning result. All old job sessions are exited.

Registered whole-game continued training is NOW RUNNING, ONLY two GPU jobs:
- source401 -> noise4101, session50606, series
  `retained-game-series-20260919T000214Z-e85cd9`, actual run
  `internal-learn-20260919T000217Z-c17cf7`, dashboard8784.
- source402 -> noise4102, session39030, series
  `retained-game-series-20260919T000216Z-e4554f`, actual run
  `internal-learn-20260919T000218Z-e62826`, dashboard8785.
Each loads its predecessor's FINAL18k synapses, same v7 settings,18k additional
decisions in a whole fresh game. No battle reset, profile upgrade or selected
intermediate weights. Both reserved3801/3802 frozen18k panels remain pending;
run one shared original per seed after BOTH trained sources finish. No GPU
assay or homeostatic behavioral trial should be launched alongside these jobs.

23:53 UTC prospective next whole-game phase: learning402 has now won its
rival at12523, level6; its matched original won earlier at8955. This is an
actual early-timed training reward, not evidence of improvement yet. Finish
both18k pairs and audit them. Then continue BOTH exact final v7 brains through
one more whole new-game learning attempt each,18k decisions, noise4101/4102
for initial401/402 respectively. Each fly will have36k actual-game training
across two full-game starts, never a battle reset. No intermediate checkpoint
selection, changed rewards, synthetic weights or model/rate changes.

Use train_game_series --initial-game-run ... --training-seeds <4101 or4102>
--eval-seeds3801 3802 --steps18000 --evaluation-steps18000 --defer-evaluation.
The new option pins the final brain as training_completed/evaluation_pending;
it does not mark a learning study successful or completed. Then evaluate BOTH
final brains on3801/3802,18k decisions per arm, learning OFF, one shared original
per seed (six arms total). Compare starter/battle access, victories and areas,
not just weights, total rewards or positions. Frozen follow-up3803/3804 is
reserved if the complete first panel looks beneficial. The older1601/1602 and
3701/3702 data are not recycled as new confirmation.

One additional practice attempt is allowed only when a completed actual-game
learning source is explicitly supplied; a new from-original series still needs
at least two attempts. Recovery preserves the exact interrupted game and the
planned deferral. Numerical v8 work remains separate and unpromoted. Run its
real-ROM smoke and full CUDA-inclusive suite in a freed slot before continuing
practice; no third GPU job. Future v8 behavioral comparisons are still queued,
not an excuse to discard the current flies' learning histories.

Verification encountered one Windows10053 loopback connection abort; the
30-test dashboard retry and subsequent complete397-test CPU run passed.
Three line-length lint failures were corrected; separate lint and26JS tests
then passed. No observer/control assertion was removed or relaxed.

23:39 UTC: launcher now explicitly labels NEW brain versus retained synapses
versus exact game/brain continuation. Loading remains explicit, never an
automatic choice of the most recently modified research checkpoint. The
latest full CPU suite after the safeguarded solver passes387 Python/26JS/lint
(`verify-ad7181d375ab4e2687bc0504918524c1`); seven additional launcher/legacy
constraint tests subsequently pass in the30-test focused subset. Six parity
cases check the old clipping/normalization operation order bit-for-bit for
positive and separately budgeted signed inputs. Default and live jobs unchanged.

23:35 UTC: v7 learning401 completed18k: starter3122,346 positions, Route1
minimum y34, NO victories. Its frozen comparison now runs in the same session,
`internal-frozen-20260918T233132Z-c90b1d`. Frozen402 completed18k: starter7780,
rival victory8955,313 positions, noRoute1; learning402 now runs
`internal-learn-20260918T232612Z-c14eea`. These are NOT matched final effects
yet. All6,000 initial raw records in the first two arms match their old shared
development prefixes exactly (apart from run ID/wall timing), not extra trials.

Optional prepared mechanism candidate: `visual-wide-homeostatic-v8.json`
changes ONLY the internal rule relative to failed fast projected-v6, retaining
rate0.2. Existing positive synapses are constrained against ORIGINAL estimated
input at the CURRENT local release means, with factor bounds and the same25%
resource budget. Unlike incremental projection, this addresses accumulated
mean-input drift and does not break the mean constraint when clipping occurs.
No action labels, desired motor rates, RAM features or external critic enter it.
Frozen mode does not perform homeostasis or change weights. Original profiles
and checkpoint identities remain unchanged. This is an engineering hypothesis,
not measured fly physiology or established useful learning.

Private-copy CPU audit initially exposed a numerical solver failure for
nearly-silent float32 release traces, plus an overly strict absolute test
tolerance. Corrected target normalization/bracketing and added subnormal tests.
Final safeguarded-Newton audit `credit-update-audit-20260918T233346Z-d87f04`
reduces maximum original-input residuals from51.05%/43.09% to<5.7e-8 relative
on BOTH sources for all three hypothetical rewards. Sources were hash-checked;
no corrected arrays were exported or used in gameplay. Measured warm CPU cost
~17..19ms/update here; no whole-application speedup claim. These are numeric
constraint checks, not behavior, and cannot establish a cure for B dominance.

CPU suite passes387 Python/26JS/lint before the final solver acceleration;
14 focused tests pass after it. Artifact `verify-f28e10b849824e15b02d3c3ebb54728c`.
No v8 GPU trial has begun. Finish the ongoing v7 pairs; when a slot becomes
available, run full regression/real-ROM smoke before any v8 behavioral test.
Synthetic cue diagnostic weights must never initialize a game. Any subsequent
game practice carries a named lineage across whole-game starts, never battles.

23:19 UTC USER REQUIREMENT: whole-game training only. The user explicitly
declined focused battle-start practice. Across new-game training attempts,
carry forward the same fly's learned synapses; it must learn to reach battles
again, not be placed there. Exact continuation of its own stopped game is
distinct from a stage-reset curriculum. Original-weight/frozen arms are
explicit controls only. Earlier pending-question entries below are historical
and now resolved: NO battle-start episodes. Keep all historical evidence.

The two existing full-game v7 pairs survived the conversation interruption;
no replacement jobs were launched. Reports `visual-model-gameplay-
20260918T230017Z-c8f22a` (401) / `...T230049Z-fade7a` (402), sessions30649/65999.
The regular launcher already supports full resume and weights-only new games;
plain -Intro does NOT load previous learning. Do not silently select the most
recent research checkpoint as the user's fly. Continued training must use an
explicit saved-brain lineage, not repeatedly initialize original weights.

CPU-only regression suite passes373 Python/26 JavaScript/lint
(`verify-aaae4a787817449fa2f7256f97ad09c8`); ten CUDA kernel tests explicitly
omitted while both GPU slots are occupied. No claim of rendered browser QA.

23:09 UTC credit audit `credit-update-audit-20260918T230848Z-497d86` detects
the fast projected candidate's mean-input projection being broken by subsequent
factor/resource bounds in hypothetical private-copy updates. This is a
mechanism lead, NOT a causal explanation or a new trained brain. At the final
saved presynaptic means, one B-readout input estimate is +51.05% in401 and
another +22.07% in402 relative to original weights. Moving release means and
recurrence also matter; a zero first-order update is not constant neural firing.
No fix or new default has been promoted from this diagnostic.

23:01 UTC: BOTH fast projected trials COMPLETE, no starter/win/Route1.
Seed401220 positions/B81.52%; seed402251/B43.20%. Parent projected185/290,
starter only401; original frozen268/188. Rate increase fails the declared
retention-priority criterion, is not promoted, and is not rescued by selecting
a favorable intermediate checkpoint. Reports `visual-learning-rate-gameplay-
20260918T224350Z-e6e62f` / `...T224450Z-a47070`; all negative evidence retained.
Longer earlier-outcome tests begin: session30649 learns401 then freezes its
original control,18k per arm;402 uses reversed arm order, also18k each.
Both start original synapses with the same intro; no training-stage resets.

The sandbox's NEW terminal launcher stalled on two read-only diagnostics;
waiting/retrying did not recover it. Canceled those orchestration requests,
not the game processes. Approved project-scoped elevated launches work.
Read-only process-tree checks found exactly two experiments plus their venv
launchers, no matching diagnostic processes; disk free432,753,778,688 bytes.
Do not launch extra jobs to compensate for tool latency. Existing experiment
sessions continued normally and both fast runs exited with complete artifacts.

22:49 UTC ledger: fast projected401/402 are the ONLY two GPU jobs, sessions
15910/49741, reports `visual-learning-rate-gameplay-20260918T224350Z-e6e62f`
and `visual-learning-rate-gameplay-20260918T224450Z-a47070`, actual learning
runs `internal-learn-20260918T224403Z-c15ce2` / `...T224502Z-9e4596`.
Both frozen64 prefixes exactly reproduce the shared original controls.
Earlier-outcome v7 real-ROM smoke passed
`internal-smoke-20260918T224333Z-520129`. Browser skill was read and connection
retried; even a minimal connection check times out, so no rendered UI QA is
claimed. Read-only live HTTP/SSE checks of both active dashboards verify raw
screen/retina/input-window synchronization and disabled manual controls
(observed samples1908/1341). Experiments are not interrupted by that UI limit.

22:44 UTC: independent3701/3702 follow-up COMPLETE and NEGATIVE. Original
starter1/2; BOTH practice brains0/2 each; no wins/Route1. Source401 tiles292/
265, source402254/295, original278/246. Audit passes
`game-retention-panel-audit-20260918T224333Z-7384d6`. Do not promote either
practiced brain or call the earlier win a robust learning result. Full suite
377 Python/26 JS/lint passes (`verify-b0c0236d5d7d450b8cfc958d95886b0f`).
Session15910 now runs fast projected development401; session91743 is the
v7 actual-ROM smoke. After smoke exits, start fast402 in the other slot.
Only these two GPU jobs. The independent retention scripts have exited.

22:43 UTC pre-training decision rule for the queued fast projected games:
tile coverage alone is not enough to prioritize retention. Both final brains
will go to3601/3602 frozen retention if BOTH development games acquire a
starter, or a new battle victory/Route1 visit appears without losing starter
acquisition on seed401 (the projected parent's successful seed). This is a
queue-priority criterion, not proof or a statistical gate. Otherwise finish
and retain both negative/mixed dose trials, then run the registered longer
earlier-outcome pairs. Do not select an intermediate successful checkpoint.

22:39 UTC: reward-timing replay now verifies TWO actual wide-v3 victories,
with the same rewards32/45 decisions earlier under the existing last-faint-v3
hook and unchanged novelty. Artifact `reward-timing-audit-20260918T222844Z-22372c`.
Prepared `visual-wide-outcome-v7`, changing ONLY that timing versus wide-v3,
not learning rate/rule/eligibility/physics or reward categories/values. Not
promoted or trained yet. The first full-game fast-rate pair remains next.
If it does not show a consistent improvement worth immediate retention tests,
the next full-game experiment is two original-weight learning/frozen pairs
with v7, seeds401/402,18,000 decisions per arm, no mid-game resets. This tests
learning with the earlier reward in a longer opportunity window, NOT an
isolated claim of improvement over late-timing learning at18k (that matched
late-learning control has not been run). Existing6k prefixes are shared
development evidence, never additional independent trials. Any apparent
benefit still needs fresh frozen retention, preselect3801/3802. Do not use
the synthetic cue brains or infer permission for focused battle practice.

22:25 UTC: first independent follow-up arms complete: original3701 gets a
starter at2114 (278 positions, no win); retained source402/3702 gets no starter
(295 positions). These are DIFFERENT seeds, not a matched effect estimate;
finish all six registered arms. Preserve this counterevidence alongside the
earlier promising result. New CPU-only verification passes366 Python/26 JS/lint
(`verify-61b91f0b43d949d5b9b401967abee6e8`), explicitly excluding the ten CUDA
kernel tests while both GPU slots are occupied. A preceding ad-hoc invocation
failed fixture setup because its temporary PARENT directory did not exist;
the normal script creates that parent and now supports `-SkipCudaUnitTests`.
No product-code failure was suppressed. All ten protected file hashes match.

22:21 UTC: independent retention is running in sessions85820/9637, reports
`game-retention-panel-20260918T221429Z-8b795c` and
`game-retention-panel-20260918T221440Z-d732f1`. Only these two GPU workers
are active. The final-practice neutral probe completed: originalB110/384,
source401219/384, source40291/384; full original rows exactly match the earlier
shared probe. These are heterogeneous tonic changes, not a contextual-learning
claim. Fast projected candidate real-ROM smoke passed
`internal-smoke-20260918T221315Z-c31c77`; full rate games remain queued.
Added a complete-panel audit with seven passing fabricated CPU-fixture tests;
do not report unfinished actual panels as confirmation. Console progress now
separates `edges_vs_original` from `updated_now`: a frozen trained brain can
have many retained changes but ZERO new updates. This is labeling only; neural
rules, trajectories, saved metrics and showcase defaults are unchanged.

22:14 UTC: complete practice audit passes (`retained-game-series-audit-
20260918T221314Z-fe4d56`): original0/2 starters/no wins; source401 retained1/2
starters/no wins; source402 retained2/2 starters/one rival win. All weights
frozen, every raw sample checked, repeated controls exact and counted ONCE.
No Route1 in any arm. Source402 had NO training battle victory, so its frozen
win is not evidence of learning specifically from battle-win reward.

This newly completed result changes experiment PRIORITY, not the model or
the old results: before the queued fast-rate games, independently repeat the
fresh-game retention comparison for BOTH final18k brains on preselected
unused3701/3702,6000 decisions per arm, one shared original per seed. Six
arms total; exact final training checkpoints, no checkpoint selection, no
new learning, no game-state assistance. Assess starter acquisition, victories,
new areas and levels together with coverage; do not equate less walking with
less playing. The original1601/1602 evidence is exploratory for this follow-up,
not additional new controls or a statistical-significance claim. Keep future
3601/3602 reserved for the fast candidate. The rate candidate is still queued,
not canceled or promoted. At most two GPU jobs including smoke/unit tests.

22:11 UTC: both final 18,000-decision practice brains now have at least one
starter acquisition in their frozen fresh-game panel where the matched
original does not. Source402 also won the rival in frozen1602 (one encounter,
level6). This is more promising for PLAYING than tile coverage alone: all
retained arms still cover fewer positions. Finish the registered final
original control and audit both complete panels before interpretation.
Next, repeat the preselected neutral motor probe3201..3203 on BOTH final
practice brains,128 warmup plus128 scored decisions/seed, all seven groups,
learning/rewards off. A tonic bias change is not automatically harmful;
compare the observed interaction outcomes, not just proximity to original
button frequencies. This is a mechanism probe, not another gameplay trial.
The pending question concerns focused battle practice only; the already
authorized full-game experiments continue without waiting for that answer.

21:49 UTC: targeted replay of the actual rival victory finds a remaining
battle-credit issue in the wide-v3 profile: final faint at5682, trainer-win
confirmation5698, delivered reward5714. That is32 decisions /7.68 neural
seconds after the faint, versus the0.6s fast eligibility trace. Artifact
`recorded-outcome-latency-20260918T214659Z-faa7b2` verifies all1,500 resumed
samples/states/rewards. The existing `last-faint-v3` hook is opt-in and NOT in
wide-v3; it was previously tested on weaker profiles, not combined silently
here. No new outcome/category is proposed. Asked the user ASYNCHRONOUSLY
whether focused actual-battle practice resets are acceptable; until answered,
continue the already registered full-game studies/rate test, do not assume
approval for that proposed training curriculum or stop ongoing experiments.

Source401's first frozen post-practice test1601 completed NEGATIVE:221 tiles
versus original250, no starter/wins in either; exit3922 versus247. Retained
B47.30% versus original28.23%. Exact recorded menu audits show23.26% versus
32.85% menu time (`recorded-menu-audit-20260918T214533Z-7821fe` versus
`...T213514Z-d953cf`). Less menu time does NOT establish better gameplay.
All remaining registered held-out evaluations continue. Source402 completed
its second practice1404:159tiles, starter5542, no win; no favorable episode
or source is selected. The two original1601 controls agree through2500
records so far; final audit will verify every record and count one control.

Exploratory CPU-only observations (NOT model changes): three original frozen
games have mean same-direction bouts1.19--1.20 decisions, median1/p95=2;
raw command jitter may limit navigation but does not establish a readout fix.
Saved natural screens retain varying raw retinal intensities. About3--5% of
calibrated visual voltages exceed the rate ceiling and~65% are rectified to
zero, similar to the synthetic cue snapshots; no blanket saturation diagnosis
is justified. Voltages are not spike counts/Hz. Larger projected hypothetical
single updates preserve mean input for tile rewards to rounding precision,
but reward1 clips135/181 of~495k edges at the lower bound, leaving~0.2%
residual in the projected update. These are PRIVATE CPU-copy diagnostics,
not new training, and no diagnostic weights were exported. Check actual
tonic drift again if the queued faster-learning model is run.

21:32 UTC scheduling correction: the default unit suite also performs tiny
real CUDA-kernel checks. Earlier full-suite calls briefly used a third CUDA
context alongside the two experiments; those test processes all exited and
none are orphaned. Count full-suite verification in the two-GPU budget from
now on, or explicitly run CPU-only subsets while both experiment slots are
occupied. No additional full CUDA suite until a slot is free. Latest completed
suite:366 Python/26 JS/lint (`verify-f926e3146d274587916c48b63e47bd4c`).
The new retained-series audit checks every registered evaluation arm, frozen
weights, matched start states and full duplicate-control trajectories; it
counts a rival win as ONE battle, not battle_win plus rival_win. Seven tiny
protocol-fixture tests pass, but are NOT behavioral evidence. Actual panels
are still running. Source401's final practice won one rival encounter at5714
and reached level6; this single win is not retained-learning proof.

21:21 UTC: projected-wide PASSES the complete unchanged independent visual
retention gate for BOTH training seeds. Paired balanced71.445/68.625%, original
shared40.698%, shuffled36.437/39.166%; paired cues72.11/70.78 and70.28/66.97%.
Gate artifact `visual-retention-gate-20260918T212057Z-3b0724`, final8192 weights,
neutral warmup128, fresh3001..3004,128 decisions/cue/seed, all learning/rewards
off. This is limited retained cue association, not Pokemon competence or a
statistical significance claim. Assay weights NEVER enter the game. This
satisfies the prospective prerequisite for the later projected-rate test.
Two registered real-game practice studies now resume from4500 in sessions
9818 (source401, `resumed-game-series-20260918T212058Z-14a067`) and91866
(source402, `resumed-game-series-20260918T212058Z-5e8d18`). These are the ONLY
two GPU jobs; both finish their original training schedule then frozen1601/
1602 comparisons. Candidate-rate games remain queued, not extra workers.
Full suite359 Python/26 JS/lint passes at21:20
(`verify-996b60e4725e46da89472a486488b83a`). A prior full-suite attempt had one
Windows HTTP connection-abort in an unchanged rejection test; the dashboard
suite alone30/30 and full rerun pass. No flaky-test bypass or guard weakening.

21:13 UTC: causal-credit402 also underperforms:128 positions/no starter,
versus wide-v3 learning255 and frozen188. Both causal-credit candidates fail
the gameplay improvement screen; leave opt-in and do not select a favorable
trajectory or spend the retention queue on them ahead of stronger evidence.
Projected-wide independent3001..3004 retention for BOTH recovered curves is
running. If that unchanged gate passes, the next prospective one-factor test
will increase ONLY its internal learning rate0.02->0.2 (bounded, same existing
edges/input projection, original synapses,401/402,6000each). It tests greater
update dose after tonic drift was reduced. The older fast-learning-v2 test
failed on weak visual-v1 without this projection; that negative result remains
relevant and is not erased. No reward values/categories, pixels, physical
dynamics or buttons change. Complete the already queued repeated-practice
studies first. Derive a hash-checked matched baseline from the completed
projected trials and their explicitly shared original frozen controls; this
is evidence composition, never an additional trial. Do not run or promote the
new dose candidate if its projected parent fails the independent cue gate.

21:00 UTC: causal-credit401 completed210 tiles/no starter versus the matched
default-wide287/starter3122 and original frozen268/no starter. This is negative
for that seed, not grounds to stop/omit its preselected402 comparison (running).
Once the projected-wide shuffled controls and independent3001..3004 retention
finish, resume BOTH interrupted default-wide practice series exactly from their
step4500 checkpoints. Finish the already registered1401/1402 and1403/1404
attempts and frozen1601/1602 evaluations, without adding episodes or selecting
a successful training source. Shared original controls are NOT independent
replications. This tests whether repeated real-game experience helps, not only
whether a one-episode trace changes. Recovery must preserve the old files,
pin/validate weights/config/game checkpoint provenance, verify overlapping
recorded decisions, and count each resumed attempt only once. Keep at most two
GPU jobs and do not promote an experimental candidate based on these partials.

20:48 UTC: projected-wide DOES reduce retained general B bias in the registered
neutral panel: B selections32.03%/29.17% versus default-wide52.34%/36.20%,
original28.65%; mean B0.939/1.009Hz versus1.752/1.090Hz, original0.928Hz.
All arms frozen, same3201..3203, all7 groups recorded; original repeated
control is shared, not independent. Corrected an earlier transcription of
original B to110/384=28.65% (not28.13%). Artifact
`game-motor-drift-probe-20260918T204118Z-6cdcbb`. Useful game learning is NOT
yet established: paired exploration is mixed and battle/Route1 outcomes absent.
The causal-credit implementation passes345 Python/26 JS tests and lint,
plus real-ROM learn/exact-resume/frozen/no-reward smoke
`internal-smoke-20260918T204550Z-141776`. It stores neural eligibility before
the held action, leaves all subsequent neural integration intact, and restores
pending credit in explicit in-flight snapshots. Old profiles do not latch and
old checkpoints keep feedback-boundary-v1. Candidate401 starts after that smoke
job exits; projected shuffled-control501 occupies the only other GPU slot.

20:44 UTC preregistration: projected-wide gameplay6000 completed with mixed
coverage185/290 versus default-wide287/255; starter1/2 in both, no wins/Route1.
Both resumed prefixes reproduce20 overlapping decisions exactly. CPU audit
`credit-update-audit-20260918T204115Z-a0e2b4` finds projected mean-input residual
at numerical-rounding scale, no active edge bounds at these final checkpoints;
do not blame clipping for the observed game limitation. Neutral audit and
shuffled visual-control completion are running, within the two-GPU limit.
Next ONE-FACTOR candidate: `visual-causal-credit-v5`, identical to showcase
wide-v3 except holding its internal factor-eligibility at the action boundary.
In streamed mode the existing reward uses a trace after12 NEW neural steps,
although those steps cannot change the already-delivered button. Holding the
earlier trace removes that unrelated innovation; it does not change physical
neural evolution, reward timing/value, visual input or decoder. This is a
causal-alignment/variance hypothesis, NOT a proven learning fix. Preselect
matched actual-game401/402 at6000 from original weights, exact frozen-prefix
reuse, then fresh frozen1601/1602 retention if warranted. Check checkpoint
compatibility, no-reward/frozen behavior and explicit in-flight trace restore.
Do not change the default while testing or use synthetic weights in the game.

2026-09-18 20:29 UTC: user explicitly resumes improvement experiments after
reporting a better showcase. Process audit found no active Python/showcase job.
Keep at most TWO GPU jobs, fixed budgets/checkpoints, and a named session ledger.
First finish BOTH preregistered projected-wide actual-game candidates401/402
from their exact step1000 checkpoints to total6000, preserving old partial runs
and checking overlapping recorded decisions bit-for-bit. Do not treat resumed
segments or reused frozen controls as independent trials. Compare all six game
arms (original frozen, wide-v3 learning, projected learning) and repeat the
registered neutral motor-drift panel3201..3203 on BOTH final candidate brains.
In parallel with useful CPU/code work, complete the interrupted projected-wide
ROM-free controls using exact saved full-state continuation, then independent
retention3001..3004; keep the original gate unchanged. Synthetic assay weights
never enter Pokemon. If the tonic-bias constraint fails, inspect its actual
update residuals and causal credit timing before proposing a further version.
Useful retained gameplay must still be tested frozen on fresh seeds; changing
weights, increased tile counts, or one successful path is not the stopping rule.
The showcase default remains unchanged while these candidates are evaluated.

2026-09-18 17:59 UTC: the user clarified that process cleanup was not a
stop-work instruction, then explicitly requested the newest best-supported
model as the showcase default. Fresh `scripts/start.ps1` launches now select
`visual-release-wide-v3`; the unfinished projected-credit variant remains
opt-in. Old profiles/checkpoints and all model/reward parameters are unchanged.
Only bounded implementation checks were launched, with at most two GPU jobs;
all finished and no Pokefly Python worker remains. Future experiment work must
retain that concurrency limit instead of relaunching the old queue en masse.
Verification: 332 Python/26 JS tests and lint pass; actual default launch80,
live stream50, and learn/resume/frozen/no-reward smoke pass. All10 protected
hashes match. Browser connection remains unavailable, so rendered-layout QA
is not claimed. See FOLLOWTHROUGH_RESULTS.md for exact verification artifacts.

Runtime status, 2026-09-18 17:38 UTC: all experiment jobs are STOPPED following
the user's process-cleanup request. This supersedes older "live"/"queued" notes
below; do not automatically relaunch that queue. Eleven experiment sessions,
their 22 Python processes, and associated launcher/runner processes were stopped
and their absence verified. No dashboard listeners remain on ports 8777-8785.
Unrelated user terminals, other projects, and Codex infrastructure were left alone.
Saved artifacts were not deleted. Interrupted trials are incomplete, not failed
or completed experimental results; there was no confirmed final graceful snapshot.
The two practice runs retain their step-4500 checkpoints; the two projected-rule
game runs retain step-1000 checkpoints. Frozen evaluation and visual-curve work
retain only what was already written. Before any future experiment work, audit
processes and set an explicit concurrency budget rather than accumulating jobs.

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

- 17:30 UTC: neutral motor-drift panel completed; ALL saved weights remain
  unchanged. Original/source401/source402 mean B rates0.928/1.752/1.090Hz;
  B selected28.65/52.34/36.20% across384 scored decisions each. Full artifact
  `game-motor-drift-probe-20260918T172152Z-e9332d` includes all7 populations
  and all3 seeds. Both exploratory projected-wide actual-game comparisons
  now launch from ORIGINAL synapses at8784/8785, not diagnostic weights.
  Weak-v1 projected rule's full independent gate fails both seeds, confirmed
  by raw/provenance audit `visual-retention-gate-20260918T172144Z-93513c`.
  Old v1 final32k source501 independent paired65.43% still fails its right
  cue50.67%; final shuffled and source601 tests remain required. No promotion.

- 17:28 UTC: the first frozen neutral probe shows game-trained401 B firing
 1.497Hz versus original0.732Hz on the SAME fresh3201 noise and gray image.
  Remaining seeds/sources are pending; this is a concrete tonic-drift lead,
  not a complete panel. Preselect exploratory ACTUAL-game tests of the already
  registered `visual-wide-projected-v4`, which changes ONLY the internal
  mean-input-preserving rule relative to wide-v3. SAME401/402 seeds,6k each,
  original synapses, same intro/rewards/pixels/buttons; no assay weights.
  Explicitly reuse the completed wide-v3 frozen controls ONLY after exact
 64-decision pixel/neural/action/start-state verification. Both candidates
  are required regardless of which source currently looks better. This does
  NOT promote the model before its queued3001..3004 final retention controls.
  Generalized the one-factor harness with explicit --factor rule; default
  learning-rate-only behavior and original report fields remain compatible.
  Queue in free slots after current diagnostic panels, not by stopping trials.

- 17:20 UTC: read-only exact replay of all four wide-v3 game arms validates
  every state/reward and measures Start/nested-menu occupancy. Frozen/learn
  fractions36.09/25.55% (401),37.86/34.93% (402). This is not semantic menu
  competence: increased B activity can also dismiss menus. Replay artifacts
  `recorded-menu-audit-20260918T171511Z-583024`, `...-1ede9a`, `...-1dfef6`,
  `...-1e3050`. All are replays, never new autonomous wins.
  Preselect a frozen neutral-screen motor-drift diagnostic: ORIGINAL once,
  then BOTH actual-game final6k synaptic states, shared RGB128,128 warmup+
 128 scored decisions, fresh noise3201/3202/3203. No reward, refitting,
  ablation, action intervention or neural export. Measure all seven motor
  rates/buttons, not a B-specific fitting target. This tests whether the
  repeated-game B increase is tonic or depends on encountered screens.
  Source401's first fresh6k practice ends271tiles/no starter; its second
  registered attempt1402 is now running. No successful episode is selected.

- 17:13 UTC: BOTH wide-v3 actual-game pairs complete6,000 decisions per arm.
  Frozen/learn tiles268/287 (401),188/255 (402); starters0/1 and0/0;
  wins0 throughout, no Route1 visit. Up fractions21.55/22.05% and17.17/23.87%.
  This is consistent short-budget exploration improvement, not retained
  gameplay competence. Both preselected saved-brain practice series now run:
  source401 at8782, source402 at8783, separate fresh attempts then frozen1601/
 1602 comparisons. No synthetic weights or reward changes.
  Completed v1 source402 retention is negative/mixed: original/retained tiles
 199/159 and266/185, starters1/2 in BOTH conditions (on different test seeds),
  no wins. Do not promote that model from its favorable first test alone.
  Source401's required panel continues; its repeated original1501 control
  reproduces all6,000 trajectories exactly and counts as ONE shared control.
  Partial final32k retention for old v1 source501 now runs on the registered
 2601..2604 seeds; original/paired only until its final shuffled state exists.
  Projected-v4 on weak visual-v1 fails the paired independent gate already:
 56.61%/58.28%, one cue42.93%/50.25%. Finish the shuffled panels; do not promote.

- 16:59 UTC: wide-v3 PASSES the preselected independent acquisition gate for
  BOTH training seeds. Paired balanced69.10%/68.80%, original44.72% shared
  control, shuffled36.37%/36.82%. Each paired cue exceeds55%:
 501 left56.43/right81.77;601 left59.34/right78.26. Same final8,192 synapses,
  neutral128 warmup, held-out2801..2804,128 decisions/cue/seed, no rewards or
  test learning. Each paired panel beats BOTH controls by at least5 points.
  This establishes limited retained two-cue association, NOT a Pokemon policy,
  physiological validation, or statistical significance. Original repeated
  controls are counted once, not as independent replications.
  New read-only gate audit recomputes scores from ALL raw choices, verifies
  final weight hashes, protocol and distinct source seeds, and rejects missing/
  duplicated controls. Artifact `visual-retention-gate-20260918T165919Z-e1b7c7`.
  It refuses legacy reports lacking explicit completed status rather than
  silently treating them as complete. Five protocol unit tests pass.
  The registered same-start reversal now starts for501/601 (absolute checkpoints
 8704/10240/16384;8,192 additional training), then fresh3101..3104 retention.
  Assay-trained weights remain forbidden in gameplay; no default model promotion.
  Reversal artifacts `visual-learning-curve-20260918T170017Z-620429` (501)
  and `...-485f3d` (601). Stronger-signal/projected-credit acquisition curves
  are `...T165047Z-6f9d9b` (501) and `...T165301Z-41ca7d` (601); their original
 3001..3004 final retention remains queued. Actual-game practice source401:
  `retained-game-series-20260918T164934Z-3b2ae4`. Full suite328 Python/26 JS
  plus lint passes (`verify-294136a72b69487a87c68540cf2cafea`).

- 16:50 UTC: both v1 long-game pairs complete. Frozen/learn tiles are352/351
  (401) and348/297 (402). Both learned brains get starters sooner, but401
  learning never reaches Route1 or wins; its frozen control wins ONE rival
  encounter.402 neither wins, and frozen explores farther north on Route1
  (y25 versus28). No overall learned-progression claim or v1 promotion.
  Corrected an earlier report's double-count: battle_win1 plus rival_win1 are
  the SAME encounter, not a rival AND wild win. Source401 final24k retained-
  weight panel now runs on the registered1501/1502 seeds, just like402.
  Artifact `retained-gameplay-20260918T164449Z-192a55`.
  Wide-v3 seed401 matched6k pair: learning287tiles/starter3122 versus frozen
 268tiles/no starter; neither wins or visits Route1. Both402 arms continue.
  Registered actual-game practice source401 begins with1401 at8782; final
  source402 will use the same protocol when ready. No best-source selection.
  Its finished comparison slot starts the queued wide+projected501 curve.

- 16:49 UTC: skip only discarded original current rows during calibrated
  visual CUDA steps. No edges/weights removed: visual rows are replaced by
  the frozen visual circuit and isolated sensory currents were already zero.
  Full public current measurements and calibration/diagnostic overrides remain
  unchanged; old visual models take the original path. Two ABBA benchmarks
  match ALL neural arrays, metadata, actions and counts over64-decision
  branches; time reduction about7.6..14% under shared GPU load, not100x.
  Artifacts `internal-propagation-benchmark-20260918T164408Z-657c5e` and
  `...T164726Z-ba9aae`. Both v1 and wide-v3 real-ROM checkpoint continuations
  match the ORIGINAL recorded samples5001..5020 exactly, plus split resume
  arrays/rewards. Artifacts `checkpoint-window-verification-20260918T164752Z-ef7a24`
  and `...T164837Z-c489f3`.323 Python/26 JS plus lint pass
  (`verify-2fbcdadd28be44ae9adfa5d42798323e`). New processes get the speedup;
  existing fixed-budget experiments are not restarted or relabeled.

- 16:40 UTC: preselect the follow-up to wide-v3 before its second independent
  acquisition test or full gameplay pairs finish. Once BOTH acquisition
  sources have passed original/shuffled/both-cue controls, test same-acquired-
  state reversal (both arms start from the paired8,192 state), additional
 512/2,048/8,192 decisions, independent retention seeds3101..3104. Same gate,
  no altered learning rate, calibration, rewards or decoder.
  Separately queue repeated ACTUAL-game practice from BOTH final6,000 learning
  runs (401 and402), never a selected best checkpoint or synthetic weights.
  Each gets two explicitly reset fresh6,000-decision attempts carrying only
  synapses: source401 training1401/1402; source402 training1403/1404. Evaluate
  final synapses frozen versus original in fresh6,000-decision games on1601/
 1602, alternating arm order. Report starters, wins, new-area/Route1 progress,
  tiles and button bias separately; mixed outcomes are not a blanket pass.
  These resets are training interventions, NOT uninterrupted game completion.
  Use free GPU slots after current registered tests, not extra user compute.
  Browser-skill retry remains unavailable; no rendered-layout claim.

- 16:36 UTC: wide-v3 acquisition seed501 passes the original/paired PART of
  the independent retention screen: original44.72%, paired69.10%, with paired
  left56.43% and right81.77%. These are saved final8,192 synapses, fresh neutral
  warmup, noise seeds2801..2804, and no test reward or weight changes.
  Artifact `visual-retention-probe-20260918T162138Z-e22d73`. Shuffled final
  weights and the independent seed601 panel are still required; no full gate
  pass or gameplay-learning claim yet. Seed601 original/paired panel starts
  in the finished panel's GPU slot (`...T163645Z-0eab8b`). Both unchanged
  paired/shuffled acquisition curves and registered actual-game pairs continue.
  Current code/docs through the verified default transport are public at
  commit `fe667e3`; no ROM, research weights or run artifacts were published.

- 16:21 UTC: promote ONLY the already verified command-transport fix to fresh
  launches, NOT any experimental visual/learning candidate. New versioned
  `sensorimotor-bounded-serial-v2` keeps the default's neural model, rewards,
  decoder and24-frame budget exactly; paired commands use11 direction,1release,
 11 function,1release. Evidence: input-priority source audit, four forced
  nickname-screen transport controls and actual serial-play comparisons already
  documented above. Original `sensorimotor-bounded-v1` file stays unchanged;
  every saved run restores its original settings. Unit comparison enforces
  the one-factor change. Fresh default real-ROM/launcher/resume checks follow.
  This fixes delivery of fly-chosen buttons; it is not learned game behavior.

- 16:17 UTC: wide-v3 paired8,192 interim scores66.09%/72.39%, but seed501's
  left cue is only49.09% (right83.10%);601 left57.45%,right87.32%. The stronger
  signal improves capacity, yet generic button drift remains a possible limit.
  Independent2801..2804 retention remains mandatory; no gate pass yet.
  Preselect `visual-wide-projected-v4`: ONLY swap the internal credit rule in
  the wide-v3 model to the already implemented local mean-input projection.
  Same frozen calibration, anatomical synapses, noise, rewards and adapter.
  This isolates credit on the stronger-signal circuit; it is not fitting a
  decoder or combining two untracked parameter changes. Queue501/601 curves
  at512/2048/8192 after existing registered jobs free capacity, then independent
 3001..3004 final retention with the unchanged gate. No new trial launched yet.
  Neither projected profile is promoted based on this hypothesis.

- 16:12 UTC: independent reversal paired panels return56.30%/56.48%, versus
  the acquired baselines41.35%/41.91% under the reversed contingency. This is
  retained change in the appropriate direction, but left-cue accuracies45.68%
  and38.41% mean BOTH still fail the same60%/each-cue55% criterion. Shuffled
  independent panels are still finishing; keep the result provisional until
  that comparison is complete. Do not reinterpret a favorable time bin as a pass.
  The unchanged v1 acquisition501 final32,768 interim score is59.18%, not an
  improvement over its8,192 small probe. Finish registered controls; do not
  promote simple extra runtime as the fix. Wide-v3 literal first1,500 sample
  Up fraction25.33%, house exit464, versus original v1 same-seed Up27.47% and
  no house exit by1,500 (later1566). Early trajectory effect, not learning proof.

- 16:00 UTC: stronger-release capacity control completed: original43.05%,
  temporary oracle forward72.71%,reverse88.88% balanced on2,001..2,004 noise
  seeds, with both cues above55%. Original files unchanged, fitted weights
  discarded and never exported. This is improved controllability, NOT reward
  learning. Artifact `oracle-synaptic-capacity-20260918T154733Z-45c072`.
  Its completed GPU slot now starts the registered actual-game wide-v3 seed401
  pair (learning first, then frozen),6,000 each,port8780. Seed402 will run the
  opposite arm order as capacity becomes available. Only original neural weights
  initialize these games; no capacity/conditioning checkpoint is accepted.

- 15:58 UTC: both same-acquired-start reversal curves completed all arms.
  Final small interim probes: paired60.20%/59.86% versus shuffled30.49%/49.00%
  (501/601). Independent neutral-warmed retention on2701..2704 now running,
  with the actual acquired8,192 checkpoint explicitly used as baseline in
  BOTH panels. Do not call these interim curves an independent gate pass.
  Sources `visual-learning-curve-20260918T150238Z-ab054c` / `...-718867`.
  Projected-credit real-ROM smoke passes and its frozen70-decision trajectory
  plus ALL arrays reproduce the original v1 control exactly. Full suite now
 311 Python/26 JS (`verify-c449852e570547aa89ec004e7a7f3245`). Source/docs pushed
  as `52c3499`; subsequent changes remain recorded here until the next push.

- 15:54 UTC: found a remaining HUMAN-display issue: the reward metric still
  showed only the current sample, usually zero. Add an authoritative cumulative
  reward field ONLY to dashboard publication (not neural input or archived
  trajectory schema). UI labels TOTAL REWARD, with per-sample/delivered values
  in its tooltip; an older already-running server truthfully labels STEP REWARD.
  No client-side accumulation that loses events on reconnect/coalesced frames.
  Live pixel/brain/button/pacing smoke verifies totals against logged prefixes:
  `internal-learn-20260918T155247Z-05a8eb`.26 JS checks pass; browser skill
  connection retried at15:53 and remains unavailable, so no rendered QA claim.

- 15:49 UTC: release-wide-v3 generic sensory panel is complete. Actual
  untrained balanced choices remain45.0%static/46.67%motion, but both cues are
  present at descending/motor inputs (small offline diagnostic, not a policy).
  Static motor contrast/std:Left-0.50,Right+0.20 versus ~0.001/0.015 in v1.
  Sign is opposite the assay's rewarded mapping, not a defect to hand-correct.
  Current audit `motor-signal-audit-20260918T154625Z-435fa9`; latency panels
  `visual-latency-probe-20260918T153147Z-722cae` / `...T153158Z-8c4c73`.
  Apply the existing unchanged offline synaptic capacity assay to this model;
  temporary supervised weights are NEVER exported or used in Pokemon.
  Preselect literal-game wide-v3 comparisons:401/402,6,000 per frozen/learn arm,
  fresh original neural weights, identical intro, alternating arm order,
  unchanged rewards/pixels/buttons. Queue after current jobs free GPU capacity.
  This is an exploratory model test while conditioning controls finish,
  NOT a promotion based on the encouraging interim2,048 scores61.1%/66.4%.
  Independent final-weight conditioning retention remains2801..2804.

- 15:40 UTC: preselect a distinct credit-assignment test on the UNCHANGED
  visual-rate-v1 circuit/calibration. The direct-current audit found correct
  learned cue contrast but also a generic upward current shift; increasing
  learning rate made game button bias worse. Test a local mean-input-preserving
  projection of the existing centered perturbation eligibility, using only
  original synaptic magnitudes and each presynaptic cell's existing running
  release mean. No action/cue labels, desired firing targets, new synapses or
  reward changes. All anatomical motor/descending targets get the same rule.
  This is a versioned engineering approximation, not a biological mechanism
  or guaranteed exact gradient. Bounds can make the projection approximate.
  Same501/601 paired/shuffled budgets512/2048/8192; independent final retention
  seeds2901..2904, unchanged gate. Existing registered curves finish unchanged.
  Do not use any ROM-free trained weight in gameplay.

- 15:37 UTC: preselect actual-game FINAL-24,000 retained-weight comparisons
  for BOTH401/402 trained brains, frozen original versus frozen retained,
  fresh intro resets, held-out seeds1501/1502,6,000 decisions per arm/seed.
  Start402 first because it completed first, not because it played better:
  297tiles,starter,Route1miny28,no wins.401 follows when its unchanged budget
  completes. Report both seeds/arms regardless of outcome; duplicated original
  controls are reproducibility checks, not additional independent controls.
  Only ACTUAL-game checkpoints, pinned to final completion, enter these tests.
  No synthetic weights, reward changes or hand-selected progression start.
  These short tests can show retained starting behavior, not long-horizon
  progression or eventual completion. Keep starter/wins/exploration and button
  availability separate; do not equate altered actions with improvement.

- 15:32 UTC: wider UNIFORM neutral calibration passes independent reset checks:
  mapped populations 0.75..1.20 Hz across seeds301..303; escape/B 0.90..1.16
  instead of 9 Hz. One escape cell needs bias -0.452, explaining the old bound
  failure. Artifact `intrinsic-probe-20260918T152313Z-f5a2bf`. Freeze the NPZ
  before tests; no game data or named motor target enters fitting. Preselect
  same frozen static/motion latency panel seeds801..804 for release-wide-v3,
  then the SAME paired/shuffled 501/601 learning curves (512/2048/8192) and
  independent final-weight retention seeds2801..2804 with the unchanged gate.
  Actual-ROM exact-resume/frozen/no-reward smoke is a software check only.
  No new visual model/default promotion on neutral calibration alone.

- 15:21 UTC: stronger learning-rate candidate FAILS both complete 6,000-game
  comparisons: 175/111 tiles, no starter, versus v1 learning 213/233 (one starter)
  and frozen 188/193. Seed401 B rises to 42.9% and lower-town dwell 25.2%.
  Reject promotion; do not extend only a favorable arm. Fallback-unit scale0.05
  also does NOT solve visual saturation: LC4 gray rises to 2.55/4.78 versus
  0.093/3.94 at scale1; motion sensitivity is mixed. Keep production scale1.
  Artifacts `calibrated-visual-probe-20260918T151830Z-5ad577` / `...-7dcd60`.
  The earlier stronger-release calibration hit its artificial bias lower bound
  (-0.14) at the overactive cell. Test a SEPARATELY VERSIONED wider uniform
  intrinsic calibration range [-0.5,0.5], same 1 Hz neutral-gray equation,
  10,000 steps/seed707 and independent reset seeds301..303. EVERY eligible
  non-sensory spiking neuron uses the same equation/bounds; no motor labels,
  game images, rewards or action quotas enter fitting. Freeze before any tests.
  Keep release1 and all other visual-v2 settings unchanged; this isolates the
  calibration-range constraint. No game promotion unless neutral reset probes
  are stable, followed by sensory/learning and actual-game comparisons.

- 15:18 UTC: test a specific VISUAL transfer-unit hypothesis on generic stimuli
  only. Reference-fitted edges use raw reference rates; original fallback edges
  currently also use those rates with gain 3. However, original visual outgoing
  edges elsewhere see r * 0.25 / 5. Thus fallback edges have a 20x higher input
  unit scale than the SAME source's original cross-boundary edges. Preselect
  fallback scale 0.05 (0.25/5), with no fitted weight/sign deletion, no changes
  to reference-covered/photo-lamina conductances, retina, biases or time constants.
  This is an engineering unit-consistency hypothesis, not proven physiology.
  Compare standalone scale1 and scale0.05 generic gray/static/eight-direction
  responses on identical current 71,080-cell graphs. No game/motor labels or
  rewards, no production profile change yet. Existing registered runs finish
  unchanged, regardless of this diagnostic result.

- 15:03 UTC: direct current audit of the acquired 8,192 weights finds the
  correct cue contrast at BOTH DNa02 steering cells in BOTH training seeds:
  contrast / sample std ~0.45..0.57 versus ~0.001..0.015 original; shuffled
  ~0.05..0.07. This reuses original input samples, so is first-order evidence,
  not a full recurrent prediction. Global current drift also remains. Artifact:
  `retained-current-audit-20260918T150116Z-225a19`.
  Explicit protocol extension BEFORE further results: start an EXPLORATORY
  same-acquired-state reversal from the 8,192 paired checkpoints even though
  the acquisition promotion gate failed. This tests contingency-following,
  not a retroactive gate pass; the longer acquisition comparison also continues.
  Both training seeds, checkpoints 8,704/10,240/16,384 (512/2,048/8,192 added);
  paired and within-cue shuffled reversal start from identical acquired full
  neural states. Final independent retention: 2701..2704, with the original
  acquired state as frozen pre-reversal baseline. Report all arms/seeds.

- 14:57 UTC: first independent shuffled-retention panels complete at 48.81% /
  49.83%, versus paired 58.49% / 59.56%. There is a modest retained cue contrast,
  but the original both-cues gate still FAILS. Continue the preselected unchanged
  training-dose test; no promotion or softened criterion.
  A distinct GAME-learning-dose mismatch is now quantified: sum(tanh(reward))
  is 1,488.92 in the 8,192-decision seed501 assay but only 14.56 in the completed
  6,000-decision seed402 game run. This is not itself an expected gradient, but
  shows why an assay learning rate may have little effect with sparse rewards.
  Preselect ONE 10x internal learning-rate candidate, 0.2 vs 0.02. Everything
  else, including raw reward amounts/timing, visual calibration, existing-edge
  bounds/budgets, retina and buttons, stays fixed. Run both original 401/402
  starts for 6,000 decisions; validate exact 64-decision frozen prefixes before
  explicitly reusing the existing full frozen and 0.02 controls. No new oracle
  or assay weights. Treat as exploratory, require frozen transfer on new seeds
  before attributing any improved game progress to retained learning.
- CUDA visual update dispatch has a bit-exact fused alternative. Random-array
  tests and complete 64-decision learning branches (all neural state, choices
  and counts) match exactly. Two shared-GPU timings show only ~2..7% end-to-end
  benefit, not a large speed claim. Real-ROM learn/resume/frozen/no-reward smoke
  passes after integration (`internal-smoke-20260918T145416Z-f513e0`). Existing
  running jobs retain their loaded implementation; neither computes different
  neural states. No model parameter, timestep or checkpoint identity changes.

- 14:43 UTC: independent gray-warmed FINAL-8192 retention is 58.49% / 59.56%
  balanced (training seeds 501/601), versus 49.16% original. Right-cue accuracy
  remains 44.76% / 47.39%, so BOTH fail the preselected promotion gate despite
  measurable cue contrast. Shuffled checkpoints are still finishing. Complete
  their identical 2501..2504 test panel; do not relabel this partial result a pass.
  Preselect unchanged acquisition continuation at 16,384 and 32,768 decisions
  for BOTH seeds/arms, then independent final-weight retention on 2601..2604.
  This is a training-dose hypothesis after an improving curve, not selection
  of a favorable checkpoint or a relaxed criterion. If it passes, reverse
  contingencies from the SAME acquired state in paired and shuffled arms.
  No synthetic weights enter Pokemon.
- Both actual-game 6,000-decision pairs completed. Learning/frozen tiles:
  seed401 213/188; seed402 233/193. All four leave the house and use Up ~25%.
  Only seed402 learning obtains a starter and reaches Route 1; no arm wins a
  battle. This is not retained-learning proof. Preselect exact continuations
  of ALL FOUR runs to 24,000 total decisions (18,000 additional), same seeds,
  unchanged model/rewards and no game reset. Preserve both less-successful
  seed401 runs, not only the promising trajectory. Evaluate final game-trained
  weights separately on held-out game/noise resets before any promotion.
- Strong-release v2 FAILED its neutral-only calibration: B population ~9.2 Hz
  versus the uniform 1 Hz target; three reset probes reproduce 9.0..9.3 Hz.
  No game tests or learning promotion. Preserve the diagnostic profile/artifact.
  The new visual-model SUPERVISED capacity assay reaches 61.38% forward and
  65.65% reversed balanced accuracy, versus 50.01% original. All temporary
  oracle weights discarded; this is capacity evidence, NOT reward learning.

- 14:28 UTC: BOTH visual-rate paired curves reach ~65.5% balanced cue accuracy
  at the preselected FINAL 8,192 checkpoint. Shuffled controls are still running;
  this is preliminary, not a pass. Start independent reward-free retention of
  original and paired weights on the already preselected seeds 2501..2504,
  128 neutral-gray warmup decisions and 128 decisions/cue, all time bins
  reported. Explicitly label this as a partial control panel; test the completed
  shuffled checkpoints on the SAME seeds once available. Source assay weights
  remain outside Pokemon. Require each-cue performance, control advantage and
  same-acquired-state reversal, not just this encouraging preliminary mean.

- 14:25 UTC: test the explicitly uncertain cross-model signal units, NOT any
  Up/steering gain. The initial transfer maps the visual rate ceiling (5) to
  maximum transmitter release 0.25, inherited from the old lamina prototype.
  Preselect the alternative existing graded-release setting 1.0 for EVERY
  calibrated visual cell (fourfold boundary signal, same internal parameters,
  anatomy, eye map and fixed output decoder). This maps the maximum to one
  spiking neuron's 20 ms release unit. It is an engineering sensitivity test,
  not measured physiological calibration. Refit ONLY uniform neutral-gray
  spiking offsets by the same 1 Hz / 10,000-step protocol, then freeze. Check
  reset activity and generic input contrast before any game promotion. No
  action labels, reward or game images choose the calibration. Existing v1
  comparisons finish unchanged; never modify their fixed profile mid-run.

- 14:17 UTC: visual-rate current audit still finds tiny static-cue contrast
  at the original steering rows (contrast / current std 0.0009 and 0.015).
  Distributed cue information does not imply useful output wiring. While all
  four registered visual/game comparisons complete, repeat the existing
  supervised capacity diagnostic on the NEW frozen visual circuit: fit original
  positive steering input synapses, 0.25..4 factors, +/-25% budgets and fixed
  two-cue mean current, using the recorded 801..804 activity; test 2001..2004.
  Count actual graded release now present in the measurements. Discard every
  fitted synapse afterward. This is NOT online learning, never exported and
  never used in Pokemon. Report both forward and reversed mappings, not just
  the favorable one. This asks whether better sensing exposed learnable input
  variation, not whether an external policy can solve the game.

- 14:14 UTC: visual-rate real-game learn/resume/frozen/no-reward smoke passes
  after diagnosing PyBoy 2.7's unsaved renderer window-line counter. A discarded
  warm render then exact state reload removes 3,470 differing first-frame pixels;
  serialized emulator state remains identical. No extra frames/rewards reach
  the fly. Old-checkpoint branch test and actual HTTP pixel/graded/action/speed
  checks also pass. Browser connection retried; still unavailable.
  Start preselected actual-application comparisons from original synapses:
  seeds 401 and 402, 6,000 decisions, new visual model frozen versus learning,
  counterbalanced arm order. Only intro setup scripted, all game actions fly
  controlled; frozen calibration and original broad rewards. These exploratory
  runs continue alongside both ROM-free curves. Require held-out frozen-weight
  transfer before treating any learning-arm advantage as retained improvement.

- 14:09 UTC: full-circuit visual-rate diagnostics now distinguish opposite
  motion at descending cells and motor inputs on all eight held-out constant
  samples and all tested switching windows (old motor-input measurement 50%).
  This is offline information measurement, NOT an action controller or learning.
  Actual frozen Left/Right discrimination remains 49.4% balanced for motion,
  43.7% for static half-field images. Preselect unchanged paired/shuffled static
  learning curves, seeds 501/601, checkpoints 512/2048/8192; retain the same
  >=60% / each-cue >=55% / >5-point control / reversal gates. Synthetic weights
  stay outside Pokemon. Perturbation eligibility now accepts actual graded
  presynaptic release for this profile only, preserving causal prior history.
  Unit tests cover that formerly missing signal. No learning claim yet.
  Real-ROM smoke found a distinct emulator resume issue: first sampled image
  after reload differs, despite exact neural-only restore. Diagnose and fix
  before counting full-application checkpoint tests as passed.

- 14:00 UTC: transferred fitted visual rate model is stable on generic gratings,
  but direction sensitivity is weak/uneven (T4/T5 group-vector selectivity
  0.025..0.242; 3.1..4.1% of all visual cells at rate ceiling). This is NOT
  validated physiology. Preserve all 3,018,819 original selected-cell edges,
  original signs, 6,006 actual photoreceptors and unchanged retinal mapping.
  Prototype selects 71,079 real visual cells, no virtual neuron or policy.
  Next: opt-in full-circuit coupling, retain all external connections, 4 ms
  internal visual substeps, fixed [0,5] -> [0,1] graded-state conversion and
  unchanged maximum release. Freeze visual parameters before gameplay. Uniform
  neutral-gray excitability calibration must be redone for remaining spiking
  cells; no game/button target enters it. Check exact checkpoint continuation,
  original-profile compatibility, raw visual sensitivity and live application.
  Do not mistake successful integration for learned gameplay.
  Integration also preserves one original L4 omitted by the reference subset:
  its type constants are shared with other L4 cells; its connections retain
  original magnitudes/signs. Full model therefore has 71,080 calibrated cells.

- Completed projected-v5 curves: paired balanced accuracy 52.4% / 56.8%,
  shuffled 47.7% / 49.4% at final 8,192 decisions (training seeds 501/601).
  Neither meets the predefined >=60% cue-learning screen. Not promoted.
  The wider-bound SUPERVISED temporary capacity diagnostic reaches 64.1%
  forward / 82.4% reversed balanced accuracy; no fit is exported or used in
  Pokemon. This diagnoses limited capacity, not reward learning.

- User approved separately calibrating the fly's INTERNAL visual-neuron model
  against generic visual stimuli/published responses, provided the fly remains
  the game player. Scope: preserve raw screen input, anatomical circuit and
  fixed button mapping; freeze visual calibration before game reward learning.
  No game states, action targets, routes, external vision policy or learned
  output adapter in calibration. First inspect the published flyvis model and
  the MaleCNS parameter-transfer implementation; verify anatomy, licensing,
  neural dynamics, calibration objective and visual responses before adopting
  any parameters. Do not assume an attractive demo proves end-to-end control:
  the separate closed-loop-fly project's own follow-up reports a readout
  artefact. Existing projected curves and the bounds diagnostic continue and
  will be fully reported, including negatives.

- 13:39 UTC: original frozen motion-grating test is also weak at the controller:
  47.0% same-direction choices conditional on Left/Right; motor-input offline
  direction classification 50%, descending 62.5% (small correlated diagnostic
  panel, not a statistical claim). No learned visual policy is introduced.
  Asked the user whether separately calibrated internal visual-neuron dynamics
  would fit the experiment; current learning comparisons continue meanwhile.
  Independently test whether the arbitrary synaptic factor bounds constrain
  capacity: rerun the existing temporary oracle positive control with positive
  factors 0.1..10 instead of 0.25..4, SAME +/-25% input budgets and mean-current
  equality, same fit seeds and independent test panel. Repeat the original arm
  exactly against the old report; count it as verification, not a new trial.
  Fit is deliberately supervised diagnostic ONLY, all temporary weights are
  discarded, no learned adapter or exported gameplay model. A better oracle
  response would justify a separate reward-learning test, not prove learning.

- 13:35 UTC: while both fixed-budget projected curves complete, test a missing
  sensory control: equal-luminance horizontal moving gratings, updated at EVERY
  neural step through the unchanged retina. Static half-white cues are not
  equivalent to visual motion. This is ROM-free, frozen original perturb-v3,
  seeds 801..804, same constant/switching lengths as the existing latency probe.
  Gratings have 32-pixel period, 50% duty cycle, two pixels/neural-step speed;
  opposite directions share the same first frame and luminance. No fitted
  encoder/readout, weight changes, labels to neurons, or changed game input.
  Measure information at the existing pathways AND the actual fixed-decoder
  choices. This is a sensitivity test, not new learning/progression evidence.

- 13:30 UTC: projected-v5 has exact frozen forward counts/actions/rates for 64
  windows versus score-v2 (`projected-forward-verification-20260918T132730Z-875628`).
  Real-ROM learn/resume/frozen/no-reward smoke passes
  (`internal-smoke-20260918T132601Z-f78aed`); 286 Python / 23 JS tests and lint pass.
  Original stopped-run checkpoint suffix is still exact after the inactive gain
  field addition (`checkpoint-window-verification-20260918T132851Z-df5f99`).
  Both preselected projected visual curves are now running (seeds 501/601).
  Before first results, define the promotion gate: FINAL 8,192 weights, not
  the best interim peak; independent frozen seeds 2501..2504; balanced cue
  accuracy >=60%, each cue >=55%, and >=5-point balanced improvement over both
  original and shuffled controls on each training seed. Require reversal from
  that acquired state against a same-start shuffled reversal, not just a new
  global preference. Passing is cue-learning evidence only, not game progress.
  The fixed gain-12 neutral calibration is already poor: A 5.5..9.3 Hz, B
  15.2..15.8 Hz and memory-output mean 21.5..21.8 Hz on frozen reset probes,
  versus the 1 Hz target. No gain-12 gameplay promotion; finish the read-only
  sensory measurement to preserve the failed hypothesis.

- 13:23 UTC: resumed after the user reported an error state. All three pending
  jobs completed normally. Broader-scope visual training at 8,192 decisions
  gives 51.6% balanced cue accuracy versus 46.0% shuffled; it mostly increases
  Left regardless of cue. Actual Route-1 scope tests: motor-only 208/327
  positions, broader 249/430, reused frozen 337/414; neither reaches a new town.
  These are not promoted or treated as retained navigation learning.
  Next, test TWO separable mechanism hypotheses, not a combined tuned model:
  (1) a local mean-input-preserving projection of score-v2 updates, using only
  each presynaptic cell's running release mean and original synapse sizes.
  This tests whether generic motor bias consumes learning capacity. Keep the
  exact stochastic forward model, calibration, decoder and all other learning
  settings. Preselect paired/shuffled visual curves, seeds 501 and 601,
  checkpoints 512/2,048/8,192; report cue-balanced discrimination as well as
  total rewarded activity. (2) a fixed whole-circuit synaptic-gain sensitivity
  test at 12 versus the original 3, with a separately frozen neutral-gray
  calibration using the existing uniform 1 Hz homeostasis protocol. No motor
  labels, rewards, game images or button fit enter calibration. Test frozen
  cue contrast/noise before committing to any long training. Both hypotheses
  are engineering models, not established fly physiology. Synthetic weights
  NEVER enter Pokemon; retained learning, independent noise and reversal are
  required before any success claim. Preserve all negative results.

- 11:30 UTC: the credit-direction audit raises target frequency but not cue
  discrimination; inter-seed update cosine is only 0.027. Existing input-count
  data shows left/right steering cue contrast only 0.10/0.16 of the within-cue
  current standard deviation (score-v2). No gain/noise/critic change is assumed
  to solve this. While the full broader-scope visual assay finishes, preselect
  EXPLORATORY actual-game tests of motor-only versus one-hop perturb-v3 learning:
  same original weights, calibrated serial controller, recorded Route-1 reset,
  legacy outcome timing, seeds 1201/1202, 6,000 decisions each. A reset is an
  intervention; neither assay/oracle weights nor a learned adapter enter the
  game. Validate original frozen 64-decision prefixes against the existing
  serial controls, then explicitly reuse those full frozen controls. This is
  not promotion or a learning claim; report BOTH learning scopes/seeds even if
  the visual assay fails. Any improvement needs frozen transfer on new seeds.

- 11:10 UTC: test whether the score-v2 local reward-credit signal points toward
  better choices at all, before inventing another critic. ROM-free DIAGNOSTIC:
  freeze original weights while accumulating the unchanged rule's centered
  reward-times-eligibility signal over 4,096 decisions on each noise seed
  501/601, same alternating cues. Measure agreement between the two estimates.
  Then temporarily move existing weights along the averaged direction and its
  NEGATIVE control, max 25% factor change per postsynaptic population (same
  positive diagonal scaling for both signs). Evaluate actual fixed-decoder
  behavior on fresh seeds 2301..2304, 128 decisions/cue; compare original,
  positive and negative directions. No fitted classifier/readout, game use,
  exported weights or online policy change during collection. Report both
  target frequency and cue-specific discrimination. This batch direction
  audit is NOT a trained-game model or a biological learning claim. Restore
  all originals and verify protected hashes afterward.

- 11:01 UTC: compare the FINAL weights from both completed 24,000-decision
  serial games (legacy outcome timing and earlier confirmed outcomes), not a
  selected intermediate best checkpoint. Frozen Route-1 transfer, seeds
  1201/1202, 6,000 decisions, same game start and forward configuration as the
  validated serial controls. Reuse those original controls explicitly. This
  asks whether the observed 2-versus-5 wins left useful retained behavior.
  No practice resets in source training, synthetic weights, new shaping or
  further learning in evaluation. The reset for transfer is an intervention.
  Compare ALL arms/seeds and report unfinished-but-paid encounters separately.
  A favorable small panel requires new-seed confirmation before promotion.

- 10:55 UTC: the matched stochastic information audit also finds cue information
  at motor inputs; missing pixels are not the explanation. Both-sign oracle
  fitting gives only 56.9% forward (positive-only 55.1%) and 76.3% reverse;
  fixed inhibition alone is not established as the bottleneck. Test ONE broader
  anatomical learning scope: existing positive inputs to all motor/descending
  cells PLUS central-brain intrinsic cells that directly project to ANY
  descending cell. This selects 21,848 target cells / 4,300,611 positive edges,
  versus 2,129 original targets; no button labels, added edges, encoder changes,
  fitted readout, new reward or new neuron dynamics. Keep perturb-v3 unchanged
  otherwise. This is an engineering capacity hypothesis, not validated fly
  plasticity. Require exact frozen forward equivalence, sparse-index tests and
  checkpoint continuation before paired/shuffled visual acquisition seed 501
  at 512/2,048/8,192. Synthetic-trained weights NEVER enter Pokemon. Require
  retained cue-specific improvement, independent seed and reversal before any
  learning claim or model promotion. Preserve all old failures and both arms.

- 10:43 UTC: the successful frozen sensory-information audit used deterministic
  threshold dynamics, NOT the stochastic model required by score-v2/v3. Do not
  assume its visual information/capacity carries over. Run the same frozen
  latency/count audit for unchanged score-v2, seeds 801..804, before another
  score-learning variant or upstream-plasticity expansion. This is a missing
  matched-model diagnostic, not new training or an external visual policy.
  Its offline classifier remains measurement-only, never used by the game.

- 10:36 UTC: positive-edge oracle achieved 55.1% forward / 79.8% reversed
  conditional accuracy (original forward 40.7%) in actual frozen circuit tests.
  These are supervised temporary synapses, NOT learned behavior. Artifact
  `oracle-synaptic-capacity-20260918T102612Z-f7f970`; all weights discarded and
  original source hashes match. Next capacity control permits both ORIGINAL
  signs with separate E/I budgets and the same mean-current constraint, fit
  seeds and test seeds. Never export/deploy. Repeat original counts to check
  the expanded in-memory intervention code has not changed the baseline; do
  not count that duplicate as independent evidence. This asks whether fixed
  inhibition limits the harder cue assignment before changing another rule.

- 10:30 UTC: eight real-battle training episodes for outcome-v3 are complete;
  the matched battle evaluation is still running. Independently preselect a
  transfer test now, not only if its remaining battle scores are favorable:
  weights from `controlled-battle-learning-20260918T095632Z-8e9695/training-908`,
  fresh full-game intro starts, frozen seeds 1501/1502, 12,000 decisions each.
  Reuse and validate the corresponding original-weight full-game controls from
  the completed series, explicitly not new trials. Reward timing may differ
  but has no effect on a frozen controller; every forward-model setting must
  match. Preserve exact starting game-state hashes and flag any paid but
  unfinished final encounter. Live dashboard enabled. Training reset to the
  same recorded rival remains an explicit intervention; this is transfer of
  REAL GAME training, not diagnostic button/cue-trained weights or a route.

- 10:24 UTC: after multiple failed cue-learning rules, run a separate ROM-free
  SYNAPTIC CAPACITY POSITIVE CONTROL before adding another rule. Use the existing
  frozen presynaptic counts (seeds 801..804) and a supervised linear program to
  maximize direct left/right cue contrast on existing positive DNa02 inputs,
  within the same 0.25..4 factors and +/-25% input budget. Also preserve each
  neuron's mean predicted input across the two cues; fix unobserved/graded
  inputs at original weights. Then test those temporary weights in the ACTUAL
  recurrent fly on new noise seeds 2001..2004, both cue assignments, 128 choices
  per cue, original fixed decoder. No rewards, game, fitted readout or neural
  weights exported. This is deliberately an oracle intervention, NOT learning
  evidence and NEVER a deployable brain. It tests whether the earlier frozen-
  current capacity bound translates into actual choices. Failure is not a
  general impossibility proof. Preserve every arm and original data hashes.

- Next mechanism test, after the 10:17 UTC inspection: actual replay at frozen
  serial/3-second samples 3,406..3,413 shows repeated Up into an impassable
  north-facing ledge. Artifact `recorded-replay-proof-20260918T101515Z-a1360c`.
  This is not another missing-Up event. Test opt-in spike-triggered adaptation
  on ALL anatomically descending/motor cells (not button-selected populations):
  0.02 current increment per spike, 3-second exponential decay. Reuse the
  existing neutral-gray 1 Hz homeostasis protocol to calibrate fixed biases
  for this changed dynamics model, not game behavior. Compare original frozen
  serial controls against the combined adaptation/calibration candidate on
  the same Route-1 state, seeds 1201/1202 and 6,000 decisions. Keep reward and
  decoder settings unchanged. No collision detector, action timer or quota,
  artificial tactile channel, route, punishment or new synapse. This is an
  engineering adaptation hypothesis, not measured fly motor physiology.
  Validate selective current, reset/checkpoint state, legacy exactness and
  neutral firing before gameplay. Positive exploration would still not prove
  learned navigation; require retained learning separately.

- 10:02 UTC: the extended score-v2 paired visual arm drops from 52.0% at
  8,192 to 46.2% at 16,384; shuffled control is still running. More training
  alone has not produced cue discrimination. Test a local presynaptic-centering
  hypothesis: subtract each neuron's prior 10-second release average in the
  existing stochastic-spike innovation tag. Keep neurons, noise, weights,
  calibration, decoder, learning rate, budget and 0.6-second trace unchanged.
  This removes a stimulus-independent presynaptic component from the update;
  it is NOT the exact likelihood gradient of the uncentered physical neuron
  and is NOT validated fly physiology. The conditional spike innovation is
  still zero-mean. No labels/actions/RAM enter the rule. Preselect ROM-free
  paired/shuffled seed 501, checkpoints 512/2,048/8,192; require cue-specific
  retention, reversal and independent training before any gameplay promotion.
  Existing weak/negative curves remain reported; synthetic weights stay out
  of Pokemon. Frozen forward trajectories must match score-v2 exactly.

- 09:59 UTC: after serial transport resolves naming, the navigation hypothesis
  is insufficient movement persistence, not unavailable Up. Preselect ONE
  fixed 3-second direction-rate trace versus the existing 1-second serial
  frozen controls: same seeds 1201/1202, same recorded Route-1 reset, 6,000
  decisions each, no changed function timing/rewards/weights. All directions
  share the trace; decaying actual spikes, not forced-duration actions. Report
  all outcomes, including coverage loss or amplified boundary sticking. This
  tests control capacity only; a favorable path still requires independent
  confirmation and learned retention, not a declaration of solved navigation.

- 09:54 UTC: first timing observer audit (`reward-timing-audit-20260918T095012Z-dcece8`)
  preserves every recorded state/reward and all 119 completed encounters across
  78,000 replayed decisions. Four rewarded encounters pay 6/7/21/24 decisions
  earlier; no added outcomes. This is replay, not new wins. Source inspection
  confirms we can also test the LAST trainer faint: live enemy HP zero, live
  active player HP positive, valid enemy party count/index, and every OTHER
  enemy's cached HP zero. Keep the conservative trainer-victory implementation
  as `confirmed-outcome-v2`; new `last-faint-v3` makes this extra distinction.
  Audit v3 separately before training. Preselect the same recorded rival reset,
  eight training seeds 901..908 and frozen original/retained seeds 1001..1008,
  3,000-decision cap, unchanged dual-trace neural model. No reused frozen
  controls or silent alteration of reward amounts. Also compare a fresh literal
  serial bedroom run against its completed same-seed 24,000-decision control.

- 09:49 UTC: test an opt-in outcome-timing correction, NOT new reward shaping.
  Keep all six categories and magnitudes. Pay trainer wins at the existing
  confirmed trainer-victory hook, captures at successful ball completion, and
  wild wins at the faint hook ONLY with zero live enemy HP and positive live
  player HP; uncertain cases fall back to the existing encounter-end checks.
  Reject demo/link encounters and pay at most once. Preserve legacy timing and
  checkpoints exactly. First replay completed trajectories through both ledgers:
  identical completed-encounter categories/totals, earlier delivery measured,
  no policy changes. Test loss/flee/double-KO/stale-party/checkpoint cases before
  controlled battle learning. This addresses avoidable post-outcome delay, not
  missing long-horizon credit or navigation. No default promotion on speculation.

- 09:37 UTC: after score-v2 passes immediate motor acquisition/reversal, extend
  its EXISTING two-cue training curve (not a new model/parameter fit). Continue
  exact paired/shuffled states from `visual-learning-curve-20260918T071646Z-518b1c`
  at2,048 to preselected8,192 and16,384. Same seed501, unchanged cue mapping,
  reward0 for other choices (not the separate negative-feedback experiment).
  Existing2,048 result47.2% paired versus43.6% shuffled stays reported. Require
  cue specificity, independent-noise retention, reversal and independent training
  seed before game claims. No synthetic weights enter Pokemon. Do not substitute
  a new external critic/encoder or fabricated dopamine anatomy for missing
  behavioral evidence. A longer budget is a diagnostic, not a stopping condition.

- 09:31 UTC: serial frozen comparison complete: both seeds reach only y24 versus
  y20 in reused controls; positions285->337 and303->414, wins0->0 and2->1.
  Clearer command access is not a navigation solution. Next one-factor sensory
  test uses the CORRECTED calibrated/serial circuit, original frozen weights,
  same recorded Route-1 state, seeds1201/1202,6,000 decisions each. Reuse serial
  snapshot controls explicitly; compare endpoint pipeline versus streamed raw
  frames within the same button phases, alternating mode order. Endpoint must
  reproduce snapshot actions/states exactly; stream differs only in fresh
  within-pulse images. Earlier18 short temporal trials used the uncalibrated
  weak-Up model, so they do not establish the effect after these corrections.
  No input features, route, reward or learned decoder is added. No default
  change without useful actual-game evidence and persistence tests.

- 09:29 UTC: existing score-v2 passes the preselected immediate-motor screen:
  Up16.4->26.6% (shuffled17.2%); reversed Down43.8% (32.0% initial,31.3% after
  acquisition,25.4% shuffled). Frozen exactly initial. Artifact
  `operant-motor-probe-20260918T091957Z-f8c78b`. Advance to delay32,2,048 earning
  decisions/stage, seed501, all controls. Candidate uses30s score accumulation;
  unlike EMA traces, this SUM's stationary innovation variance rises with tau.
  Preselect learning-rate rescaling from0.002 by
  sqrt((1-exp(-2*0.02/30))/(1-exp(-2*0.02/0.6))) (~0.00028748), preserving
  baseline update variance in the stationary uncorrelated-score approximation.
  No fitted parameter search, changed neurons/decoder/reward or action feedback.
  It is a temporal-credit plus analytic variance-control hypothesis, not yet
  evidence of delayed learning; require acquisition/reversal against controls
  then independent seed before game promotion. Earlier4x noise-dual failed
  acquisition: paired Up23.8% versus shuffled30.5% and original24.6%.

- 09:26 UTC: readout-noise mechanism audit, not another fitted policy. Feed the
  SAME actual frozen motor spikes from existing acquired original/paired/shuffled
  perturb-v3 brains through parallel-v2 and the already implemented fixed1s
  sustained decoder. Preselect new retention seeds1701/1702,128 neutral warmup
  and128 decisions per cue, report ALL time bins. First audit training seed501;
  if meaningful cue-contrast gain persists versus original/shuffled, also audit
  independent training seed601. No new training/learned thresholds/gains here;
  weights were trained under the original decoder, so any rescore is only a
  mechanism finding and requires new matched training before deployment claims.
  The frozen visual input audit already shows strong cue information upstream;
  this asks whether rapid noisy readout hides retained local sensory contrasts.

- 09:20 UTC: first serial frozen Route-1 comparison is mixed: positions285->337,
  northernmost y20->24, no wins in either; no new town. Do not label increased
  coverage a navigation solution. Fresh serial bedroom run independently wins
  rival at5,092; exact replay validates all states/rewards through5,500
  (`recorded-replay-proof-20260918T091426Z-1c786a`). Delayed noise-dual2048 paired
  acquisition remains flat (23.8% Up vs24.6% before); remaining controls run.
  Next causal-estimator check: EXISTING likelihood-score-v2 model, immediate
  operant feedback,512/stage, seed501, paired/shuffled/frozen, reward-free test
  and reversal. This model's score has an analytic conditional-zero-mean check,
  unlike approximate Bernoulli-current perturbation credit. Require >5pp
  acquisition/reversal advantage versus pre/shuffled before a delayed version;
  do not invent a new optimizer or promote its failed visual screen.
  All10 protected hashes rechecked unchanged at09:17 UTC.

- 09:12 UTC: serial transport passes actual-ROM exact resume/frozen/no-reward
  checks (`internal-smoke-20260918T090405Z-7e75ea`); user legacy checkpoint
  reproduces next20 decisions (`legacy-resume-serial-20260918`). Running serial
  bedroom/frozen tests were launched before logging cleanup: their inherited
  `pulse_frames` reports23 for paired commands, although actual delivery is
  correctly 11+release+11+release. Do not rewrite original evidence or treat
  that legacy counter as held duration. New serial logs include explicit
  phase schedules and22 held frames; UI hover describes sequential delivery.
  This is telemetry only, no behavior change or need to discard ongoing trials.

- 09:10 UTC: delayed-feedback duration control: preselect 2,048 earning decisions
  per acquisition/reversal stage (4x the failed 512-decision screen) with the
  UNCHANGED noise-dual model, seed501, delay32, all paired/shuffled/frozen arms.
  No new rate/trace/bound/feedback parameters. This asks whether the existing
  causally centered noise tag needs more samples; the shorter result remains
  reported and this is NOT an independent seed replication. Require retained
  gains versus both controls before proceeding to a fresh seed or actual-game
  application; diagnostic synapses remain excluded from Pokemon.

- 09:03 UTC: FOUND a concrete simultaneous-button trap. Literal sustained-menu
  run has stayed in Bulbasaur's nickname screen from ~2,326 to >11,500 decisions.
  Party count was already 1 but level was 0: first-party-count alone incorrectly
  labeled a completed starter. Measurement now records both and requires a
  positive initialized level for `first_starter`; old artifacts are not rewritten.
  Pinned naming-screen assembly gives D-pad priority over Start/A/B. Sustained
  directions make almost every function command a combination, so they are
  masked. Optional `serial-v2` delivery splits the SAME 24-frame budget into two
  11-held/1-released pulses, direction then function; no new buttons, RAM input,
  reward or learned adapter. Legacy simultaneous/default paths stay unchanged.
  Forced-input COPY diagnostic `forced-button-delivery-probe-20260918T090315Z-892e4e`:
  all four direction+Start controls remain unfinished after 16 decisions; all
  four serial counterparts initialize party by decision 3. This is transport
  proof, NOT autonomous progress. 45 emulator/temporal tests pass.
  Preselect actual fresh-bedroom seed64, 24,000 decisions, original synapses,
  profile `sensorimotor-serial-v1` (same sustained-menu circuit, timing ONLY).
  Compare full outcomes with ongoing same-seed simultaneous control. Also run
  same original frozen Route-1 controls, seeds1201/1202, 6,000 each, timing ONLY;
  reuse validated completed Start-5s controls explicitly. No waypoint training.

- Delayed-credit 32-decision/512-earning assays finished: all four new/local
  trace variants fail acquisition. Noise-dual paired Up25.0% vs pre24.6% and
  shuffled23.0%; reversal Down24.2% vs shuffled30.9%. Event-dual Up21.9%
  (shuffled26.6%), reversed Down21.9% (shuffled20.7%). Neither is promoted.
  Long-trace presence and kernel amplitude are insufficient behavioral proof.

- 08:50 UTC: parallel comparison for the same identified negative-aftereffect
  mechanism: an event-based local Hebbian tag (pre-trace * actual post-spike),
  with reward centered by the SAME internal reward mean. Remove only the
  adaptive postsynaptic subtraction from eligibility; retain the recorded
  baseline for checkpoint/state compatibility. Same bounded impulse-dual
  constants and delay-32/512 assay, seed501. This tests actual spiking tags
  versus known-noise perturbation tags without changing game feedback or
  giving the learner an action label. Synthetic weights never enter Pokemon.

- 08:46 UTC: kernel counterfactual confirms delayed-tag reversal. One extra
  local event with fixed presynaptic mean, identical subsequent activity:
  after 7.68 s covariance fast/slow differences -0.04935/+0.01029, so original
  mean mixing is NEGATIVE (-0.01953). The adaptive firing baseline carries a
  negative aftereffect from the same event. Known independent-noise centering
  instead gives +0.0000044/+0.02580, both positive. Impulse scaling alone makes
  covariance +0.06274 but did not improve first paired Up behavior (22.7%).
  Test same bounded impulse-dual model with existing perturb-v2 credit ONLY:
  actual known neural-noise innovations, no moving postsynaptic baseline in the
  eligibility signal. Same delay-32/512 protocol and seed501. This is a concrete
  delayed-credit mechanism, not a new reward, forced button, or biology claim.
  `sensorimotor-noise-dual-v1` stays diagnostic until controlled behavior passes.
- Fixed Start cadence completed both frozen controls: positions 230->285 and
  225->303; menu time 37.22->14.07% and 37.08->14.59%; no town in either.
  Wild wins 1->0 and 0->2, so no clear battle conclusion. Literal fresh bedroom
  candidate now runs at port8780 (`internal-learn-20260918T084057Z-b50b15`).

- 08:39 UTC: 32-decision delayed operant paired arms show no acquisition at
  512 earning decisions: short Up 23.4%, dual Up 24.2% versus 24.6% before.
  Shuffled/frozen controls are finishing; do not claim delayed credit works.
  Specific internal hypothesis: averaging EMA eligibility traces gives a new
  event ~50x smaller initial amplitude in the 30 s trace than the 0.6 s trace.
  Add opt-in impulse-balanced mixing: E=(E_fast+r*E_slow)/sqrt(1+r+4r/(1+r)),
  r=tau_slow/tau_fast. The denominator preserves the fast trace's variance in
  a continuous shared-white-innovation approximation, NOT an exact neural
  noise or physiology result. No time constants, learning rate, weight bounds,
  connectivity, decoder or rewards change. Defaults/legacy mixing remain exact.
  Preselect same delay-32 assay seed 501; demand acquisition/reversal versus
  controls, immediate-credit retention, independent seed and real battles
  before any promotion. This is a mechanism test, not an end-game claim.

- 08:32 UTC: replay-only reward-latency audit verifies all states/rewards.
  Rival: faint -> delivery 32 decisions (7.68 neural seconds), confirmed trainer
  victory -> delivery 18 (4.32 s). Wild wins: faint -> delivery 7/21 decisions
  (1.68/5.04 s). Artifacts `recorded-outcome-latency-20260918T082827Z-7d13d1`
  and `...T082850Z-bfb366`. No outcome category/magnitude/timing is changed.
  Preselect ROM-free constant-image operant test with 32-decision reward delay,
  512 earning decisions plus an explicit 32-decision delivery-only tail per
  phase, seed 501, short versus dual trace, acquisition/reversal and matched
  shuffled/frozen controls. This tests delayed causal credit, not Pokemon play;
  trained diagnostic weights remain excluded from the game.
- First Start-cadence comparison: menu frames 37.22% -> 14.07%, sampled positions
  230 -> 285, Route-1 minimum y 22 -> 20, wild wins 1 -> 0. No new town.
  Second seed pending. Improvement in menu interruption does not establish
  navigation or battle-learning success. Audit `recorded-menu-audit-20260918T082938Z-1354e9`.

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
