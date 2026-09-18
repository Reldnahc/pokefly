# Follow-through: movement, internal learning and actual gameplay

Work is ongoing. This supersedes the earlier "no improvement" conclusion for
movement and basic motor learning, but does not claim Pokemon completion or
established screen-specific gameplay learning. Full protocol and failed
candidates remain in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).

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
brain also sometimes gets a starter. Three held-out seeds (701/702/703), each
with original versus retained weights and learning disabled, are in progress.
Each gets the same starting state and 6,000 decisions. All results will remain
in the report, including regressions and failures.

## What still needs evidence

- Useful learned gameplay across repeated independent trials, rather than
  changed weights, greater activity, or a lucky route.
- Screen-conditioned choices. Two-cue/reversal tests currently have mixed
  results. Increased Left/Right activity alone is not visual discrimination;
  conditional accuracy and cue-reversed/shuffled controls also matter.
- Sustained progress beyond the early game; battle/capture competence and badges.
- Faithful biological dynamics. Connectome connectivity alone does not provide
  calibrated physiological parameters or a complete natural learning mechanism.

Two further opt-in hypotheses are being tested: bounded total synaptic input
with flexible individual connections (`sensorimotor-budget-v1`), and credit
from actual existing neural-noise perturbations (`sensorimotor-perturb-v1`).
Neither is promoted merely because it changes weights or passes a motor assay.

## Verification so far

- Exact application checkpoint continuation, including neural arrays and rewards.
- Frozen and zero-reinforcement controls leave original weights unchanged.
- Real live stream matches input pixels, neural buttons and releases; live speed
  changes work with autonomous human controls disabled.
- 178 Python tests, 21 JavaScript tests; lint passing. Additional full-profile
  live checks are pending. A fused eligibility update matches the NumPy rule
  bit-for-bit; its isolated kernel benchmark was about 4x faster, not the game.
- Protected ROM/model/user-run hashes match the original manifest. ROM and
  generated artifacts remain Git-ignored.
- In-app browser unavailable despite retries. No claim of browser-rendered
  visual layout verification; actual application/emulator/HTTP checks did run.
