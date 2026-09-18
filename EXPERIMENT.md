# Baseline internal-learning experiment, version 1

This page specifies the preserved baseline. The implemented graded-vision,
adaptive-KC, and DAN-targeted visual-memory variants are documented separately
in [MODEL_VARIANTS.md](MODEL_VARIANTS.md). Select them explicitly using
`--config configs/hybrid-v1.json`. The main launcher now uses
`configs/sensorimotor-bounded-v1.json`: persistent sensory isolation, fixed
neutral excitability calibration, and bounded internal motor-input learning.
Equations and caveats are in MODEL_VARIANTS; evidence is in
[FOLLOWTHROUGH_RESULTS.md](FOLLOWTHROUGH_RESULTS.md). No variant silently replaces old checkpoints.
New opt-in temporal-input, quiescent-sensory and partial compartment-learning
candidates are also specified in MODEL_VARIANTS.md and evaluated in RESULTS.md.
They do not change the preserved baseline below.
Our goal is a fly that plays and interacts with Pokemon, not game mastery.

This is a runnable research implementation, not a demonstrated Pokemon-playing
fly. Numeric settings below are provisional engineering choices made during
implementation, not user-approved biological constants. The selected user
requirements are preserved in PROJECT_BRIEF.md.

## Data boundary and control loop

Each decision uses the full 160x144 game frame, the fixed grayscale/bilinear
photoreceptor map, and 12 neural steps of 20 ms. Only neural activity reaches
the fixed motor decoder. The chosen buttons are held for 23 emulator frames and
released for one frame. Game time, neural time, and wall time are separate.

Normal training now defaults to no decision limit (`--steps 0`); Ctrl+C saves
at a complete boundary. Explicit budgets remain available for evaluations.
The live speed selector changes only the wall-clock wait between decisions,
including waking an existing wait when the cap changes. Max means no cap;
100x is a requested ceiling, not demonstrated throughput. The display reports
recent actual game speed using completed frames / elapsed wall time (~60 fps).
Every trajectory row logs target rate/revision and actual rate under `pacing`.
Wall-clock measurements, unlike neural/game state, need not match on resume.

The reward observer then reads the game outcome and sends **one scalar reward**
to synaptic plasticity. It never sends position, menu state, opponent identity,
or a suggested action to the sensory encoder or decoder. The dashboard shows
the pre-action frame that actually generated that neural decision, together
with the delivered/released pulse and post-action reward/telemetry.

There is no external learned vision layer, trained button readout, random-button
exploration, direct visual-pathway bypass, or route script in `train`.
`--intro` is an explicit, logged setup intervention; default training does not
script the intro. `play` remains a separately labeled legacy baseline.

## Fixed motor registry

| Output | Type / side | MaleCNS body IDs | Button |
| --- | --- | --- | --- |
| Forward | DNg100 | 10045, 10056 | Up |
| Backward | MDN | 10763, 11288, 11332, 12348 | Down |
| Left steering | DNa02 L | 523769 | Left |
| Right steering | DNa02 R | 10360 | Right |
| Proboscis extension | MN9 / CB0701 | 10331, 16949 | A |
| Escape / takeoff | DNp01 | 10001, 10010 | B |
| Wing song | pIP10 | 11116, 523998 | Start |

MN9 identity is cross-checked against the downloaded brain and the curated
[left](https://www.virtualflybrain.org/blog/2022/01/01/mn9_l-malecns10331-vfb_jrmc20e1/)
and [right](https://www.virtualflybrain.org/blog/2022/01/01/mn9_r-malecns16949-vfb_jrmc20e2/)
Virtual Fly Brain annotations (which cite MaleCNS v0.9). Both body IDs and the
CB0701 alias match the local model. Experimental MN9 function is described by
[McKellar et al.](https://elifesciences.org/articles/54978). This establishes a
candidate biological identity, not faithful behavior in this simplified model.

The decoder uses per-cell mean spike rates, a 150 ms exponential trace, and a
0.5 Hz threshold. Fresh runs use `parallel-v2`: the largest active direction
and largest active A/B/Start score win independently. Thus a valid Up can be
delivered with A, rather than being discarded by A's stronger signal. At most
one direction and one function button are pressed; no diagonals, opposites,
or A+B combinations. Exact ties rotate independently within each channel.
A silent population cannot cause a pulse from residual trace alone. Start has
a fixed one-second neural-time cooldown. These rules never adapt to rewards.
No physical fly body is simulated. Old checkpoints retain the historical global
winner under `exclusive-v1`, including their original tie cursor. To reproduce
that arbitration in a new run set `brain.motor.arbitration` to `exclusive-v1`.
This button-adapter correction does not change the baseline neural equations.

Logs retain a canonical command (`up+a`) and its `buttons` array. Summary
`actions` counts decision windows; `button_counts` counts each delivered key,
so its total can exceed the number of decisions. All keys share the same pulse
and release frame. A menu or obstacle can still consume a delivered input;
the adapter never reads RAM to suppress/remap it.

## What learns

Exactly **61,210 existing excitatory KC-to-MBON edges** are eligible in the
downloaded connectome. The other connections and both adapters stay fixed.
The base connectome files are never modified. Each eligible edge retains its
original sign and is bounded to 0.25–4 times its original magnitude.

The implemented three-factor rule is an engineering hypothesis:

1. Presynaptic spike trace: exponential average, 0.2 s time constant.
2. Eligibility: exponential average of `pre_trace * (post_spike - post_baseline)`,
   2 s time constant; postsynaptic baseline uses a 10 s time constant.
3. At each outcome, `D = tanh(reward)` and
   `weight_factor += 0.02 * D * eligibility`, clipped to the fixed bounds.

`D` is a **synthetic global dopamine-like modulator**, not measured dopamine or
a calibrated PAM/PPL1 compartment circuit. We do not claim that making every
MBON share this rule reproduces fly memory. Real mushroom-body plasticity is
more structured; see, for example, the primary modeling study
[An incentive circuit for memory dynamics in the mushroom body](https://pmc.ncbi.nlm.nih.gov/articles/PMC8975552/).

Intrinsic exploration is the simulator's neuron-level Bernoulli noise:
1.2 Hz, amplitude 0.22. It is not an external random action policy. The
upstream homogeneous LIF equations, tonic drive, and connectivity are retained.
PCG64 noise and fixed-order CUDA sparse reduction replace hardware-sized GPU
noise state and nondeterministic accumulation to support reproducible checkpoints.
That changes random draws/rounding order, not the neuron equations or wiring.

Changing weights is evidence of implemented plasticity, **not** learned game skill.
The tests separately verify that saved weights change actual simulator currents.

## General rewards

| Outcome | Provisional amount | Repeat rule |
| --- | ---: | --- |
| New tile | 0.05 | Once per `(map,x,y)` in continued training |
| New area | 1 | Once per map ID in continued training |
| Ordinary battle win | 1 | Once per completed encounter |
| Successful capture | 1 | Once per completed encounter; exclusive of win |
| Gym-leader victory | +2 | Adds to a confirmed trainer win |
| Rival victory | +2 | Adds to a confirmed trainer win |

No special house-exit/downstairs/starter reward, raw event-bit bonus, party/level
bonus, time cost, button reward, HP shaping, species bonus, or punishment is added.
There is no ordered objective list. A house exit is only an evaluation measurement.

Ordinary genuine repeat wins/catches still pay: anti-farming/diminishing-return
rules are not yet selected. A resumed checkpoint preserves all novelty and
encounter state. `--weights` deliberately starts a **new trial**, not a continuation;
its game/novelty/noise/traces are fresh. Use `--resume` to continue training history.

### Outcome evidence

Instruction hooks are pinned to
[pret/pokered symbols](https://github.com/pret/pokered/blob/3f618d59edf43918f48f5e558c34e04cb2fc5619/pokered.sym)
and checked against short local ROM instruction signatures before installation.
The cartridge SHA-1 must also match the supported release. The observer follows
the [battle code](https://github.com/pret/pokered/blob/a1a22aaf84d1675bcdbaeb194592379d586d838e/engine/battle/core.asm)
and [capture code](https://github.com/pret/pokered/blob/a1a22aaf84d1675bcdbaeb194592379d586d838e/engine/items/item_effects.asm).

An encounter starts at StartBattle. Enemy-faint/trainer-victory execution provides
outcome evidence; EndOfBattle confirms result and a surviving player party before
paying victory. A stale zero result or battle-to-overworld transition is not enough.
The ball-completion hook remembers a successful capture before its RAM field is
cleared, including PC-box delivery and Safari captures. Old-man demonstrations,
link battles, gifts/trades, failed throws, and missing encounter history do not pay.
Gym Giovanni is distinguished from non-gym Giovanni through the game's gym flag.

PyBoy implements execution hooks with temporary in-memory debug opcodes. No
ROM file is modified; hooks are removed while saving and restored afterward.
Unit fixtures cover these outcome branches and the real-ROM tests verify hook
signatures/execution during navigation. Natural end-to-end battle/capture runs
remain unverified: current autonomous trials have not reached those events.

## Persistence and evaluation

Checkpoints contain game state, next input frame, neural voltages/spikes, PCG64
state, all selected weights and eligibility/baseline traces, decoder timing,
reward novelty/encounter history, and experiment configuration. Arrays use NPZ
without pickle. A checksummed completion manifest is written last; old generations
are retained. Ctrl+C is deferred to a completed decision boundary. An exception
mid-decision does not create a falsely resumable partial checkpoint.

Saving does not automatically load the previous fly. A fresh `--intro` launch
starts original weights and fresh state; `--resume CHECKPOINT` restores the
whole saved fly/game. The original files and previous run directories remain
unchanged. Each continuation writes a new directory, whose latest checkpoint
should be selected for the next continuation. Checkpoints are saved every 500
decisions by default and on clean stop. Wall-clock speed is a launch/runtime
setting, not inherited from a checkpoint and not an experiment-model change.

Resume requires matching model/configuration/backend and dependency versions.
Same-machine CUDA tests compare an uninterrupted run with a 20-decision resumed
continuation: actions, rewards, final image, and every saved neural array must
match exactly. Cross-device bit identity is not promised. `--weights` evaluates
retained weights with fresh neural dynamics and the same fixed adapters.

`evaluate` runs matched seeds and the same starting emulator state in learning,
frozen, and absent-reward conditions, plus retained/frozen when a checkpoint is
provided. It reports tiles, rewards, house exits, battle wins, and captures.
It does not automatically announce learning from a small effect or one lucky run.
`internal-probe` tests same-noise repeatability, spatial pixel sensitivity,
selected motors, KC saturation, and a separately labeled fourfold-weight capacity
control. That artificial perturbation is never used by normal training.

## Baseline limitations (retained for comparison)

The local model is not calibrated for biological vision or associative memory.
In the two-seed 384-neural-step probe, all four spatial pairs changed some
decoded actions, and the internal-weight perturbation could change actions.
But all 4,064 KCs averaged above 45 Hz (near the 50 Hz timestep ceiling), and
Up/B each had a spatial pair with zero changed spikes. Usable sensory processing,
balanced motor control, and meaningful learning remain unresolved.

The dashboard is a real 2D spike view, not synapse transmission animation.
Logs retain every action, reward, aggregate neural measurement, and checkpoints;
they are not full screen/per-neuron replay. Unit tests use DOM/canvas stubs.
Real-browser visual verification remains blocked by the browser connection
timing out after retries; HTTP/SSE integration does not replace visual QA.
