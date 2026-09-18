# Pokefly project brief

Draft 0.18 — September 17, 2026

This is a living record of the user's goals and the design discussion. It
distinguishes explicit requests, proposed requirements, and unresolved choices.
It does not mark proposed features as implemented or all recommendations as
approved. Setup and launch commands remain in [README.md](README.md).

## Intended outcome

Connect an open-source simulation based on a fruit fly's reconstructed brain
wiring to Pokemon Red, investigate how it perceives, acts, and learns while playing,
and watch its neural activity while it plays.

The user explicitly clarified: **we want it to play Pokemon, not master Pokemon**.
Winning, completing a route, or optimizing completion time is not the definition
of success. Component measurements and honest limitations must remain visible.

The user explicitly distinguishes learning to play Pokemon from solving a
tailored maze inside Pokemon. Navigation is only one part of the experiment;
the house-exit check must not become the definition of gameplay competence.
The intended controller should be evaluated beyond navigation, including
interactions, menus, battles, and catching Pokemon, without a walkthrough encoded
in its rewards.

The user has selected internal learning: learned changes belong inside the
simulated fly brain. Fly movement outputs should be mapped to Game Boy buttons
through a fixed adapter, rather than a separately trained button-selection policy.

The user also requires pixel-driven vision: start with the full game screen and
map its spatial image into visual-neuron input through a fixed adapter. The
prototype's eight hand-selected visual features are not the target design.

The user requires a human-facing display showing what the fly sees, its brain
activity, and its button presses. Showing rewards and what actually changes
through learning remains a proposed addition to that confirmed core.

## Explicit requests and interests

- Make the open-source fly brain play Pokemon Red.
- Do not redefine the goal as mastering or beating Pokemon.
- Keep the human showcase on one desktop screen without page scrolling; mobile
  optimization is unnecessary. Explanatory prose belongs in documentation, not
  across the main showcase interface.
- Use the legally obtained ROM supplied by the user at the project root.
- Keep the ROM out of Git.
- Gather the supporting software and establish a runnable local experiment.
- Clarify whether the controller can learn and whether dopamine and rewards
  need to be implemented.
- Investigate the existing online visualizations of firing neurons and activity
  through connections, including whether one can be reused here.
- Provide a live human-facing display showing the fly's visual input, its brain
  activity, and its button presses together.
- Change simulation speed while running through the web interface, with an
  actual-speed readout distinct from the requested cap; preserve neural and
  game timesteps, learning rules, and button pulses.
- Normal launches should run indefinitely until stopped, not stop automatically
  at 1,000 decisions. Preserve explicit limits for bounded tests/evaluations.
- Write these goals down, report the contents back, and ask follow-up questions.
- First repeatable evaluation milestone: reliably leave Red's house. This is
  a measurement of progress, not a special reward objective.
- Investigate learning to play Pokemon, not just solving an isolated maze.
  Passing the house-exit check alone does not establish the broader outcome.
- Rewards should express general exploration and achievement categories:
  discovering new tiles and areas, defeating gym leaders, and defeating rivals.
  Do not encode a walkthrough, an ordered list of exact steps, or bespoke
  rewards for leaving Red's house or reaching downstairs.
- Include general battle and capture rewards so winning ordinary battles and
  successfully catching Pokemon receive feedback, not just exploration or
  special-opponent victories. Exact signals and weights remain to be designed.
- Action decisions use screen input. Game memory is allowed only for rewards
  and measurements. The user reaffirmed that RAM and game-state changes may
  be used to evaluate rewards.
- Use raw screen pixels as the visual source, preserving two-dimensional
  spatial information through a fixed pixel-to-neural-input mapping. Do not
  substitute the prototype's eight-feature summary or an external trained
  visual interpreter for this requirement.
- Learning must occur inside the fly model. Map fly movement behaviors to
  buttons; do not train an external machine-learning layer to choose actions.
- Use the direct forward/backward/left/right movement-to-direction mapping.
- Include A, B, and Start in the initial motor-mapping design, not a later phase.
- Use proboscis-extension drive for A, escape/takeoff drive for B, and wing-song
  drive for Start. These are the agreed initial mappings; neural identification,
  validation, and implementation are still required.

### Confirmed first milestone and input boundary

Success on this first check means leaving the house and reaching Pallet Town;
moving from the bedroom to the first floor alone does not satisfy it. Reliability
must be measured over repeated trials. The number of trials, success-rate threshold,
starting-state variations, and action/time budget remain to be agreed.

This is an evaluation milestone only. Leaving Red's house must not receive a
dedicated completion bonus. Discovering Pallet Town can still earn the same
generic new-area or new-tile rewards that apply everywhere else; neither the
destination nor the route receives special treatment.

The visual encoder and button selector must not receive decoded game-memory
features such as player coordinates, map identity, HP, or menu state. Memory
may measure progress and produce the agreed reward signal, including feedback
to a future learning mechanism. Such feedback must not provide action
instructions or a hidden navigation policy.

Completing the entire game, a deadline, a training budget, public hosting, and
a particular visualization style have not been specified.

## Current implementation and evaluation

The user authorized independent implementation and experiments after stopping
their run, with a written plan first. [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)
records the protocol and progress. The protected run stopped at 68,811 decisions;
it had reached Pallet Town without a special house-exit reward. That observation
does not establish reliable house exit or reward-based learning.

New opt-in candidates implement interleaved raw frames, a quiescent unused-sensory
boundary, and a partial compartment/expectation-centered plasticity rule. The
normal sensory-isolated profile, game rewards and motor mappings remain unchanged.
Eighteen matched frozen timing trials do not show a consistent Up improvement;
the stronger sensory boundary also fails its forward-access screen. A saved-
weights versus original-weights outdoor comparison changes some actions, but
does not show consistent exploration improvement. Exact old-checkpoint resume
still passes. Detailed measurements, limitations and the controlled visual-choice
learning assay are recorded in [RESULTS.md](RESULTS.md). All 36 learning-test arms
completed; neither rule established reliable learned choice or reversal. This is
a negative result for the tested model and budget, not proof that flies cannot learn.

The launcher now accepts `-Weights` for a new game with learned connections,
distinct from `-Resume` for exact continuation and `-Intro` alone for a fresh
original brain. Saved brain state is never silently upgraded to a new profile.

The user chose to persist the tested nonvisual sensory isolation for this
experiment. `sensory-isolated-v1` is now the main launcher's default: incoming
network current to 11,931 anatomically selected nonvisual sensory cells is zeroed.
Visual neurons, outgoing signals, tonic input, neuronal noise, motor readouts and
rewards are preserved. This is a controlled model simplification, not a statement
about normal fly biology. The original graph files are untouched, and the
unisolated hybrid remains available as a control. Old checkpoints keep isolation
off; new ones save the setting and selected-cell identity for exact continuation.
Three matched frozen trials increase Up from 2/3/4 to 45/48/43 pulses per 1,000
decisions. Pixel-sensitive responses remain, but visual-KC activity is lower.
These results support forward access, not demonstrated learning or biological
fidelity. The full evidence and tradeoffs are in [RESULTS.md](RESULTS.md).

The user approved correcting button arbitration, then investigating weak forward
activity. Fresh controllers now choose one direction and one A/B/Start command
independently, preserving all mappings, rates, thresholds, cooldowns, and reward
rules. The display highlights both delivered keys. Old checkpoints explicitly
keep their original single-winner decoder; their continuation remains exact.

Three matched frozen 1,000-decision trials now deliver 2/3/4 Up pulses, versus
0/0/0 with the old decoder. Up+A and Up+B also move upward in an isolated clear-
floor emulator check. Neural firing remains weak: a separate frozen audit finds
strong inhibition driven partly by recurrent, unstimulated nonvisual sensory
activity. The initial diagnostic increased forward spikes; that specific incoming-
current gate is now the persisted experimental option described above. Broad
removal of forward inhibition and direct neural stimulation remain diagnostics
only. No tonic-drive changes were promoted.

The explicit `hybrid-v1` variant adds graded photoreceptor/lamina activity,
KC spike adaptation, and a visual-KC/DAN-target-overlap plasticity experiment.
Original connections, ROM, pixel boundary, fixed motor mapping, and general
rewards remain intact. Constants are provisional modeling choices, not biological
calibration. Full anatomical dopamine compartments remain unimplemented; the
new selective modulation is an explicitly labeled approximation.

Matched circuit tests reduce sustained KC activity from approximately 50 Hz to
1.9 Hz. All seven readouts respond to direct neural activation, but natural
pixel-driven Up is still weak. The separate image-learning assay finds stored
weight changes but no reliable image-specific preference or reversal.

The earlier single-winner decoder's three 1,000-decision hybrid gameplay trials
produce more actual tile transitions
and B/Right pulses, but fewer distinct tiles than the baseline. Learning and
frozen controls visit the same 8/8/6 tiles. No delivered Up pulses, battles,
captures, or house exits occur in these trials. This is improved neural stability
and participation, not demonstrated useful learning. See [RESULTS.md](RESULTS.md).

The desktop showcase now uses a viewport-height layout for the screen, retina, anatomical
activity, delivered buttons, and compact metrics. Graded visual signals are
distinct from spikes. Browser visual QA is unavailable because the in-app
browser backend cannot connect; stream and unit tests are separate checks.
The requested UI agent finished and was stopped after delivering the changes.
Exact resume tests include adaptive and graded state. Details and moved display
explanations are in [MODEL_VARIANTS.md](MODEL_VARIANTS.md).

### Earlier baseline implementation (retained)

The autonomous `train` path now implements raw pixels through the fixed retina,
existing fly circuits, and a fixed seven-button motor decoder. Learning changes
61,210 existing KC→MBON connections; no external action policy is trained.
MN9 / CB0701 body IDs 10331 and 16949 identify the agreed A readout. All seven
mappings are connected; biological function and motor calibration remain unvalidated.

General new-tile/new-area, battle-win, capture, gym-win, and rival-win observers
are implemented using exact-ROM execution hooks and RAM. No house-exit, button,
time, raw-event-bit, party/level, or damage-shaping reward is used. Battle/capture
branches have unit-fixture coverage, not yet natural autonomous-play validation.

The initial plasticity rule uses neural eligibility and a synthetic global
dopamine-like gate. It is not a calibrated PAM/PPL1 compartment model. Numeric
parameters and repeat policies are provisional engineering defaults, not newly
approved user decisions; [EXPERIMENT.md](EXPERIMENT.md) specifies them completely.

The dashboard now supports autonomous actions, rewards, and measured weight
changes alongside the observer/manual diagnostic. Checkpoints preserve weights,
neural dynamics, RNG, decoder state, reward history, next frame, and game state.
A real-ROM/CUDA test verifies a 20-decision exact continuation. Frozen/absent-reward
controls keep weights unchanged; retained weights affect real simulator currents.

Learning to play is **not demonstrated**. Three matched 1,000-decision trials
per condition produced zero house exits, battle wins, or captures in learning,
frozen, absent-reward, and retained/frozen conditions. Mean visited tiles were
10.67, 9.00, 9.00, and 11.33 respectively; this small experiment does not establish
reliable learning. A two-seed circuit probe found pixel-dependent choices and
internal-weight influence, but all KCs averaged above 45 Hz, near the model's
50 Hz ceiling. Up/B each had a spatial pair with no changed spikes.

Scientific calibration and useful game competence remain open. Browser visual
QA is still blocked by connection timeouts after retries. Launch instructions
and verification commands are in README.md.

### Earlier pixel-input diagnostic slice (retained)

The user has authorized implementation. The first slice now provides:

- A fixed 2D screen-to-photoreceptor adapter: RGB mean intensity and bilinear
  sampling, with no OCR, learned visual layer, game semantics, or RAM input.
- A source-pinned optic-column map: 2,628 directly annotated and 3,242 inferred
  receptor locations, out of 6,006 receptors. The 136 unplaced cells receive zero
  external visual drive; this does not imply zero recurrent/noise-driven firing.
  The split-eye normalized column chart is an approximate engineering mapping,
  not measured fly optics. Upstream neuron dynamics remain unchanged.
- `vision-probe`: same-noise comparisons of spatial/intensity patterns, a
  repeated-black control, and a separate labeled downstream-injection control.
  The normal pixel-input path does not use that bypass.
- `watch`: a loopback dashboard adapting the fly.ai point/glow renderer, with
  the actual source frame, sampled retinal input, firing neurons, and delivered
  button pulses sharing sample IDs. It displays a 2D anatomical projection, not
  synaptic-edge animation. Missing positions and sampled activity are labeled.
- Observer/Wait-only mode by default. Optional `--manual` pulses are explicitly
  human actions, never reported as fly choices; optional `--intro` scripts only
  initial setup. The dashboard does not add inputs to the neural controller.
- Unit tests and a 50-sample real-ROM/GPU smoke test. The smoke check matched the
  displayed PNG to retinal drive at display precision, verified a delivered and
  released human pulse, and checked the unchanged ROM hash. Browser visual QA
  remains pending because its tool connection has been timing out; JavaScript
  tests use DOM/canvas stubs and do not replace a real-browser layout check.

Local CUDA probes (seeds 64/65, 96 steps or 1.92 neural seconds per stimulus)
found repeatable pixel-dependent spike differences downstream. Across the
left/right and top/bottom pairs, only 1–8 of 4,064 Kenyon cells changed their
spike trains. Every proposed motor population had at least one spatial pair
with zero spike differences. This passes a minimal sensitivity check, not a
usable-vision or gameplay-control validation. No learned improvement is claimed.

The missing internal-plasticity, fixed-decoder, A-identity, and general reward
components now exist in `train`, as described above. Their scientific usefulness
is not established by implementation. No house-exit bonus or hidden external
policy has been added.

Launch commands, artifact locations, and display limitations are in README.md.

## Legacy prototype — facts, not the desired endpoint

The separate `play` controller is a legacy readout-learning baseline. It does not
meet the user's confirmed internal-learning or pixel-driven-vision requirements.
It may serve as an explicit comparison experiment, but it is not the architecture
selected for the target controller.

The local Python/PyBoy prototype runs the MaleCNS-based `flybrain` model on the
available RTX 4070 Ti: 166,700 modeled neurons and 25,582,938 directed modeled
connections. The introductory menus are scripted by default, with neural
control beginning in Red's bedroom.

- Learning currently changes an external button-selection readout. The fly's
  internal connection strengths remain fixed.
- The readout uses descending-neuron activity and a reward-based update rule
  (linear SARSA with eligibility traces).
- The screen is reduced to eight coarse visual features before entering the
  fly brain. This is a temporary baseline shortcut, not the approved target
  sensory interface. Game memory supplies rewards and measurements, not action inputs.
- There is no implemented dopamine-dependent plasticity inside the brain.
- A 500-action test visited 20 tiles across the two floors of Red's house. It
  included 106 exploratory actions. This verifies operation, not improvement
  caused by training. No Pokemon or badges were obtained.
- A subsequent 25-action check loaded the game and readout with learning and
  exploration disabled; the readout weights remained unchanged.
- Screenshots, action logs, rewards, aggregate spike counts, emulator snapshots,
  and readout weights are saved. Full per-neuron firing histories are not saved.
- Checkpoint loading restores the game and readout, but restarts neural dynamics,
  random streams, eligibility traces, and reward novelty history.
- The ROM, saves, model downloads, caches, and run artifacts are gitignored.

## Pixel-driven vision — confirmed requirement, interface implemented

The user approved this intended path:

`Screen pixels -> fixed spatial visual interface -> fly neural circuits -> fixed button mapping`

The source is the full rendered Pokemon screen, with two-dimensional layout
preserved instead of reducing the scene to a few global measurements. Pixels
still need conversion into neural stimulation: "raw pixels" does not mean an
unmapped bitmap can be passed directly to arbitrary neurons. Any resampling,
intensity normalization, temporal filtering, or resolution reduction must be
explicit and visible in the human display. The first diagnostic choices are
recorded above; adequate visual dynamics and biological calibration remain open.

The external visual adapter must not identify doors, enemies, menu choices,
text, or recommended actions. No learned external vision layer, OCR/game-state
parser, or RAM-derived feature may replace the pixel-driven pathway. Its mapping
stays fixed during learning; interpretation and learning belong inside the
modeled fly. The same interface applies during navigation, menus, and battles.

The installed model has an `eye_drive` interface for 6,006 photoreceptors, but
its prebuilt eye mapping stores horizontal azimuth, not a complete 2D retinal
mapping. The supplied [Eyes code](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py)
uses a one-dimensional panorama. The authors also
[report](https://github.com/alextitonis/fly.ai#what-we-found) that photoreceptor
signals fail to propagate through the first visual relay in their tests,
attributing this to their spiking-only model's lack of graded signaling. These
are upstream experimental findings, not a demonstrated pixel-vision capability
or the new local paired-spike test result. Different stimuli and sensitivity
criteria can reveal some spike changes without establishing usable vision.

A usable two-dimensional sensory mapping and adequate early visual dynamics
therefore still need validation beyond the implemented approximation. Measured
neuron soma positions are not automatically visual-field coordinates. Biological
retinal calibration and any changes to early visual dynamics remain design work; merely sending
a larger array does not establish that the model sees the image. Any proposed
bypass of the early visual pathway must be disclosed and discussed, not silently
substituted for the agreed approach.

Proposed validation checks, before interpreting gameplay training as visual learning:

- Verify the fixed mapping preserves distinguishable horizontal and vertical
  image patterns, including localized and moving stimuli.
- Test whether pixel-driven signals propagate beyond the early visual relay,
  reach candidate learning circuits, and can influence the selected motor outputs.
- Use controlled inputs and matched-noise comparisons to distinguish visual
  responses from background firing. Flashing neurons alone are not sufficient.
- Show the actual spatial sensory representation and resulting activity in
  the dashboard; label the old eight-feature controller if used for comparison.

The fixed pixel-input interface is implemented without replacing upstream
visual dynamics. Useful visual processing has not been validated, and the
eight-feature baseline must not be presented as meeting this requirement.

## Learning and reinforcement — confirmed direction, open implementation

The user chose learning inside the fly model, with fly movement behaviors
mapped to Game Boy buttons. External readout learning and combined internal/
external learning are not the selected target. The full discussion now concerns
how to implement and evaluate internal learning.

The intended division of responsibility is:

- Sensory adapter: turn raw screen pixels into spatially mapped neural input
  using a fixed, documented transformation. It must not contain a trained visual
  interpreter or gameplay policy; see the pixel-driven vision requirement above.
- Fly brain: process those inputs, produce movement-related neural activity,
  and retain experience through changes to its modeled internal connections.
- Motor adapter: translate movement-related activity into button presses using
  fixed, documented rules. Rewards must not train this adapter.
- Reward observer: use allowed game-memory measurements to generate feedback
  to the brain's learning mechanism, without selecting actions. RAM and changes
  between game states are permitted evidence for evaluating outcomes; they are
  not direct sensory features or instructions to the button selector.

Adapter settings may need calibration before an experiment. The proposed
evaluation protocol holds them fixed during training and testing, so internal
learning can be assessed separately from interface changes.

For this internal-learning controller, the proposed implementation requirements are:

1. Translate general exploration and achievement outcomes into reinforcement
   signals without encoding an ordered walkthrough or special navigation goals.
2. Model how dopamine-related activity influences plasticity in selected
   connections, initially investigating mushroom-body memory circuits.
3. Use recent neural activity to assign credit to those connections and apply
   bounded changes to their strengths. Merely stimulating dopamine-labeled
   neurons does not establish lasting learning in the current model.
4. Record which connections changed, by how much, and under which learning rule.
5. Preserve the learned state across checkpoints. Define separately what is
   required for an exact interrupted-run continuation.
6. Test whether the resulting behavior improves and whether that improvement
   persists when reinforcement or further learning is disabled.
7. Verify that screen-driven activity reaches the chosen learning circuits and
   that changes in those circuits can influence the selected motor outputs.

The initial experimental rule, connection subset, synthetic dopamine gate, and
intrinsic neural noise are now specified in EXPERIMENT.md; their biological
adequacy remains an open question. Any external random button choices
must be labeled as a separate baseline or intervention; they must not be
silently attributed to the fly's movement decisions.

These are engineering requirements to evaluate, not a claim that biological
learning has already been reproduced.

### Movement-to-button mapping — agreed initial mappings

| Modeled fly output | Confirmed Game Boy action |
| --- | --- |
| Forward movement drive | Up |
| Backward movement drive | Down |
| Left steering drive | Left |
| Right steering drive | Right |
| Proboscis-extension drive | A: interact/confirm/advance text |
| Escape/takeoff drive | B: cancel/back |
| Wing-song drive | Start: open/close the menu |

This is a direct control mapping, not a claim that fly locomotion
naturally has Pokemon's screen coordinates. A heading-based virtual body would
be a different design and has not been selected. Neuron populations, thresholds,
timing, conflicting-output handling, and no-action behavior need calibration.

The user approved extension -> A, escape -> B, and wing-song -> Start as the
initial mapping. All three are now implemented in the fixed decoder. DNp01 and
pIP10 supply escape and wing-song readouts; identified MN9 / CB0701 cells supply
proboscis extension. All readouts still require model-function validation.

Broader feeding-command activity for A, proboscis retraction for B, and a
grooming-related output for Start were discussed but not selected. They are
not automatic fallbacks if a chosen output fails validation; such a change
would need to be discussed.

These are arbitrary, fixed controller assignments, not claims that the fly
naturally means "yes," "no," or "menu." A is not a universal yes signal: its
effect depends on the game screen, while the adapter still emits the same A
button. Menu state from game memory must not choose or remap actions.

The current system has no simulated physical fly body. Initially we would read
activity in identified movement-command or motor-neuron populations, not measure
actual mouth or wing motion. A biological function reported in a real fly must
not be assumed to work faithfully in this simplified neural model.

Proboscis-extension and retraction motor neurons have experimentally characterized
roles ([McKellar et al., 2020](https://elifesciences.org/articles/54978)). Extension
and retraction also occur within a single feeding sequence
([Flood et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3727048/)). Consequently,
A = extension and B = retraction could cause confirm-then-cancel behavior; this is
an engineering risk to test, not an observed result in Pokefly. Retraction should
not simply be inferred from extension activity going quiet.

DNp01 is associated with fast escape takeoff
([Namiki et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6019073/)); pIP10
activation can drive courtship-related wing extension
([Inagaki et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4151318/)). The latter
is a song-related candidate, not a generic flight signal. Neither association
establishes that the chosen simulated outputs are independent or learnable.

A local metadata check on September 17 found two neurons labeled DNp01 and two
labeled pIP10. Proboscis-extension cells have not yet been positively matched
to local neuron IDs; the presence of generic central-brain motor labels or an
MN10 label does not establish that identity. Cross-referencing authoritative
annotations and testing sensory/learning access to each output remain required.

Proposed fixed adapter rules include bounded button presses, explicit release,
and onset detection with a refractory interval for Start to avoid rapid menu
toggling. Pulse lengths, repeat behavior, thresholds, and simultaneous-output
resolution remain open. These rules must be calibrated and then held fixed,
not trained from rewards. Intro scripting remains a separate open choice;
including Start does not itself authorize automated menu navigation.

### Reward design — general motivations, not a walkthrough

The user clarified that the important distinction is not just buttons versus
outcomes. A bespoke reward for leaving Red's house would still encode a specific
step through the game. The selected direction is general exploration and
achievement rules that apply wherever the fly encounters qualifying outcomes,
without an authored sequence of objectives.

The user identified these reward categories:

| Category | Intended general rule |
| --- | --- |
| New tile | Reward discovering a previously unvisited tile, without favoring prescribed coordinates or a route. |
| New area | Reward discovering a previously unvisited area, without a special target destination. |
| Battle victory | Reward verified wins in ordinary wild or trainer battles, not only gym-leader or rival battles. |
| Successful capture | Reward actually catching a Pokemon, not merely selecting or throwing a ball. |
| Gym-leader victory | Reward a verified victory against a gym leader as a category, not a scheduled instruction to defeat a particular leader next. |
| Rival victory | Reward a verified rival victory as a category, not a scheduled next step or merely starting an encounter. |

RAM and game-state transitions may identify these outcomes precisely. Event
flags, trainer identities, and map IDs can be used by the reward observer to
recognize the general categories; identifying an event is different from
prescribing which event the fly must pursue next. The brain still receives screen
input plus reinforcement, not decoded RAM features or an externally selected
next objective. The motor adapter remains fixed.

There must be no bonus just for a button press or fly motor output, no bespoke
house-exit or downstairs bonus, and no hand-authored waypoint or task sequence.
Rules should not be unlocked one at a time to steer the fly along a walkthrough.
The user is choosing general motivations, not removing all preferences from
the reward design.

Leaving the house remains the first evaluation milestone. If that also reveals
a new area or tile, it can receive the same generic novelty feedback as any
other discovery, but no additional house-specific reward. The earlier proposed
house-exit bonus, downstairs bonus, and exit-only reward baseline are rejected.

Reward magnitudes, relative strengths, caps, what constitutes a distinct area,
novelty persistence across attempts/checkpoints, repeated-victory handling,
punishment, and the reward-to-dopamine mapping remain open. The earlier numeric
reward proposal and positive-rewards-only protocol were not approved. Novelty
and achievement bookkeeping will need explicit repeat/farming rules.

The new `GeneralRewards` observer implements only the six agreed categories,
with provisional weights and repeat rules in EXPERIMENT.md. Novelty persists
across full resumes. Encounter-aware detection uses checked instruction hooks,
not a stale result field or party-count increase alone. Natural-game battle and
capture validation remains pending. `ProgressReward.observe` remains only in
the legacy baseline; its event-bit, party/level, time-cost, and per-run-reset
settings are not used by the internal controller.

### Battle and capture feedback — required categories, proposed details

The user requires battle-related rewards that support both winning and catching.
These remain general rules across encounters, not instructions to use a certain
move, select a particular menu entry, catch a named species, or fight a named
opponent next. Rewarding those outcomes is not a guarantee that the current
neural model will learn the necessary behavior.

Beyond the agreed victory and capture categories, these details are proposals:

| Candidate signal | Proposed role and guardrail |
| --- | --- |
| First successful catch of a previously unowned species | Additional collection-novelty reward, without favoring a prescribed species list or order. |
| Defeat an opposing Pokemon within a battle | Smaller intermediate feedback, distinct from winning the whole battle; pay once per opponent. |
| Reduce an opponent's HP | Optional small, capped progress feedback. Track new lowest HP for the same opponent in the same encounter, so healing and re-damaging do not repeatedly pay. |
| Player Pokemon faints or the player loses a battle | Possible negative feedback; penalties and their magnitudes are not approved. |

Successful catching must be worthwhile relative to defeating a wild Pokemon;
damage/knockout rewards must not make destroying every catch opportunity the
dominant paid outcome. Victory, capture, intermediate progress, and gym/rival
bonuses need explicit rules for combination and duplicate prevention. Repeated
easy battles or duplicate catches should be checked for reward-farming incentives;
diminishing-return policies remain to be discussed. No blanket reward for entering
a battle, selecting an attack, throwing a ball, switching, or healing is proposed.
Fleeing should not automatically be treated as losing; any penalty is a separate
design decision.

RAM-based detection must distinguish victory, defeat, escape, and capture rather
than count every battle-to-overworld transition as a win. It must also distinguish
real player captures from the scripted catching demonstration, gifts/trades, and
save-state loads. The game's [RAM definitions](https://raw.githubusercontent.com/pret/pokered/master/ram/wram.asm)
include battle-result and captured-species fields, but these need encounter-aware
validation against the supported ROM; a stale field value is not a new event.

Party-size increases alone are not a capture detector: the game's
[capture code](https://raw.githubusercontent.com/pret/pokered/master/engine/items/item_effects.asm)
also sends caught Pokemon to a PC box when the party is full and updates Pokedex
ownership. A capture observer must handle that case; ownership novelty should
be recorded separately from an ordinary successful capture. None of this RAM
telemetry may become direct input to action selection.

The required victory/capture categories have an encounter-aware observer in
`rewards.py`. The old `RedState` interface stays unchanged for compatibility.
Optional damage/KO/species-novelty shaping and penalties remain proposals, not
implemented paid signals.

### Rewards already present

These are the prototype's current settings, not agreed final reward weights.

| Outcome | Current reward |
| --- | ---: |
| First visit to a tile during a run | +0.2 |
| First visit to a map during a run | +2 |
| Each newly observed event bit | +1 |
| Each increase in maximum observed party size | +2 |
| Each increase in maximum observed total party levels | +0.1 |
| Each newly observed badge | +10 |
| Each decision after gameplay begins | -0.01 |

The initial state establishes the baseline. A repeated tile or event bit does
not earn its novelty reward again during the same run. The event-bit reward is
a rough proxy; individual flags are not all verified story milestones.

Target refinements should validate the agreed general reward categories rather
than turn individual story steps into rewarded objectives. Raw event-bit changes
must not be mistaken for verified victories or captures. Party/level
bonuses and failure signals such as damage, fainting, or stalling penalties are
not yet agreed; the baseline's settings do not settle those choices.

## Human display — confirmed core, diagnostic and autonomous implementation

The user requires a live human-facing display with three core views: what the
fly sees, its brain activity, and its button presses. A single local dashboard
now supports diagnostic and autonomous modes; the eventual layout and 2D/3D
presentation can still change.

Reuse an existing renderer where practical and adapt its connection to the
Pokemon experiment. The preferred starting point is the dashboard in the same
`fly.ai` repository as our neural model.

The first implementation puts visual input and delivered-button controls on
the left and the 2D brain projection on the right. This is an implementation
choice for review, not a newly confirmed user preference.

| Confirmed core view | Proposed implementation details |
| --- | --- |
| What the fly sees | Show the source Pokemon screen and, separately, the actual encoded sensory input supplied to the neural model. Keep both clearly labeled. |
| Its brain | Visualize actual simulated neuron firing, with the option to highlight sensory, learning, and motor-output populations. Do not use decorative activity unrelated to the running model. |
| Its button presses | Show a D-pad plus A, B, and Start; distinguish pressed/held/released states and no-input periods, with a short recent-action history. |

The distinction between source screen and brain input is important: the legacy
`play` prototype reduces the screen to eight feature values, rather than supplying
the full image to a modeled retina. Its sensory-input view should therefore
show those actual values/drives, not label an unmodified screenshot as the
complete representation reaching the brain. If the encoder changes, this view
must follow the real encoder output rather than a cosmetic approximation of
"fly vision."

For the confirmed pixel-driven target, show the actual spatial input mapping
alongside the source screen, including any resampling. Eight feature bars alone
would describe the old baseline, not satisfy the new visual-input requirement.

The button display should report inputs actually delivered to the emulator,
not just the decoder's intended choice. Proposed annotations include the
corresponding fly movement output and whether the action came from the fly,
scripted intro, a manual intervention, or external baseline exploration. Such
interventions must not be presented as neural decisions. Show the active
controller/learning mode so the current frozen-brain baseline cannot be confused
with the future internal-learning controller.

The three views need shared decision IDs or timestamps so the viewer can relate
the observed screen and sensory input to neural activity, button execution, and
the resulting next screen. Display delays, spike aggregation windows, and any
sampling should be labeled. This human observer view must not add RAM-derived
inputs or dashboard overlays to what the fly receives.

Live wall-clock speed controls are now explicitly requested and implemented.
Max removes pacing waits; a 100x target does not establish that throughput.
This is not a new game action or a change to the neural experiment. Requested
and measured rates are logged. Pause/stepping remain unrequested extensions.

Additional proposed dashboard information, beyond the confirmed core:

- Rewards, game milestones, and progress over time.
- When internal plasticity exists, dopamine-related signals and measured
  changes to connection strengths.
- Optional pause or stepping controls, if requested; intervention
  logging and their effect on evaluation would need to be defined.

Our downloaded data already includes measured positions for 140,638 of the
166,700 neurons. The existing dashboard can display this positioned subset;
neurons without positions must not be presented as having measured locations.

The upstream view primarily flashes neurons. Showing activity along selected
connections would require an extension. Connection activity and changing
connection strength should be displayed distinctly. A selected circuit or a
clearly labeled sample is the proposed default instead of displaying every
connection at once. The desired 2D/3D presentation remains open.

The `watch` implementation now streams actual spike IDs and timing, with an
explicit 6,000-cell display cap and separate unsampled motor-cell firing IDs.
All-population counts remain complete. Default aggregation is 200 ms of neural
time, paired with 12 Game Boy frames; the same post-action frame is both shown
and supplied to the brain. Live clients can skip samples, while action logs
retain every completed sample. A restarted experiment requires reloading its
map and control token. Aggregate logs still cannot reconstruct a full
neuron-by-neuron or full-screen replay. Browser visual QA remains pending.

`watch` remains explicitly a frozen-brain observation/manual diagnostic.
`train` uses the viewer for autonomous internally plastic experiments, showing
the pre-action input frame, neural decision, delivered pulse, and post-action
reward. It disables human controls and labels synthetic dopamine and measured
edge changes. Neither path runs the legacy learned readout.

## How we will judge progress — proposal

The first agreed evaluation milestone is reliably leaving Red's house, without
a house-specific reward. Measure repeatable improvement across multiple runs
rather than judging one playthrough. Compare learning-enabled runs against
frozen and random-action baselines with comparable starting states, general
reward rules, and action budgets.

This first check tests basic control and exploration, not general Pokemon
competence. The proposed broader evaluation should examine the same internally
learning controller across exploration, dialogue/menu interactions, battles,
and catching Pokemon.
Use the same fixed adapters and general reward definitions across these contexts;
do not replace them with task-specific external policies or a sequence of
hand-authored next objectives. Evaluation checkpoints are measurements, not
instructions supplied to the fly. Exact tests and success criteria remain open.

For a dopamine/plasticity claim, additionally compare plasticity disabled and
shuffled or absent reinforcement. Check retention using a fixed action decoder
where possible, and report negative results. Changed weights or flashing
neurons alone are not evidence of better game performance.

The eight-feature visual encoder is a known limitation and is not the selected
target. Validate the required pixel-driven visual pathway alongside the planned
learning circuits; reward design, persistence, and evaluation also need work.

## Confirmed decisions and continuing discussion

| Question | Agreed direction | Status |
| --- | --- | --- |
| Where should learning happen? | Inside the fly model; map fly movement outputs to buttons through a fixed adapter. | Confirmed; experimental KC→MBON plasticity implemented, effectiveness unproven |
| What is the broader experiment? | Learning to play Pokemon across its gameplay contexts, not merely navigating a tailored maze. | Confirmed direction; broader evaluation details remain open |
| What is the first repeatable evaluation milestone? | Reliably leave Red's house and reach Pallet Town; no special completion bonus. | Confirmed by the user; measurement only |
| What information may drive button decisions? | Screen input; RAM and game-state changes may supply rewards and measurements, not direct action instructions. | Confirmed and reaffirmed by the user |
| How should the screen reach the fly? | Raw screen pixels through a fixed, spatially preserving visual-neuron interface; no eight-feature substitute or external trained visual interpreter. | Confirmed; approximate 2D interface implemented and tested; useful visual processing remains unvalidated |
| What should rewards encourage? | General exploration, battling, and collecting: new tiles/areas, battle victories, successful captures, and gym-leader/rival victories; no ordered walkthrough or bespoke house-exit reward. | Confirmed direction and categories; numeric settings and detection details remain open |
| Should ordinary battles and catching have feedback? | Include general battle-win and successful-capture rewards, not only exploration or special-opponent achievements. | Confirmed by the user; species-novelty, damage/knockout shaping, and penalties remain proposals |
| How should movement map to directions? | Forward to Up, backward to Down, left steering to Left, right steering to Right. | Confirmed; fixed readouts and provisional timing implemented, calibration unvalidated |
| Should A, B, and Start be included now? | Include all three in the initial mapping design. | Confirmed by the user |
| How should A, B, and Start map? | Proboscis extension to A, escape/takeoff to B, wing-song to Start. | Confirmed; MN9, DNp01, pIP10 wired to decoder; model-function validation remains |
| What must the human display show? | The fly's visual input, its brain activity, and its actual button presses together. | Diagnostic/autonomous display implemented; real-browser visual QA blocked by tool connection |

Starter acquisition and the first rival battle are not the selected evaluation
milestone or a prescribed next-step sequence. Rival victories are now an agreed
general reward category; a special starter-acquisition reward has not been agreed.

Further design choices can wait: the final gameplay target, preferred 2D/3D
brain view, how much of the intro should be scripted, run duration, and whether
the dashboard needs user controls in addition to observation.

The initial motor-to-button assignments and pixel-driven visual direction are
agreed. The continuing learning discussion should cover the 2D sensory mapping
and visual-pathway dynamics, the dopamine/plasticity mechanism, motor-readout
calibration, and how to measure retained learning.

## Suggested order of work — not yet a committed schedule

1. Continue the full conversation about implementing internal learning and fixed
   motor mappings. The learning location, first milestone, allowed inputs, and
   initial movement-to-button assignments are confirmed, as are pixel-driven
   vision and general exploration/achievement rewards. Interface details, reward
   settings, and the learning mechanism remain open.
2. Establish and validate a fixed 2D pixel-to-visual-neuron mapping and working
   early visual pathway, including access to learning and motor-output circuits.
3. Adapt the existing renderer into a synchronized human dashboard for visual
   input, brain activity, and executed button presses; label the active baseline
   or internal-learning controller accurately.
4. Improve experiment logging, define the general reward rules and their RAM-based
   detectors, and define a separate milestone evaluation protocol; refine the
   validated visual interface as needed before fixing an experiment configuration.
5. Implement the selected internal-learning controller with fixed adapters and
   saved learned brain state, then compare it against the documented baselines.
6. Evaluate broader gameplay as capabilities emerge, without turning evaluation
   checkpoints into an ordered reward schedule or walkthrough.

The runnable internal-learning architecture is implemented. Useful vision,
motor/dynamics calibration, natural battle/capture validation, and demonstrated
retained gameplay improvement remain open research and verification work.

## Reusable work discussed

- [fly.ai](https://github.com/alextitonis/fly.ai): current neural simulator;
  the documented model has no neuromodulation or internal plasticity.
- [fly.ai SSH Fighter dashboard](https://github.com/alextitonis/fly.ai/tree/main/sshfighter):
  `fly_dashboard.py` and `dashboard.html`, showing firing neurons at measured
  positions. Reuse must preserve the applicable upstream notices.
- [DOOMFLY](https://github.com/nftechie/doomfly): experimental dopamine-gated
  plasticity on selected existing mushroom-body connections. Its authors report
  failed validation gates and do not claim demonstrated learned survival; it is
  a research reference, not a proven solution to import unchanged.
- [MaleCNS](https://male-cns.janelia.org/): the underlying connectome dataset,
  with attribution and licensing separate from the application code.

Source descriptions above were reviewed in the preceding discussion on
September 17, 2026. Reused fly.ai sources are pinned to
`9af6fb6345cb55e359e71d09fae99f2c6dd46456`; optic-column annotations are pinned to
`67767d2233657983993ff6c2be48e836a935863c`. See THIRD_PARTY_NOTICES.md for notices,
data attribution, and the annotation checksum.
