# Pokefly

Project goals, proposed features, and open decisions are recorded in
[PROJECT_BRIEF.md](PROJECT_BRIEF.md).

The confirmed target is pixel-driven vision through a fixed spatial input
adapter, learning inside the fly model, and fixed movement-to-button mappings.
The pixel-input pipeline, fixed seven-button decoder, internal synaptic
plasticity, general outcome rewards, checkpointing, and live dashboard now run.
This is an experimental implementation: basic internal motor learning is now
demonstrated, but useful screen-specific gameplay learning is not established.
The learning rule, provisional settings, and limitations are in
[EXPERIMENT.md](EXPERIMENT.md).
The goal is participation, not mastering or completing Pokemon. Current evidence
is in [FOLLOWTHROUGH_RESULTS.md](FOLLOWTHROUGH_RESULTS.md): neutral neural
excitability calibration increased Up from 3.9% to 22.1% and exploration from
25 to 232 positions in a matched 6,000-decision comparison. Bounded internal
plasticity learns and reverses motor preferences against frozen/shuffled controls.
Actual learning runs obtained starters; one won the first rival battle and
entered Route 1. Saved weights obtained starters in 3/3 held-out trials versus
1/3 with original weights, but did not improve battle wins in that whole-game
panel. Separately, a dual-trace candidate trained on eight explicitly reset
rival encounters won 6/8 on each of two frozen test panels, versus 4/8 and 3/8
with original weights. This is limited battle retention, not whole-game
autonomy or statistical proof. Screen-conditioned learning remains unproven;
experiments continue. These early results do not
prove eventual game completion. Failed earlier
experiments remain in [RESULTS.md](RESULTS.md), with the continuing protocol in
[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).

A running experiment connecting Pokemon Red in
[PyBoy](https://github.com/Baekalfen/PyBoy) to the 166,700-neuron MaleCNS
connectome simulation in [fly.ai](https://github.com/alextitonis/fly.ai).

`train` is the selected architecture: pixels in, fixed movement-to-button mapping
out, and learning on existing connections inside the simulated brain. The current
launcher uses 495,962 existing excitatory inputs to anatomically classified
descending/motor neurons, not a learned button adapter. Preserved KC-to-MBON
controls have 61,210 baseline or 4,407 visual-KC eligible edges.
See [MODEL_VARIANTS.md](MODEL_VARIANTS.md) for equations and limitations.
`watch` is a frozen-brain diagnostic with optional human controls.
`play` preserves the older eight-feature/external-readout baseline separately.

## Run on this machine

Python dependencies, the public connectome, and the derived retina map are
installed locally. The user's ROM at the project root is discovered automatically.
Launch the autonomous experimental controller with its live display:

```powershell
.\.venv\Scripts\python.exe -m pokefly train --device cuda --intro `
  --config configs/sensorimotor-bounded-v1.json
```

Shortcut: `.\scripts\start.ps1 -Intro`. Normal training now runs until Ctrl+C
by default; use `-Steps 1000` (CLI: `--steps 1000`) for a bounded trial.
The launcher defaults to `sensorimotor-bounded-v1`. Use `-Profile intrinsic-v1`
for calibrated dynamics with the earlier KC rule, `-Profile sensory-isolated-v1`
for the earlier uncalibrated model, or `-Profile baseline` for original dynamics.
Bare `python -m pokefly train` without a config still selects the baseline.
Fresh runs use the corrected `parallel-v2` decoder: one direction plus one of
A/B/Start, with the same fixed thresholds and explicit release. Old checkpoints
retain `exclusive-v1` on resume; start a fresh run to use the corrected adapter.

Open [the local dashboard](http://127.0.0.1:8777). It shows the exact frame fed
to the brain, retinal input, real firing activity, delivered neural button
pulses, outcome rewards, and measured internal weight changes. Hybrid mode
separately shows blue graded visual activity and amber spikes. Human controls
are disabled during autonomous experiments. `--intro` explicitly scripts setup
to the bedroom; omit it to let the fly attempt the opening screens.

The **Speed** selector changes pacing during a run: 1x, 2x, 5x, 10x,
100x target, or Max (uncapped). **Actual** is recent completed game time per
wall-clock second, approximately 60 Game Boy frames = 1x. A target is a cap,
not a promise that the hardware can reach it. The default cap is five decisions
per second: at 24 game frames per decision, about 2x. Use `-Hz 0` in the launcher
(CLI: `--hz 0`) to start uncapped while keeping the display. Speed changes only
waiting between decisions, never neural dt, learning constants, or button pulses.
Controls require a live same-origin session and never enable human game buttons.
Already-running older processes need one restart/resume to load this feature.

The showcase uses a desktop-only, viewport-height layout: no page scrolling or
mobile stacking. Longer explanations live in documentation and hover details.
Actual layout verification at desktop resolutions is still pending because the
in-app browser is unavailable; UI logic/CSS-contract tests are not visual QA.

Ctrl+C finishes the current decision and saves. `--steps 0` runs until stopped;
`--no-dashboard --hz 0` runs an unthrottled headless experiment. Each run is saved
under `runs/internal-.../` with configuration, trajectory, summary, images, and
checksummed checkpoint generations. The ROM and original connectome are untouched.

**Saving is automatic; loading is explicit.** Each fresh launch (including
`-Intro`) starts a new fly from the original model, not from the last run.
It does not delete earlier learning. Checkpoints save every 500 decisions and
on a clean stop. A crash can lose work since the last completed checkpoint.
Completed checkpoints and logs are retained; long unlimited runs use increasing disk space.

Use **resume** to continue the entire saved experiment: game, learned internal
weights, neural activity, random stream, decoder traces, and reward history.
Each resumed session writes its own run directory; use that session's checkpoint
for the next continuation:

```powershell
.\scripts\start.ps1 -Resume .\runs\YOUR_RUN\latest-checkpoint.json
```

For a new retention trial, use `--weights CHECKPOINT --mode frozen --load-state STATE`.
`--weights` keeps learned connections but starts fresh neural dynamics and reward
novelty; it is not a continuation. To keep training the same fly/game, use `--resume`.
The launcher now supports the same distinction:

```powershell
# Start from the bedroom again, retaining this fly's learned connections.
.\scripts\start.ps1 -Weights .\runs\YOUR_RUN\latest-checkpoint.json -Intro
# Add -Mode frozen to measure retention without further learning.
```

`-Weights` and `-Resume` are mutually exclusive and restore their saved profile.
Do not combine either with `-Profile`. `-Intro` alone still starts original weights.
Wall-clock pacing is independent of the saved brain: each launch uses its `-Hz`
or `--hz` setting (default 5), which the live selector can then change.
`--mode no-reward` runs a zero-reinforcement control; `--mode frozen` retains
measured rewards but forbids weight changes. Configurable experimental parameters
are accepted with `--config PATH.json` and saved in every run. Checkpoint modes
and configuration cannot silently change during an exact resume.
Old checkpoints retain their original isolation, calibration and learning rule.
Start a fresh trial using its config and `--load-state` (launcher: `-LoadState`)
to reuse the game's starting state. No silent
conversion of old learned weights or full checkpoints is performed.

Opt-in research profiles use `-Profile stream-v1`, `endpoint-v1`, `quiescent-v1`,
or `compartment-ema-v1`. These are versioned tests, **not proven improvements**.
Stream shows the last frame of a multi-image input window; the other profiles
are timing, sensory-boundary and internal-plasticity controls. Equations,
anatomical limitations and checkpoint semantics are in
[MODEL_VARIANTS.md](MODEL_VARIANTS.md).

```powershell
.\.venv\Scripts\python.exe -m pokefly prepare
.\.venv\Scripts\python.exe -m pokefly evaluate --device cuda `
  --load-state .\runs\YOUR_PREPARE_RUN\start.state --trials 3 --steps 1000
.\.venv\Scripts\python.exe -m pokefly circuit-evaluate --device cuda
.\.venv\Scripts\python.exe -m pokefly association-evaluate --device cuda --trials 3
```

Evaluation compares learning, frozen, and absent-reward trials with matching
seeds; add `--checkpoint PATH` to compare retained weights frozen as well.
House exit is measured, never given a special reward. General battle/capture
detectors are implemented, but autonomous live battle/capture outcomes have not
yet been reached. Sensory isolation increases Up to 45/48/43 pulses per 1,000
decisions in matched frozen trials (unisolated: 2/3/4). Pixel sensitivity remains,
but visual-memory-cell activity is lower. There is no validated learning-to-play claim.

### Optional observation/manual diagnostic

To inspect the frozen pixel-driven brain with human controls:

```powershell
.\.venv\Scripts\python.exe -m pokefly watch --device cuda --manual
```

Open [the local dashboard](http://127.0.0.1:8777). It shows the real screen,
photoreceptor input, anatomical neuron activity, population spike counts, and
delivered button pulses. Click the buttons or use arrows, A, S (B), and Enter
(Start). Controls are human input, explicitly labeled; the fly is not selecting
buttons. Omit `--manual` for an observer that only sends Wait.

The default starts at the opening screens. Add `--intro` to explicitly script
the menus/naming and begin in Red's bedroom, or `--load-state PATH` to use an
existing emulator snapshot. Scripted intro actions are counted separately and
are not fly behavior. Loading a state does not restore neural voltages or RNG.

Ctrl+C saves and stops. `--steps 50` stops after 50 samples; the default runs
until interrupted. `--port 8778` selects another loopback port. The server does
not expose the ROM or serve arbitrary files; manual requests require a
same-origin token. Do not expose this development server through a proxy.

Every observation run creates an ignored `runs/vision-watch-.../` directory with
`config.json`, `trajectory.jsonl`, start/latest PNGs and emulator states, and
`summary.json`. Logs contain actual actions, spike counts, and RAM telemetry.
They are **not a complete screen/spike replay**. Snapshot files update every
100 samples and on normal exit; they never overwrite the supplied ROM or saves.

### What the pixel path does

The full 160x144 screen is converted to grayscale and bilinearly sampled at
fixed 2D photoreceptor coordinates. There is no OCR, object detector, learned
encoder, RAM-derived visual feature, or direct LC-neuron bypass in `watch`.

The map covers 5,870 of 6,006 photoreceptors: 2,628 directly annotated and 3,242
placed via their strongest annotated synaptic partner. The 136 unplaced cells
receive zero **visual drive**; they can still fire from network activity/noise.
The split-eye normalized optic-column chart is an engineering approximation,
not measured optical geometry. Source revisions and checksums are retained.

Each default sample advances 12 Game Boy frames, then feeds the resulting
screen to 10 neural steps (200 ms). That same screen, drive, and spike window
share a sample ID in the viewer. The target rate is five samples/second; game,
neural, and wall clocks are logged separately. Slower clients receive the latest
sample and may skip intermediate updates; the action log remains complete.

The 2D point viewer adapts fly.ai's renderer. It uses the 140,638 cells with
anatomical positions; it does not invent positions for the others. Positions
are normalized/clipped at the 0.2/99.8 percentiles for display. Up to 6,000
positioned firing cells are shown per sample, with a short fading glow. Candidate
motor-cell highlights use exact individual firing IDs; population counts include
all cells and all spikes. The retina panel is an 8-bit display copy of the float32
drive. No synaptic-edge animation or biological-vision validity is claimed.

### Rebuild and probe vision

```powershell
.\.venv\Scripts\python.exe -m pokefly vision-prepare
.\.venv\Scripts\python.exe -m pokefly vision-probe --device cuda
```

`vision-prepare` fetches/checks the pinned public optic-column annotations and
builds `fly-data/retina.npz`. `vision-probe` needs no ROM. It compares black/white,
left/right, top/bottom, and checkerboard stimuli with identical neural noise
within each pair, plus a repeated-black control. A separately labeled LPLC2
injection checks downstream responsiveness; it is never used by `watch`.
Reports are saved in ignored `runs/vision-probe-.../report.json` files.

On the tested CUDA configuration (seeds 64 and 65, 96 steps per stimulus), the
repeatability check passed and some downstream spike trains differed with spatial
input. However, only 1–8 of 4,064 Kenyon cells differed across the spatial pairs,
and every proposed motor population had at least one pair with zero differences.
The A population is now identified as MN9 / CB0701 (body IDs 10331 and 16949).
The necessary sensitivity
gate is **not** a useful-vision, autonomous-control, or learning certification.

## Legacy external-readout baseline

This remains available for comparison; its trained external policy is not the
user's selected learning architecture:

```powershell
.\.venv\Scripts\python.exe -m pokefly play --device cuda --steps 5000
```

It scripts the opening menus and naming screens, stops that script in Red's
bedroom, and hands control to the fly. Intro actions are counted separately in
the run configuration. Close the window or press Ctrl+C in the terminal to stop.
Use `--speed 1` for normal emulator speed, or `--headless` for faster experiments.

Every run creates a new ignored `runs/fly-.../` folder containing:

- `start.png` / `latest.png`: screen captures, updated every 100 decisions.
- `start.state` / `latest.state`: emulator snapshots.
- `readout.npz`: learned button-readout weights.
- `trajectory.jsonl`: actions, exploration flags, neural spikes, rewards and game state.
- `config.json` / `summary.json`: settings and measured results.

Load a previous experiment (replace the example folder with your run):

```powershell
.\.venv\Scripts\python.exe -m pokefly play --device cuda --steps 5000 `
  --load-state .\runs\YOUR_RUN\latest.state `
  --checkpoint .\runs\YOUR_RUN\readout.npz
```

This restores the game and learned readout. Brain voltages, random streams,
eligibility traces, and novelty history restart; it is not an exact continuation
of the full training process. Existing milestones establish the new run's
baseline. Add `--no-learn --epsilon 0` to evaluate the fixed readout without
exploratory button choices. This alone does not establish that learning helps;
that requires repeated comparisons against untrained and random policies.

## Recreate the environment

Use Python 3.12 on Windows, then:

```powershell
.\scripts\setup.ps1
```

The script creates `.venv`, installs the tested versions from
`constraints-win-py312.txt`, and downloads approximately 258 MB of connectome
data with SHA-256 verification. GPU runtime dependencies require additional
disk space (several GB). Use `-CpuOnly` on a machine without an NVIDIA GPU.

Manual equivalent:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c constraints-win-py312.txt -e ".[gpu,dev]"
.\.venv\Scripts\python.exe -m pokefly download
.\.venv\Scripts\python.exe -m pokefly vision-prepare
.\.venv\Scripts\python.exe -m pokefly doctor --brain
```

`FLY_DATA` optionally overrides `fly-data/`. By default model data and compiler
caches stay in this project. `.env.example` is documentation; the CLI does not
load `.env` automatically. The Windows CuPy wheel currently emits a CUDA-path
warning even when its pip-provided runtime works; `doctor --brain` actually
executes the legacy feature-driven network to test it. Use `vision-probe` for
the new pixel-input path. Setup also downloads the small public optic-column
table and builds the retina map.

PyBoy 2.7.0 is pinned because
[2.7.1 was withdrawn for a Red/Blue rendering regression](https://pypi.org/project/pyboy/).

## ROM and save handling

Supply your own legally obtained `.gb` file in the project root or `roms/`.
If multiple ROMs are present, pass `--rom 'path with spaces.gb'`. No ROM is
downloaded or distributed. The memory adapter validates the exact English Red
release against [pret/pokered's published SHA-1](https://github.com/pret/pokered):
`ea9bcae617fdf159b045185467ae58b2e4a48b9a`. Other regions and ROM hacks require
an adapter before they can use the RAM-based rewards.

ROMs (including uppercase extensions), saves, connectome files, caches and run
artifacts are gitignored. Experiments start with isolated cartridge RAM and
save their snapshots in their own run folder. Adjacent personal battery saves
are neither loaded nor overwritten.

Other commands:

```powershell
.\.venv\Scripts\python.exe -m pokefly inspect
.\.venv\Scripts\python.exe -m pokefly prepare
.\.venv\Scripts\python.exe -m pokefly manual
```

`inspect` validates and boots the ROM, reports state, and saves a screenshot.
`prepare` saves a repeatable bedroom state after the scripted intro.
`manual` lets you make a custom starting state: arrow keys move, A is Game Boy A,
S is B, and Enter is Start. Closing the window saves `start.state` in a new run
folder. `play --no-intro` lets the fly attempt the opening screens itself.

## How the legacy controller works

1. Reduce the 160x144 screen to eight features: upper/lower darkness, motion and
   contrast in each screen half.
2. Inject those features into left/right LC10a, LPLC1, LC4 and LPLC2 populations.
   These assignments are an experimental encoding, not biological equivalences.
3. Advance the full frozen connectome for 12 steps (240 ms of neural time by
   default). Game and neural clocks are explicitly separate.
4. Read decaying spike traces from all 1,314 descending neurons.
5. Choose Up/Down/Left/Right/A/B/Start/Wait through a linear SARSA(lambda)
   readout, with 20% random exploration by default. Hold the button for 23
   emulator frames and release it for one frame.

WRAM supplies rewards and telemetry, not policy inputs. New tiles, maps, event
bits, party members, levels and badges earn rewards; each milestone is paid
once per run. Intro scratch memory is excluded. The default reward is crude,
and eight screen features discard most of the information in the game.
The new `train` pipeline is separate: it uses internal plasticity, fixed motor
readouts, and verified-outcome reward observers. Its neural dynamics and motor
calibration still need scientific validation. The legacy event/party/level
rewards are not used by `train`.

## Verification so far

On an RTX 4070 Ti with the supplied English ROM:

- The real pixel-dashboard smoke test completed 50 samples. It checked that the
  displayed PNG exactly reproduced the retinal drive (at display precision),
  that a human pulse was delivered/released/logged, and that the ROM hash did
  not change.
- Python tests cover the mapping, motor decoder, internal plasticity, graded
  dynamics, outcome reward fixtures, server boundaries, and baseline behavior.
  JavaScript tests exercise the shipped UI logic with DOM/canvas stubs: these do
  not verify real-browser layout. Browser visual QA remains pending because the
  in-app browser backend is unavailable after retries.
- The internal-controller smoke test verifies a 20-decision exact CUDA resume,
  retained weights affecting simulator current, unchanged frozen/no-reward weights,
  and the unchanged ROM hash. See `scripts/smoke_internal.py`.
- Circuit evaluation compares the preserved baseline, adaptation-only, and
  hybrid dynamics. Separate image conditioning tests retention, nonpredictive
  rewards, and reversal. See [RESULTS.md](RESULTS.md) for positive and negative
  findings; neither sensitivity nor changing weights establishes useful learning.

The following older measurements concern **only the legacy baseline**:

- 500 decisions completed at about 30 decisions/s, visiting 20 distinct tiles
  across Red's bedroom and first floor. 106 actions used random exploration;
  394 came from the neural readout. No Pokemon or badges were obtained.
- A fixed-readout check loaded the saved game and weights and ran another 25
  decisions with no exploration and exactly zero weight change.
- A same-seed bright/dark input probe changed the descending-neuron features,
  demonstrating a functioning sensory-to-output path.
- Unit tests cover ROM selection, reward baselines/repeated milestones,
  button release, visual encoding and checkpoint compatibility.

These are integration checks, not evidence of superior learning or an ability
to complete Pokemon Red. The actual game runs above were headless; launch
`play` without `--headless` to watch locally.

```powershell
.\scripts\test.ps1 -Live -Device cuda
.\.venv\Scripts\python.exe -m pokefly vision-probe --device cuda
.\.venv\Scripts\python.exe -m pokefly internal-probe --device cuda
.\.venv\Scripts\python.exe -m pokefly doctor --brain --device cuda
```

Node is only needed for the JavaScript unit tests, not for running the dashboard.
`scripts/test.ps1` uses fresh project-local temporary/cache directories to avoid
Windows permission conflicts with earlier sandboxed runs. Omit `-Live` to skip
the ROM/GPU integration test. Python tests and lint always run; JavaScript unit
tests run when Node is installed (`node --test tests/dashboard.test.cjs`).
The live suite covers baseline and every versioned profile, including temporal
input, quiescent sensory cells and the partial compartment candidate. These
checks use the user's local ROM, start their own short-lived localhost servers,
and preserve evidence in new run directories. Passing them establishes software
behavior and persistence, not learned gameplay.

## Attribution

Pinned reuse revisions and the full upstream MIT notice are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), also included in the package.

- [fly.ai / flybrain](https://github.com/alextitonis/fly.ai): Alex Titonis and
  contributors, MIT-licensed simulation and derived model distribution.
- [MaleCNS v1.0](https://male-cns.janelia.org/): FlyEM, HHMI Janelia and
  collaborators; connectome data and derived files remain under CC BY 4.0.
  See the dataset's attribution terms and Berg et al. (2026).
- [PyBoy](https://github.com/Baekalfen/PyBoy): emulator and input/screen API.
- [pret/pokered symbols](https://github.com/pret/pokered/tree/symbols): memory
  layout and supported-ROM identification.
