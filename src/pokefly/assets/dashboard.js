/* Brain point/glow rendering adapts fly.ai's MIT dashboard; see THIRD_PARTY_NOTICES.md. */
"use strict";
const $ = (id) => document.getElementById(id);
const brain = $("brain"), bctx = brain.getContext("2d");
const game = $("game"), gctx = game.getContext("2d");
const retina = $("retina"), rctx = retina.getContext("2d");
const buttons = [...document.querySelectorAll("[data-action]")];
const signals = {
  photoreceptors: "Photoreceptors", lamina_L1_L5: "Lamina L1–L5",
  visual_projection: "Visual projection", Kenyon_cells: "Kenyon cells",
  MBONs: "Mushroom-body outputs", descending: "Descending neurons",
  forward_Up: "DNg100 → Up", backward_Down: "MDN → Down",
  steer_Left: "DNa02 L → Left", steer_Right: "DNa02 R → Right",
  proboscis_A: "MN9 / CB0701 → A",
  escape_B: "DNp01 → B", song_Start: "pIP10 → Start",
};
let state = null, base = null, glow = null, stream = null;
let drawnSample = -1, lastReceived = 0, live = false;
let connectionEpoch = 0;
let speedRevision = -1, speedHz = null, speedPending = false;
const px = (i) => 18 + state.x[i] * (brain.width - 36) / 1000;
const py = (i) => 18 + state.y[i] * (brain.height - 36) / 1000;

function setLabel(id, text, detail = text) {
  $(id).textContent = text;
  $(id).title = detail;
}

function connection(ok, message) {
  if (!ok) connectionEpoch += 1;
  live = ok;
  $("connection").textContent = message;
  $("connection").classList.toggle("live", ok);
  for (const button of buttons) button.disabled = !ok || !state?.manual;
  $("speed-target").disabled = !ok || !state?.speed_control || speedPending;
  if (!ok) setLabel("speed-actual", "Actual —", "Waiting for a live speed measurement");
}

function showPacing(pacing, force = false) {
  if (!state?.speed_control || !pacing || !Number.isFinite(pacing.target_hz)
      || !Number.isInteger(pacing.revision) || pacing.revision < speedRevision
      || (pacing.revision === speedRevision && !force)) return;
  speedRevision = pacing.revision;
  speedHz = pacing.target_hz;
  const multiplier = speedHz * state.emulator_frames_per_sample / 60;
  const preset = speedHz === 0 ? 0
    : [1, 2, 5, 10, 100].find(value => Math.abs(value - multiplier) < 1e-8);
  $("speed-custom").hidden = preset !== undefined;
  $("speed-custom").textContent = `${multiplier.toFixed(2)}× target`;
  $("speed-target").value = preset === undefined ? "custom" : String(preset);
  setLabel("speed-status", speedHz === 0 ? "Uncapped" : `Target ${multiplier.toFixed(1)}×`,
    "Target, not a performance promise. Actual speed uses completed game frames and recent wall time (~60 fps = 1×). "
    + "Neural timesteps, frames per decision, button pulses, and learning rules do not change.");
}

async function changeSpeed() {
  if (!live || !state?.speed_control || speedPending) return;
  const multiplier = Number($("speed-target").value);
  if (!Number.isFinite(multiplier)) return;
  const hz = multiplier * 60 / state.emulator_frames_per_sample;
  const epoch = connectionEpoch;
  speedPending = true;
  $("speed-target").disabled = true;
  setLabel("speed-status", "Setting speed…");
  try {
    const response = await fetch("/speed", {
      method: "POST", headers: {"Content-Type": "application/json", "X-Pokefly-Token": state.control_token},
      body: JSON.stringify({hz}),
    });
    const result = await response.json();
    if (epoch !== connectionEpoch || !live) return;
    if (!response.ok) throw new Error(result.error || "Speed rejected");
    showPacing(result.pacing, true);
  } catch (error) {
    if (epoch === connectionEpoch && live) {
      showPacing({target_hz: speedHz, revision: speedRevision}, true);
      setLabel("speed-status", "Speed unchanged", error.message);
    }
  } finally {
    speedPending = false;
    $("speed-target").disabled = !live || !state?.speed_control;
  }
}

function prepareBrain() {
  base = document.createElement("canvas");
  glow = document.createElement("canvas");
  base.width = glow.width = brain.width;
  base.height = glow.height = brain.height;
  const context = base.getContext("2d");
  context.fillStyle = "rgba(155,180,176,0.14)";
  for (let i = 0; i < state.x.length; i++) context.fillRect(px(i), py(i), 1.3, 1.3);
  context.fillStyle = "rgba(121,203,208,0.3)";
  for (const i of state.groups.photoreceptors || []) context.fillRect(px(i), py(i), 1.7, 1.7);
  for (const [key, label] of Object.entries(signals)) {
    const row = document.createElement("div"); row.className = "signal";
    const name = document.createElement("span"); name.textContent = label;
    const count = document.createElement("b"); count.id = `signal-${key}`; count.textContent = "—";
    count.title = "Spike count in the displayed sample";
    row.append(name, count); $("signals").append(row);
  }
}

function drawBrain(packet) {
  const graded = atob(packet.graded_u8 || "");
  if (graded.length !== (state.graded_indices || []).length) {
    throw new Error("Graded activity does not match this brain map. Reload the page.");
  }
  const context = glow.getContext("2d");
  context.globalCompositeOperation = "destination-out";
  context.fillStyle = "rgba(0,0,0,0.5)";
  context.fillRect(0, 0, brain.width, brain.height);
  context.globalCompositeOperation = "source-over";
  context.fillStyle = "#efbb68";
  for (const i of packet.spikes) context.fillRect(px(i) - 1, py(i) - 1, 2, 2);
  bctx.fillStyle = "#101619"; bctx.fillRect(0, 0, brain.width, brain.height);
  bctx.drawImage(base, 0, 0); bctx.drawImage(glow, 0, 0);
  for (let k = 0; k < graded.length; k++) {
    const i = state.graded_indices[k];
    bctx.fillStyle = `rgba(102,190,242,${graded.charCodeAt(k) / 255})`;
    bctx.fillRect(px(i), py(i), 1.8, 1.8);
  }
  const motorFired = new Set(packet.motor_fired);
  for (const key of Object.keys(signals).slice(6)) {
    for (const i of state.groups[key] || []) {
      const on = motorFired.has(i);
      bctx.beginPath(); bctx.arc(px(i), py(i), on ? 7 : 4, 0, Math.PI * 2);
      bctx.fillStyle = on ? "#8dd7b1" : "rgba(141,215,177,0.15)"; bctx.fill();
      bctx.lineWidth = 1; bctx.strokeStyle = "#8dd7b1"; bctx.stroke();
    }
  }
}

function drawRetina(packet) {
  const drive = atob(packet.drive_u8);
  rctx.fillStyle = "#182026"; rctx.fillRect(0, 0, retina.width, retina.height);
  // Only supported coordinates are plotted. Background gaps are not fabricated input.
  const map = state.retina;
  for (let k = 0; k < map.indices.length; k++) {
    const value = drive.charCodeAt(map.indices[k]);
    rctx.fillStyle = `rgb(${value},${value},${value})`;
    rctx.fillRect(map.u[k] * (retina.width - 4), map.v[k] * (retina.height - 4), 4, 4);
  }
}

async function draw(packet) {
  if (packet.run_id !== state.run_id) {
    connection(false, "RUN CHANGED / RELOAD");
    setLabel("control-message", "New run · reload this page", "A new experiment is running. Reload to load its input map and control token.");
    return;
  }
  lastReceived = Date.now();
  const epoch = connectionEpoch;
  const frame = new Image(); frame.src = packet.screen;
  await frame.decode();
  if (epoch !== connectionEpoch) return; // A pending frame cannot undo a disconnect/error.
  if (packet.sample <= drawnSample) return; // A slow older image cannot replace a newer sample.
  drawnSample = packet.sample;
  connection(true, "LIVE");
  gctx.imageSmoothingEnabled = false; gctx.drawImage(frame, 0, 0);
  drawRetina(packet); drawBrain(packet);
  $("sample").textContent = `Sample ${packet.sample.toLocaleString()}`;
  $("total-spikes").textContent = packet.spikes_total.toLocaleString();
  $("brain-time").textContent = packet.brain_seconds.toFixed(1);
  $("compute").textContent = packet.compute_ms.toFixed(1);
  showPacing(packet.pacing);
  if (Number.isFinite(packet.pacing?.actual_game_speed)) {
    setLabel("speed-actual", `Actual ${packet.pacing.actual_game_speed.toFixed(1)}×`,
      `${packet.pacing.decisions_per_second.toFixed(1)} decisions/s over recent wall time. `
      + "Includes pacing and previous checkpoint/display overhead; not browser playback speed.");
  }
  setLabel("action-source", packet.action_source === "fly"
    ? (packet.action === "wait" ? "Fly chose Wait" : "Fly pulse · released")
    : packet.action_source === "human" ? "Human pulse · released" : "Idle · released",
    state.button_timing === "serial-v2"
      ? "Amber marks selected buttons delivered in sequence: direction, release, function, release. Same total frame budget; no game-state routing."
      : "Amber marks the buttons delivered together after this input frame. Every pulse releases all buttons.");
  const delivered = new Set(packet.buttons ?? (packet.action === "wait" ? [] : packet.action.split("+")));
  for (const button of buttons) button.classList.toggle("delivered", delivered.has(button.dataset.action));
  const history = packet.recent_actions || [];
  const actionLabel = (item) => `#${item.sample} ${item.action.toUpperCase()}`;
  setLabel("history", history.length ? history.slice(-3).map(actionLabel).join(" · ") : "No pulses yet",
    history.length ? history.map(actionLabel).join(" · ") : "No button pulses delivered yet");
  for (const key of Object.keys(signals)) {
    const count = $("signal-" + key);
    count.textContent = (packet.groups[key] ?? 0).toLocaleString();
    count.title = "Spike count in the displayed sample";
    count.classList.toggle("graded", false);
  }
  for (const [key, value] of Object.entries(packet.graded_groups || {})) {
    if (!(key in signals)) continue;
    setLabel("signal-" + key, `${value.toFixed(3)} release`,
      "Mean normalized graded release [0,1] at the last neural step; not a spike count");
    $("signal-" + key).classList.toggle("graded", true);
  }
  if (packet.learning) {
    const learning = packet.learning;
    const hasTotal = typeof packet.reward_total === "number" && Number.isFinite(packet.reward_total);
    $("reward-label").textContent = hasTotal ? "TOTAL REWARD" : "STEP REWARD";
    setLabel("reward", Number(hasTotal ? packet.reward_total : packet.reward).toFixed(3),
      (hasTotal ? "Cumulative measured reward for this game attempt, including its resumed history. "
        : "This older running server supplies only the current decision's reward. ")
      + `This decision: ${Number(packet.reward).toFixed(3)}; delivered to the fly: `
      + `${Number(packet.delivered_reward ?? packet.reward).toFixed(3)}. `
      + "Measured reward can increase with plasticity disabled; that does not establish learning.");
    $("dopamine").textContent = learning.dopamine_surrogate.toFixed(3);
    $("dopamine").title = learning.prediction_error
      ? "External reward transformed by tanh. Internal compartment prediction errors: "
        + learning.prediction_error.map(v => Number(v).toFixed(4)).join(", ")
        + ". Reward expectation is a model state, not a new game penalty."
      : "Synthetic reward modulator; not measured dopamine.";
    $("plastic-changes").textContent = learning.changed_edges.toLocaleString();
    const events = packet.reward_events.length
      ? packet.reward_events.map(e => e.category.replaceAll("_", " ")).join(", ") : "no reward event";
    setLabel("learning-detail", `${learning.changed_this_reward.toLocaleString()} changed this sample · ${events}`,
      `Maximum relative weight change: ${(learning.max_relative_change * 100).toFixed(3)}%. `
      + "Changed synapses are measured, not evidence of learned gameplay. Reward: " + events + ".");
  }
  const info = packet.telemetry;
  setLabel("measurement", `Map ${info.map} · tile (${info.x}, ${info.y}) · ${packet.emulator_frames.toLocaleString()} GB frames`);
  setLabel("brain-detail", `${packet.spikes.length.toLocaleString()} / ${packet.located_firing.toLocaleString()} firing cells drawn`,
    `${state.missing_positions.toLocaleString()} model cells lack anatomical positions. Population counts include all cells. `
    + "Amber spike glow persists briefly; blue is graded release [0,1] at the last neural step. No synapse activity is inferred.");
}

async function control(action) {
  if (!live || !state?.manual) return;
  try {
    const response = await fetch("/control", {
      method: "POST", headers: {"Content-Type": "application/json", "X-Pokefly-Token": state.control_token},
      body: JSON.stringify({action}),
    });
    const result = await response.json();
    setLabel("control-message", response.ok
      ? `Queued ${action.toUpperCase()} · awaiting delivery`
      : `Control rejected: ${result.error}`);
  } catch {
    setLabel("control-message", "Pulse failed · check local process");
  }
}

async function init() {
  const response = await fetch("/static.json");
  if (!response.ok) throw new Error("Model metadata could not be loaded");
  state = await response.json();
  $("speed-controls").hidden = !state.speed_control;
  showPacing(state.pacing);
  $("speed-target").addEventListener("change", changeSpeed);
  prepareBrain();
  $("mode").textContent = state.autonomous ? `AUTONOMOUS / ${state.mode.toUpperCase()}`
    : state.manual ? "MANUAL DIAGNOSTIC" : "OBSERVER / WAIT ONLY";
  setLabel("mode-notice", "EXPERIMENTAL", state.autonomous
    ? "Raw pixels → neural dynamics → fixed button mapping. Synthetic reinforcement; learned gameplay and biological fidelity are not established."
    : "Diagnostic mode: screen pixels drive the brain, but the brain does not choose buttons. Weights are frozen.");
  setLabel("screen-timing", "Post-action frame", "Post-action frame supplied to the diagnostic brain.");
  if (state.autonomous) {
    $("learning-mode").textContent = state.mode === "no-reward" ? "NO REINFORCEMENT"
      : state.internal_learning ? "INTERNAL PLASTICITY ON" : "PLASTICITY OFF";
    setLabel("control-message", "Human controls off · amber = delivered pulse");
    setLabel("screen-timing", "Pre-action frame", "Frame supplied to the brain; the displayed button pulse followed this frame.");
    if (state.visual_timing && state.visual_timing !== "snapshot-v1") {
      setLabel("screen-timing", "Input window · last frame", "Last raw frame of the preceding neural input window. Brain activity spans that window; the displayed button pulse followed it. Timing: " + state.visual_timing);
    }
    $("ram-boundary").textContent = "RAM: rewards + telemetry";
    setLabel("motor-detail", "Fixed readouts · MN9 → A",
      "Seven fixed neural readouts; MN9 / CB0701 proboscis extension maps to A. No physical fly body is simulated; motor function remains unvalidated.");
    $("learning-detail").textContent = "Waiting for synaptic measurements";
    if (state.motor_arbitration === "sustained-v3") {
      $("motor-detail").title += " Sustained directional readout: decaying past motor spikes can maintain a direction during a silent sample. Functions require current spikes. Button pulses are not additional neural spikes.";
    }
  }
  $("brain-size").textContent = `${state.neurons.toLocaleString()} neurons · ${Math.round(state.neural_seconds_per_sample * 1000)} ms/sample`;
  $("synapses").textContent = `${state.connections.toLocaleString()} connections · `
    + (state.autonomous ? `${state.plastic_edges.toLocaleString()} plastic-eligible · ` : "frozen weights · ")
    + `${state.dynamics_profile || "baseline"} · 2D anatomy`;
  if (state.sensory_isolation?.enabled) {
    $("synapses").textContent += " · sensory isolated";
    $("synapses").title = `${state.sensory_isolation.neurons.toLocaleString()} nonvisual sensory cells: `
      + (state.sensory_isolation.mode === "nonvisual-quiescent-v1"
        ? "incoming current isolated and unsupported sensory cells held at rest. Experimental quiescent boundary. "
        : "incoming network current isolated; outgoing signals, tonic input and neural noise remain. ")
      + "Connection count is the original anatomical graph. Experimental setting, not a validated intact fly.";
  }
  if (state.intrinsic_calibration) {
    $("synapses").textContent += " · calibrated";
    $("synapses").title = ($("synapses").title || "")
      + " Frozen neural excitability calibrated on a neutral image, without game rewards or button labels. Experimental model; not measured physiology.";
  }
  if (state.visual_calibration) {
    $("synapses").textContent += " · visual-rate";
    $("synapses").title = ($("synapses").title || "")
      + ` ${state.visual_calibration.neurons.toLocaleString()} visual neurons use frozen calibrated rate dynamics. `
      + "Blue is normalized continuous activity, not spikes. Same raw retina and fixed motor-to-button mapping; no external visual policy.";
  }
  const map = state.retina;
  setLabel("mapping", `${(map.direct + map.inferred).toLocaleString()} / ${map.photoreceptors.toLocaleString()} receptors mapped`,
    `${map.direct.toLocaleString()} annotated, ${map.inferred.toLocaleString()} inferred; ${map.unplaced_zero_driven} unplaced receptors receive zero input. `
    + "RGB → grayscale → bilinear sample. Approximate split-eye chart, not measured fly optics.");
  if (state.manual) setLabel("control-message", "Human controls on · arrows / A / S / Enter",
    "Click or use arrow keys, A, S (B), and Enter (Start). Amber confirms a delivered pulse, not a held button.");
  for (const button of buttons) button.addEventListener("click", () => control(button.dataset.action));
  const keys = {ArrowUp:"up", ArrowDown:"down", ArrowLeft:"left", ArrowRight:"right", a:"a", s:"b", Enter:"start"};
  document.addEventListener("keydown", (event) => {
    if (!live || !state.manual || event.repeat || event.ctrlKey || event.altKey || event.metaKey) return;
    // Preserve Enter/Space activation of a focused button for accessibility.
    if (event.target instanceof HTMLButtonElement && ["Enter", " "].includes(event.key)) return;
    const action = keys[event.key] || keys[event.key.toLowerCase()];
    if (action) { event.preventDefault(); control(action); }
  });
  stream = new EventSource("/events");
  stream.onmessage = async (event) => {
    try { await draw(JSON.parse(event.data)); }
    catch (error) {
      connection(false, "DISPLAY ERROR"); setLabel("control-message", error.message);
    }
  };
  stream.onerror = () => connection(false, "DISCONNECTED");
  setInterval(() => {
    if (lastReceived && Date.now() - lastReceived > 3000) connection(false, "STALE / RECONNECTING");
  }, 1000);
}
window.addEventListener("pagehide", () => stream?.close());
init().catch((error) => { connection(false, "UNAVAILABLE"); setLabel("control-message", error.message); });
