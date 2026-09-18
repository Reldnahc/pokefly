// Unit tests of the shipped UI script with DOM/canvas stubs, NOT browser/layout verification.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const assets = path.join(__dirname, "../src/pokefly/assets");
const source = fs.readFileSync(path.join(assets, "dashboard.js"), "utf8");
const html = fs.readFileSync(path.join(assets, "dashboard.html"), "utf8");
const populationNames = [
  "photoreceptors", "lamina_L1_L5", "visual_projection", "Kenyon_cells", "MBONs", "descending",
  "forward_Up", "backward_Down", "steer_Left", "steer_Right", "proboscis_A", "escape_B", "song_Start",
];
const settle = () => new Promise((resolve) => setImmediate(resolve));

async function harness({manual = true, autonomous = false, mode = "learn", graded = false, isolated = false,
                        speedControl = autonomous, speedError = false, hz = 5,
                        visualTiming = "snapshot-v1", quiescent = false, intrinsic = false} = {}) {
  const elements = new Map(), requests = [], intervals = [], pendingImages = new Map();
  let stream, now = 10000, speedDelay = null;
  let pacing = {target_hz: hz, revision: 0};
  class Element {
    constructor(tag) {
      this.tag = tag; this.textContent = ""; this.children = []; this.listeners = {};
      this.dataset = {}; this.disabled = true; this.width = 900; this.height = 790;
      const classes = new Set();
      this.classList = {
        toggle: (key, on) => on ? classes.add(key) : classes.delete(key),
        contains: (key) => classes.has(key),
      };
      this.context = {calls: []};
      for (const method of ["fillRect", "drawImage", "beginPath", "arc", "fill", "stroke"]) {
        this.context[method] = (...args) => {
          this.context.calls.push({method, args, fill: this.context.fillStyle});
        };
      }
    }
    set id(value) { this._id = value; elements.set(value, this); }
    get id() { return this._id; }
    getContext() { return this.context; }
    append(...children) { this.children.push(...children); }
    addEventListener(type, listener) { this.listeners[type] = listener; }
  }
  class Button extends Element {}
  for (const match of html.matchAll(/id="([^"]+)"/g)) {
    new Element("element").id = match[1];
  }
  const buttons = ["up", "down", "left", "right", "a", "b", "start"].map((action) => {
    const button = new Button("button"); button.dataset.action = action; return button;
  });
  const metadata = {
    manual, control_token: "test-token", run_id: "test-run", neural_seconds_per_sample: 0.2,
    speed_control: speedControl, pacing, emulator_frames_per_sample: 24,
    autonomous, mode, internal_learning: mode === "learn", plastic_edges: 2,
    neurons: 4, connections: 5, missing_positions: 1, x: [100, 200, 300], y: [200, 300, 400],
    groups: Object.fromEntries(populationNames.map((key) => [key, []])),
    graded_indices: graded ? [0, 2] : [], dynamics_profile: graded ? "hybrid-v1" : "baseline",
    visual_timing: visualTiming,
    intrinsic_calibration: intrinsic ? {rule: "uniform-neutral-rate-homeostasis-v1"} : null,
    sensory_isolation: {enabled: isolated, neurons: isolated ? 2 : 0,
                        mode: quiescent ? "nonvisual-quiescent-v1" : "nonvisual-incoming-v1"},
    retina: {indices: [0, 1], u: [0, 1], v: [0, 1], direct: 1, inferred: 1,
             photoreceptors: 3, unplaced_zero_driven: 1},
  };
  metadata.groups.forward_Up = [0, 1];
  const document = {
    listeners: {},
    getElementById: (id) => {
      assert.ok(elements.has(id), `HTML is missing #${id}`); return elements.get(id);
    },
    querySelectorAll: (selector) => {
      assert.equal(selector, "[data-action]"); return buttons;
    },
    createElement: (tag) => new Element(tag),
    addEventListener(type, listener) { this.listeners[type] = listener; },
  };
  const window = {listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; }};
  vm.runInNewContext(source, {
    document, window, HTMLButtonElement: Button,
    Image: class { decode() { return pendingImages.get(this.src) || Promise.resolve(); } },
    EventSource: class {
      constructor(url) { assert.equal(url, "/events"); stream = this; }
      close() { this.closed = true; }
    },
    fetch: async (url, options) => {
      requests.push({url, options});
      if (url === "/static.json") return {ok: true, json: async () => metadata};
      if (url === "/speed") {
        if (speedDelay) await speedDelay;
        if (speedError) return {ok: false, json: async () => ({error: "test rejection"})};
        pacing = {target_hz: JSON.parse(options.body).hz, revision: pacing.revision + 1};
        return {ok: true, json: async () => ({pacing})};
      }
      assert.equal(url, "/control");
      return {ok: true, json: async () => ({queued: JSON.parse(options.body).action})};
    },
    atob: (value) => Buffer.from(value, "base64").toString("binary"),
    setInterval: (fn) => intervals.push(fn), Date: {now: () => now},
  }, {filename: "dashboard.js"});
  await settle();
  assert.ok(stream, "UI must connect to the live stream after loading metadata");
  const packet = (sample = 1) => ({
    run_id: "test-run", sample, screen: `data:image/png;base64,sample${sample}`,
    spikes: [0, 1], motor_fired: [1], located_firing: 2, unique_firing: 3,
    drive_u8: Buffer.from([0, 255, 0]).toString("base64"),
    spikes_total: 8, brain_seconds: sample * 0.2, compute_ms: 4,
    emulator_frames: sample * 12, action: "wait", action_source: "idle",
    groups: Object.fromEntries(populationNames.map((key) => [key, 1])),
    recent_actions: [], telemetry: {map: 38, x: 3, y: 6},
    ...(speedControl ? {pacing: {...pacing, actual_game_speed: 1.7, decisions_per_second: 4.25}} : {}),
    graded_u8: graded ? Buffer.from([0, 255]).toString("base64") : "",
    graded_groups: graded ? {photoreceptors: 0.5, lamina_L1_L5: 0.25} : {},
  });
  return {
    elements, buttons, requests, intervals, document, window, stream, pendingImages, packet,
    send: (data) => stream.onmessage({data: JSON.stringify(data)}),
    advance: (ms) => { now += ms; },
    delaySpeed: (promise) => { speedDelay = promise; },
  };
}

test("one sample drives screen, retina, counts, and exact motor-cell highlights", async () => {
  const ui = await harness();
  const data = ui.packet();
  await ui.send(data);
  assert.equal(ui.elements.get("connection").textContent, "LIVE");
  assert.equal(ui.elements.get("sample").textContent, "Sample 1");
  assert.match(ui.elements.get("measurement").textContent, /Map 38/);
  assert.match(ui.elements.get("brain-size").textContent, /200 ms\/sample/);
  assert.equal(ui.elements.get("total-spikes").textContent, "8");
  const draw = ui.elements.get("game").context.calls.find((call) => call.method === "drawImage");
  assert.equal(draw.args[0].src, data.screen);
  const retinalRects = ui.elements.get("retina").context.calls.filter((c) => c.method === "fillRect");
  assert.equal(retinalRects.length, 3); // background + two supported receptor locations
  assert.equal(retinalRects[1].fill, "rgb(0,0,0)");
  assert.equal(retinalRects[2].fill, "rgb(255,255,255)");
  const motorFills = ui.elements.get("brain").context.calls.filter((c) => c.method === "fill");
  assert.equal(motorFills[0].fill, "rgba(141,215,177,0.15)");
  assert.equal(motorFills[1].fill, "#8dd7b1");
});

test("manual queue acknowledgement is distinct from actual delivery", async () => {
  const ui = await harness();
  assert.ok(ui.buttons.every((button) => button.disabled));
  await ui.send(ui.packet());
  const a = ui.buttons.find((button) => button.dataset.action === "a");
  assert.equal(a.disabled, false);
  await a.listeners.click();
  const request = ui.requests.at(-1);
  assert.equal(request.options.headers["X-Pokefly-Token"], "test-token");
  assert.deepEqual(JSON.parse(request.options.body), {action: "a"});
  assert.equal(a.classList.contains("delivered"), false);
  await ui.send({...ui.packet(2), action: "a", action_source: "human",
                 recent_actions: [{sample: 2, action: "a"}]});
  assert.equal(a.classList.contains("delivered"), true);
  assert.match(ui.elements.get("history").textContent, /#2 A/);
});

test("observer, disconnected, and stale displays cannot send controls", async () => {
  const observer = await harness({manual: false});
  await observer.send(observer.packet());
  await observer.buttons[0].listeners.click();
  assert.ok(observer.buttons.every((button) => button.disabled));
  assert.equal(observer.requests.length, 1);
  const ui = await harness();
  await ui.send(ui.packet());
  ui.stream.onerror();
  await ui.buttons[0].listeners.click();
  assert.equal(ui.requests.length, 1);
  await ui.send(ui.packet(2));
  ui.advance(4000); ui.intervals[0]();
  assert.ok(ui.buttons.every((button) => button.disabled));
  assert.match(ui.elements.get("connection").textContent, /STALE/);
});

test("a slow older image cannot replace a newer synchronized sample", async () => {
  const ui = await harness();
  let release;
  const old = ui.packet(1);
  ui.pendingImages.set(old.screen, new Promise((resolve) => { release = resolve; }));
  const firstDraw = ui.send(old);
  await ui.send(ui.packet(2));
  release(); await firstDraw;
  assert.equal(ui.elements.get("sample").textContent, "Sample 2");
});

test("a restarted run cannot reuse an old input map or control token", async () => {
  const ui = await harness();
  await ui.send(ui.packet(10));
  await ui.send({...ui.packet(1), run_id: "different-run"});
  assert.equal(ui.elements.get("connection").textContent, "RUN CHANGED / RELOAD");
  assert.ok(ui.buttons.every((button) => button.disabled));
  ui.window.listeners.pagehide();
  assert.equal(ui.stream.closed, true);
});

test("an image pending at disconnect cannot re-enable manual controls", async () => {
  const ui = await harness();
  let release;
  const data = ui.packet();
  ui.pendingImages.set(data.screen, new Promise((resolve) => { release = resolve; }));
  const pending = ui.send(data);
  ui.stream.onerror();
  release(); await pending;
  assert.equal(ui.elements.get("connection").textContent, "DISCONNECTED");
  assert.ok(ui.buttons.every((button) => button.disabled));
});

test("malformed stream data is reported and disables input", async () => {
  const ui = await harness();
  await ui.send(ui.packet());
  await ui.stream.onmessage({data: "{"});
  assert.equal(ui.elements.get("connection").textContent, "DISPLAY ERROR");
  assert.ok(ui.buttons.every((button) => button.disabled));
});

test("autonomous mode shows neural decisions and real plasticity without human controls", async () => {
  const ui = await harness({manual: false, autonomous: true});
  await ui.send({...ui.packet(), action: "a", action_source: "fly", reward: 0.05,
    reward_events: [{category: "new_tile"}],
    learning: {dopamine_surrogate: 0.05, changed_edges: 2, changed_this_reward: 1,
               max_relative_change: 0.001}});
  assert.match(ui.elements.get("mode").textContent, /AUTONOMOUS/);
  assert.equal(ui.elements.get("learning-mode").textContent, "INTERNAL PLASTICITY ON");
  assert.equal(ui.elements.get("reward").textContent, "0.050");
  assert.equal(ui.elements.get("plastic-changes").textContent, "2");
  assert.match(ui.elements.get("action-source").textContent, /Fly pulse/);
  assert.match(ui.elements.get("screen-timing").textContent, /Pre-action/);
  assert.ok(ui.buttons.every(button => button.disabled));
  assert.ok(ui.buttons.find(button => button.dataset.action === "a").classList.contains("delivered"));
});

test("frozen control is labeled plasticity off", async () => {
  const ui = await harness({manual: false, autonomous: true, mode: "frozen"});
  assert.equal(ui.elements.get("learning-mode").textContent, "PLASTICITY OFF");
});

test("temporal input is labeled as a window, not a single causal frame", async () => {
  const ui = await harness({manual: false, autonomous: true, visualTiming: "stream-v1"});
  assert.match(ui.elements.get("screen-timing").textContent, /Input window/);
  assert.match(ui.elements.get("screen-timing").title, /activity spans that window/);
});

test("quiescent boundary and internal prediction errors are labeled honestly", async () => {
  const ui = await harness({manual: false, autonomous: true, isolated: true, quiescent: true});
  assert.match(ui.elements.get("synapses").title, /held at rest/);
  assert.doesNotMatch(ui.elements.get("synapses").title, /noise remain/);
  await ui.send({...ui.packet(), reward: 0, reward_events: [],
    learning: {dopamine_surrogate: 0, prediction_error: [-0.01, -0.01, 0], changed_edges: 2,
               changed_this_reward: 1, max_relative_change: 0.1}});
  assert.match(ui.elements.get("dopamine").title, /prediction errors/);
  assert.match(ui.elements.get("dopamine").title, /not a new game penalty/);
});

test("live speed controls change wall pacing without enabling autonomous game buttons", async () => {
  const ui = await harness({manual: false, autonomous: true});
  const select = ui.elements.get("speed-target");
  assert.equal(select.disabled, true);
  await ui.send(ui.packet());
  assert.equal(select.disabled, false);
  assert.equal(select.value, "2");
  assert.equal(ui.elements.get("speed-actual").textContent, "Actual 1.7×");
  const oldPacket = ui.packet(2);
  select.value = "100";
  await select.listeners.change();
  const sent = ui.requests.at(-1);
  assert.equal(sent.url, "/speed");
  assert.equal(sent.options.headers["X-Pokefly-Token"], "test-token");
  assert.deepEqual(JSON.parse(sent.options.body), {hz: 250});
  assert.equal(ui.elements.get("speed-status").textContent, "Target 100.0×");
  assert.equal(ui.elements.get("speed-actual").textContent, "Actual 1.7×");
  await ui.send(oldPacket);
  assert.equal(select.value, "100"); // In-flight old SSE cannot undo the newer acknowledgement.
  select.value = "0";
  await select.listeners.change();
  assert.equal(ui.elements.get("speed-status").textContent, "Uncapped");
  assert.ok(ui.buttons.every(button => button.disabled));
});

test("speed controls are disabled on disconnect, stale data, and unsupported servers", async () => {
  const ui = await harness({manual: false, autonomous: true});
  await ui.send(ui.packet());
  ui.stream.onerror();
  await ui.elements.get("speed-target").listeners.change();
  assert.equal(ui.requests.length, 1);
  assert.equal(ui.elements.get("speed-target").disabled, true);
  assert.equal(ui.elements.get("speed-actual").textContent, "Actual —");
  await ui.send(ui.packet(2));
  ui.advance(4000); ui.intervals[0]();
  assert.equal(ui.elements.get("speed-target").disabled, true);
  const old = await harness({manual: false, autonomous: true, speedControl: false});
  await old.send(old.packet());
  assert.equal(old.elements.get("speed-controls").hidden, true);
  assert.equal(old.elements.get("speed-target").disabled, true);
});

test("custom caps, server rejection, and pending acknowledgements are handled truthfully", async () => {
  const ui = await harness({manual: false, autonomous: true, hz: 7, speedError: true});
  await ui.send(ui.packet());
  const select = ui.elements.get("speed-target");
  assert.equal(select.value, "custom");
  assert.equal(ui.elements.get("speed-custom").textContent, "2.80× target");
  select.value = "0";
  await select.listeners.change();
  assert.equal(select.value, "custom");
  assert.equal(ui.elements.get("speed-status").textContent, "Speed unchanged");
  const pending = await harness({manual: false, autonomous: true});
  await pending.send(pending.packet());
  let release;
  pending.delaySpeed(new Promise(resolve => { release = resolve; }));
  pending.elements.get("speed-target").value = "0";
  const change = pending.elements.get("speed-target").listeners.change();
  pending.stream.onerror();
  release(); await change;
  assert.equal(pending.elements.get("speed-target").disabled, true);
  assert.equal(pending.elements.get("connection").textContent, "DISCONNECTED");
});

test("the sensory-isolated experiment is labeled without calling cells or connections deleted", async () => {
  const ui = await harness({manual: false, autonomous: true, graded: true, isolated: true});
  assert.match(ui.elements.get("synapses").textContent, /sensory isolated/);
  assert.match(ui.elements.get("synapses").title, /incoming network current isolated/);
  assert.match(ui.elements.get("synapses").title, /outgoing signals, tonic input and neural noise remain/);
});

test("intrinsic calibration is labeled once, not appended on every frame", async () => {
  const ui = await harness({manual: false, autonomous: true, graded: true, isolated: true, intrinsic: true});
  const label = ui.elements.get("synapses").textContent;
  const detail = ui.elements.get("synapses").title;
  assert.match(label, /calibrated/);
  assert.match(detail, /without game rewards or button labels/);
  for (let sample = 1; sample <= 3; sample++) await ui.send(ui.packet(sample));
  assert.equal(ui.elements.get("synapses").textContent, label);
  assert.equal(ui.elements.get("synapses").title, detail);
});

test("a combined neural pulse highlights every delivered button then clears on wait", async () => {
  const ui = await harness({manual: false, autonomous: true});
  await ui.send({...ui.packet(), action: "up+a", buttons: ["up", "a"], action_source: "fly",
                 recent_actions: [{sample: 1, action: "up+a"}]});
  assert.deepEqual(ui.buttons.filter(b => b.classList.contains("delivered"))
    .map(b => b.dataset.action), ["up", "a"]);
  assert.match(ui.elements.get("history").textContent, /#1 UP\+A/);
  assert.ok(ui.buttons.every(b => b.disabled));
  await ui.send({...ui.packet(2), buttons: []});
  assert.ok(ui.buttons.every(b => !b.classList.contains("delivered")));
});

test("graded release uses blue intensity and separate units without inflating spikes", async () => {
  const ui = await harness({manual: false, autonomous: true, graded: true});
  await ui.send({...ui.packet(), spikes: [1], spikes_total: 3});
  const releaseRects = ui.elements.get("brain").context.calls.filter((call) =>
    call.method === "fillRect" && call.args[2] === 1.8);
  assert.equal(releaseRects.length, 2);
  assert.equal(releaseRects[0].fill, "rgba(102,190,242,0)");
  assert.equal(releaseRects[1].fill, "rgba(102,190,242,1)");
  const receptors = ui.elements.get("signal-photoreceptors");
  assert.equal(receptors.textContent, "0.500 release");
  assert.ok(receptors.classList.contains("graded"));
  assert.match(receptors.title, /not a spike count/);
  assert.equal(ui.elements.get("signal-lamina_L1_L5").textContent, "0.250 release");
  assert.equal(ui.elements.get("signal-Kenyon_cells").textContent, "1");
  assert.equal(ui.elements.get("total-spikes").textContent, "3");
  assert.match(ui.elements.get("synapses").textContent, /hybrid-v1/);
});

test("a graded packet for a different neural map cannot enable controls", async () => {
  const ui = await harness({graded: true});
  await ui.send({...ui.packet(), graded_u8: Buffer.from([255]).toString("base64")});
  assert.equal(ui.elements.get("connection").textContent, "DISPLAY ERROR");
  assert.match(ui.elements.get("control-message").textContent, /does not match/);
  assert.ok(ui.buttons.every((button) => button.disabled));
});

test("compact showcase keeps experiment caveats and full history available in hover details", async () => {
  const ui = await harness({manual: false, autonomous: true});
  const recent = Array.from({length: 8}, (_, index) => ({sample: index + 1, action: "a"}));
  await ui.send({...ui.packet(8), recent_actions: recent});
  assert.equal(ui.elements.get("mode-notice").textContent, "EXPERIMENTAL");
  assert.match(ui.elements.get("mode-notice").title, /not established/);
  assert.match(ui.elements.get("mapping").title, /not measured fly optics/);
  assert.equal(ui.elements.get("history").textContent, "#6 A · #7 A · #8 A");
  assert.match(ui.elements.get("history").title, /#1 A/);
  assert.match(ui.elements.get("brain-detail").title, /No synapse activity is inferred/);
});

test("desktop stylesheet declares a viewport-bounded grid and preserves canvas proportions", () => {
  // A source contract, not a claim of browser-rendered visual/layout verification.
  assert.match(html, /html,body\s*\{[^}]*overflow:hidden/);
  assert.match(html, /height:100dvh/);
  assert.match(html, /grid-template-rows:auto minmax\(0,1fr\) auto/);
  assert.match(html, /\.screenbox canvas\s*\{[^}]*object-fit:contain/);
  assert.match(html, /#brain\s*\{[^}]*object-fit:contain/);
  assert.doesNotMatch(html, /@media\s*\(max-width/);
  assert.match(html, /SYNTHETIC D/);
});
