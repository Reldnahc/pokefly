"""Loopback-only viewer; anatomical plotting adapts fly.ai (see third-party notices).

No filesystem-serving route, ROM download, external assets, or policy lives here.
Optional human button pulses are bounded and separate from neural output.
"""

from __future__ import annotations

import base64
import io
import json
import queue
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

import numpy as np
from PIL import Image

from pokefly.emulator import ACTIONS
from pokefly.pacing import RuntimePacing


def png_data(frame: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(frame[:, :, :3]).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class BrainView:
    """Project actual anatomical positions, never invented neuron locations."""

    def __init__(self, controller):
        brain = controller.brain
        if brain.positions is None:
            raise ValueError("Brain viewer requires anatomical position metadata")
        valid = np.isfinite(brain.positions).all(axis=1)
        shown = np.flatnonzero(valid)
        xy = brain.positions[shown][:, [0, 2]].astype(float)
        if not len(xy):
            raise ValueError("Brain has no usable anatomical positions")
        low, high = np.percentile(xy, [0.2, 99.8], axis=0)
        xy = np.clip((xy - low) / np.maximum(high - low, 1), 0, 1)
        self.lookup = np.full(brain.n, -1, np.int32)
        self.lookup[shown] = np.arange(len(shown))
        motor_groups = [
            idx
            for name, idx in controller.groups.items()
            if name.endswith(("_Up", "_Down", "_Left", "_Right", "_A", "_B", "_Start"))
        ]
        self.motor_neurons = (
            np.unique(np.concatenate(motor_groups)) if motor_groups else np.empty(0, dtype=int)
        )
        self.rng = np.random.default_rng(2026)  # Viewer-only RNG, never the brain's.
        retina = controller.retina
        hybrid = getattr(controller, "hybrid", None)
        self.graded_neurons = hybrid.graded_host if hybrid else np.empty(0, np.int64)
        self.graded_located = self.lookup[self.graded_neurons] >= 0
        self.graded_masks = {
            name: np.isin(self.graded_neurons, controller.groups.get(name, []))
            for name in ("photoreceptors", "lamina_L1_L5")
        }
        self.static = {
            "neurons": brain.n,
            "connections": len(brain.weights),
            "positioned_neurons": len(shown),
            "missing_positions": int((~valid).sum()),
            "x": np.round(xy[:, 0] * 1000).astype(int).tolist(),
            "y": np.round(xy[:, 1] * 1000).astype(int).tolist(),
            "groups": {
                name: self.lookup[idx][self.lookup[idx] >= 0].tolist()
                for name, idx in controller.groups.items()
            },
            "population_sizes": {name: len(idx) for name, idx in controller.groups.items()},
            "retina": {
                "indices": np.flatnonzero(retina.mapped).tolist(),
                "u": np.round(retina.u[retina.mapped], 5).tolist(),
                "v": np.round(retina.v[retina.mapped], 5).tolist(),
                **retina.summary(),
            },
            "spike_display_limit": 6000,
            "internal_learning": False,
            "motor_mapping_validated": False,
            "dynamics_profile": controller.config.dynamics.profile
            if hasattr(controller, "config")
            else "baseline",
            "graded_indices": self.lookup[self.graded_neurons][self.graded_located].tolist(),
            "a_population": "MN9 / CB0701 (MaleCNS 10331, 16949); model function unvalidated",
            "sensory_isolation": getattr(controller, "sensory_isolation", {"enabled": False}),
            "intrinsic_calibration": getattr(controller, "intrinsic_calibration_info", None),
        }

    def activity(self, observation) -> dict:
        located = self.lookup[observation.fired]
        located = located[located >= 0]
        total_located = len(located)
        if total_located > 6000:
            located = self.rng.choice(located, 6000, replace=False)
        motor_fired = self.lookup[self.motor_neurons[observation.counts[self.motor_neurons] > 0]]
        continuous = observation.continuous
        values = np.asarray(continuous["graded_values"]) if continuous else np.empty(0)
        graded_groups = {}
        if continuous and len(values):
            for name, selected in self.graded_masks.items():
                graded_groups[name] = float(values[selected].mean()) if selected.any() else 0.0
        return {
            "spikes": located.tolist(),
            "motor_fired": motor_fired[motor_fired >= 0].tolist(),
            "unique_firing": len(observation.fired),
            "located_firing": total_located,
            "spikes_total": observation.spikes,
            "groups": observation.groups,
            "graded_u8": base64.b64encode(
                np.rint(values[self.graded_located] * 255).astype(np.uint8).tobytes()
            ).decode("ascii")
            if continuous
            else "",
            "graded_groups": graded_groups,
            "graded_units": "mean normalized release state [0,1], last neural step; NOT spikes",
            # 8-bit display copy only; the brain receives the full float32 drive.
            "drive_u8": base64.b64encode(
                np.rint(observation.drive * 255).astype(np.uint8).tobytes()
            ).decode("ascii"),
        }


class Dashboard:
    def __init__(
        self,
        static: dict,
        *,
        port: int = 8777,
        manual: bool = False,
        pacing: RuntimePacing | None = None,
    ):
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self.manual = manual
        self.pacing = pacing
        self.token = secrets.token_urlsafe(24)
        self.static = {
            **static,
            "manual": manual,
            "control_token": self.token,
            "speed_control": pacing is not None,
        }
        self.commands: queue.Queue[str] = queue.Queue(maxsize=16)
        self.condition = threading.Condition()
        self.latest: bytes | None = None
        self.sequence = 0
        self.closed = False
        self.stream_slots = threading.BoundedSemaphore(8)
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def allowed_host(self):
                return self.headers.get("Host") in dashboard.hosts

            def send_headers(self, status, mime, length=None):
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                    "script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
                )
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.end_headers()

            def reply(self, status, payload, mime="application/json"):
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_headers(status, mime, len(body))
                self.wfile.write(body)

            def do_GET(self):
                if not self.allowed_host():
                    self.reply(403, {"error": "loopback Host required"})
                    return
                assets = {
                    "/": ("dashboard.html", "text/html; charset=utf-8"),
                    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
                }
                if self.path in assets:
                    asset, mime = assets[self.path]
                    self.reply(200, files("pokefly").joinpath("assets", asset).read_bytes(), mime)
                elif self.path == "/static.json":
                    pacing_state = dashboard.pacing.state() if dashboard.pacing else None
                    self.reply(
                        200,
                        {
                            **dashboard.static,
                            **(
                                {"pacing": pacing_state, "target_hz": pacing_state["target_hz"]}
                                if pacing_state
                                else {}
                            ),
                        },
                    )
                elif self.path == "/health":
                    self.reply(200, {"ok": True, "frames": dashboard.sequence})
                elif self.path == "/events":
                    self.events()
                else:
                    self.reply(404, {"error": "not found"})

            def events(self):
                if not dashboard.stream_slots.acquire(blocking=False):
                    self.reply(503, {"error": "too many viewers"})
                    return
                try:
                    self.send_headers(200, "text/event-stream")
                    last = -1
                    while True:
                        with dashboard.condition:
                            dashboard.condition.wait_for(
                                lambda seen=last: dashboard.sequence != seen or dashboard.closed,
                                timeout=1,
                            )
                            if dashboard.closed:
                                break
                            current, packet = dashboard.sequence, dashboard.latest
                        if packet is not None and current != last:
                            self.wfile.write(b"data: " + packet + b"\n\n")
                        else:
                            self.wfile.write(b": heartbeat\n\n")
                        last = current
                        self.wfile.flush()
                except (OSError, TimeoutError):
                    pass
                finally:
                    dashboard.stream_slots.release()

            def do_POST(self):
                if self.path not in ("/control", "/speed"):
                    self.reply(404, {"error": "not found"})
                    return
                if (
                    not self.allowed_host()
                    or self.headers.get("Origin") not in dashboard.origins
                    or self.headers.get("X-Pokefly-Token") != dashboard.token
                ):
                    self.reply(403, {"error": "same-origin control token required"})
                    return
                speed = self.path == "/speed"
                if speed and dashboard.pacing is None:
                    self.reply(403, {"error": "live speed control is disabled"})
                    return
                if not speed and not dashboard.manual:
                    self.reply(403, {"error": "manual controls are disabled"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 128:
                        raise ValueError("invalid request length")
                    if self.headers.get("Content-Type") != "application/json":
                        raise ValueError("JSON required")
                    payload = json.loads(self.rfile.read(length))
                    if speed:
                        if not isinstance(payload, dict) or set(payload) != {"hz"}:
                            raise ValueError("expected a single decision rate")
                        result = dashboard.pacing.set_hz(payload["hz"])
                        self.reply(200, {"pacing": result, "scope": "wall-clock-only"})
                        return
                    if not isinstance(payload, dict) or set(payload) != {"action"}:
                        raise ValueError("expected a single action")
                    action = payload["action"]
                    if not isinstance(action, str) or action not in ACTIONS:
                        raise ValueError("unknown button")
                    dashboard.commands.put_nowait(action)
                except (ValueError, TypeError, OSError):
                    self.reply(
                        400,
                        {"error": "invalid speed request" if speed else "invalid action request"},
                    )
                    return
                except queue.Full:
                    self.reply(429, {"error": "button queue full"})
                    return
                self.reply(202, {"queued": action, "source": "human"})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.port = self.server.server_port
        self.hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        self.origins = {f"http://{host}" for host in self.hosts}
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def publish(self, packet: dict):
        encoded = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode()
        with self.condition:
            self.latest = encoded
            self.sequence += 1
            self.condition.notify_all()

    def next_action(self) -> str:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return "wait"

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=3)
        self.server.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_args):
        self.close()
