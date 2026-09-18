import base64
import http.client
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pokefly.dashboard import BrainView, Dashboard, png_data
from pokefly.pacing import RuntimePacing
from pokefly.pixel_brain import PixelObservation


@pytest.fixture
def dashboard():
    with Dashboard({"neurons": 3}, port=0, manual=True) as server:
        yield server


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=3)
    try:
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        connection.close()


def controls(server, **overrides):
    return {
        "Origin": server.url,
        "X-Pokefly-Token": server.token,
        "Content-Type": "application/json",
        **overrides,
    }


def test_local_assets_and_no_filesystem_or_rom_routes(dashboard):
    status, body, headers = request(dashboard, "GET", "/")
    assert status == 200 and b"LEARNING OFF" in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert request(dashboard, "GET", "/dashboard.js")[0] == 200
    for path in ("/../.env", "/Pokemon%20Red.gb", "/fly-data/brain.npz"):
        assert request(dashboard, "GET", path)[0] == 404
    status, body, _ = request(dashboard, "GET", "/static.json")
    assert status == 200 and json.loads(body)["neurons"] == 3
    assert request(dashboard, "GET", "/health", headers={"Host": "evil.example"})[0] == 403


def test_control_requires_same_origin_and_token(dashboard):
    body = json.dumps({"action": "a"})
    assert request(dashboard, "POST", "/control", body, controls(dashboard))[0] == 202
    assert dashboard.next_action() == "a"
    assert dashboard.next_action() == "wait"
    for extra in ({"Origin": "https://example.com"}, {"X-Pokefly-Token": "wrong"}):
        assert request(dashboard, "POST", "/control", body, controls(dashboard, **extra))[0] == 403
    assert request(dashboard, "POST", "/control", body)[0] == 403
    assert dashboard.next_action() == "wait"


def test_observer_mode_cannot_inject_buttons():
    with Dashboard({}, port=0) as server:
        assert request(server, "POST", "/control", '{"action":"a"}', controls(server))[0] == 403


def test_authenticated_speed_edits_do_not_enable_manual_buttons():
    pacing = RuntimePacing(5)
    with Dashboard({"autonomous": True}, port=0, pacing=pacing) as server:
        for hz in (2.5, 250, 0):
            status, body, _ = request(
                server, "POST", "/speed", json.dumps({"hz": hz}), controls(server)
            )
            assert status == 200
            assert json.loads(body)["pacing"]["target_hz"] == hz
            assert json.loads(body)["scope"] == "wall-clock-only"
        _, body, _ = request(server, "GET", "/static.json")
        assert json.loads(body)["speed_control"] is True
        assert json.loads(body)["pacing"] == pacing.state()
        assert request(server, "POST", "/control", '{"action":"up"}', controls(server))[0] == 403
        assert server.next_action() == "wait"


def test_speed_edits_require_same_origin_token_and_supported_mode(dashboard):
    body = '{"hz":0}'
    assert request(dashboard, "POST", "/speed", body, controls(dashboard))[0] == 403
    with Dashboard({}, port=0, pacing=RuntimePacing(5)) as server:
        for extra in (
            {"Origin": "https://evil.example"},
            {"X-Pokefly-Token": "wrong"},
            {"Host": "evil.example"},
        ):
            assert request(server, "POST", "/speed", body, controls(server, **extra))[0] == 403
        assert request(server, "POST", "/speed", body)[0] == 403
        assert server.pacing.state()["target_hz"] == 5


@pytest.mark.parametrize(
    "body",
    [
        '{"hz":-1}',
        '{"hz":3001}',
        '{"hz":true}',
        '{"hz":"0"}',
        '{"hz":NaN}',
        '{"hz":Infinity}',
        '{"hz":null}',
        '{"hz":[]}',
        '{"hz":0,"action":"up"}',
        "[]",
        "{",
        "x" * 129,
    ],
)
def test_invalid_speed_payload_is_rejected(body):
    with Dashboard({}, port=0, pacing=RuntimePacing(5)) as server:
        assert request(server, "POST", "/speed", body, controls(server))[0] == 400
        assert server.pacing.state() == {"target_hz": 5, "revision": 0}


@pytest.mark.parametrize(
    "body",
    [
        '{"action":"select"}',
        '{"action":"a","frames":100000}',
        "[]",
        '{"action":[]}',
        "{",
        "x" * 129,
    ],
)
def test_invalid_control_rejected(dashboard, body):
    assert request(dashboard, "POST", "/control", body, controls(dashboard))[0] == 400
    assert dashboard.next_action() == "wait"


def test_queue_is_bounded(dashboard):
    for _ in range(16):
        dashboard.commands.put_nowait("up")
    assert request(dashboard, "POST", "/control", '{"action":"a"}', controls(dashboard))[0] == 429


def test_stream_delivers_one_synchronized_packet(dashboard):
    packet = {"sample": 7, "action": "a", "spikes": [1, 2], "groups": {"test": 3}}
    dashboard.publish(packet)
    connection = http.client.HTTPConnection("127.0.0.1", dashboard.port, timeout=3)
    try:
        connection.request("GET", "/events")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.readline().removeprefix(b"data: ")) == packet
    finally:
        connection.close()
    assert dashboard.sequence == 1


def test_view_uses_real_positions_and_counts_all_spikes():
    brain = SimpleNamespace(
        n=4,
        positions=np.array([[0, 0, 0], [1, 1, 1], [np.nan] * 3, [2, 2, 2]]),
        weights=np.ones(5),
    )
    retina = SimpleNamespace(
        mapped=np.array([True, False]),
        u=np.array([0.25, np.nan]),
        v=np.array([0.75, np.nan]),
        summary=lambda: {"photoreceptors": 2},
    )
    view = BrainView(SimpleNamespace(brain=brain, retina=retina, groups={"test": np.array([1, 2])}))
    packet = view.activity(
        PixelObservation(np.array([1, 0]), np.array([1, 2, 3, 0]), {"test": 5}, 3)
    )
    assert view.static["missing_positions"] == 1
    assert view.static["groups"]["test"] == [1]
    assert packet["spikes"] == [0, 1]
    assert packet["unique_firing"] == 3
    assert packet["spikes_total"] == 6
    assert base64.b64decode(packet["drive_u8"]) == bytes([255, 0])


def test_png_is_an_image_not_a_rom_endpoint():
    value = png_data(np.zeros((144, 160, 4), np.uint8))
    assert value.startswith("data:image/png;base64,")
    assert base64.b64decode(value.split(",", 1)[1]).startswith(b"\x89PNG")


def test_motor_highlights_identify_individual_firing_cells():
    brain = SimpleNamespace(n=3, positions=np.arange(9).reshape(3, 3), weights=np.ones(2))
    retina = SimpleNamespace(
        mapped=np.array([True]),
        u=np.array([0.5]),
        v=np.array([0.5]),
        summary=lambda: {},
    )
    view = BrainView(
        SimpleNamespace(
            brain=brain,
            retina=retina,
            groups={"forward_Up": np.array([0, 1])},
        )
    )
    observation = PixelObservation(np.array([1.0]), np.array([0, 2, 1]), {"forward_Up": 2}, 3)
    assert view.activity(observation)["motor_fired"] == [1]


def test_shutdown_releases_port(dashboard):
    port = dashboard.port
    dashboard.close()
    assert not dashboard.thread.is_alive()
    with Dashboard({}, port=port):
        pass


def test_graded_state_is_not_a_spike_and_missing_positions_still_count():
    brain = SimpleNamespace(
        n=4,
        positions=np.array([[0, 0, 0], [np.nan] * 3, [1, 1, 1], [2, 2, 2]]),
        weights=np.ones(2),
    )
    retina = SimpleNamespace(
        mapped=np.array([True]), u=np.array([0.5]), v=np.array([0.5]), summary=lambda: {}
    )
    view = BrainView(
        SimpleNamespace(
            brain=brain,
            retina=retina,
            hybrid=SimpleNamespace(graded_host=np.array([0, 1, 2])),
            groups={"photoreceptors": np.array([0, 1]), "lamina_L1_L5": np.array([2])},
        )
    )
    observation = PixelObservation(
        np.array([1.0]),
        np.array([0, 0, 0, 1]),
        {"photoreceptors": 0, "lamina_L1_L5": 0},
        3,
        {"graded_values": [0.2, 0.8, 0.4]},
    )
    packet = view.activity(observation)
    assert packet["spikes_total"] == 1
    assert packet["spikes"] == [2]
    assert view.static["graded_indices"] == [0, 1]
    assert packet["graded_groups"]["photoreceptors"] == 0.5  # Includes unpositioned cell.
    assert base64.b64decode(packet["graded_u8"]) == bytes([51, 102])
    assert "NOT spikes" in packet["graded_units"]
