"""Live coverage: the file hand-off, coverage maths, the live service loop (with a fake SLAM)
and the capture site's WebSocket."""

import io
import json
import time
import uuid

import numpy as np
import pytest
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from castlive.common import db
from castlive.live import coverage, service
from castlive.live import session as S

CID = str(uuid.uuid4())


def jpeg(w=320, h=240, colour=(200, 120, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, "JPEG")
    return buf.getvalue()


def ring(n, radius=2.0, height=1.0, turn=1.0):
    """Cameras on a (partial) circle round the origin, looking at it; world up is +Z."""
    a = np.linspace(0, 2 * np.pi * turn, n, endpoint=False)
    c = np.c_[radius * np.cos(a), radius * np.sin(a), np.full(n, height)]
    f = -c / np.linalg.norm(c, axis=1, keepdims=True)
    u = np.tile([0.0, 0, 1], (n, 1))
    return c, f, u


def test_points_pack_round_trip():
    xyz = np.array([[0.5, -1.25, 3.9], [0, 0, 0]])
    rgb = np.array([[1, 2, 3], [250, 0, 128]], np.uint8)
    back, colours = S.unpack_points(S.pack_points(xyz, rgb))
    assert np.allclose(back, xyz, atol=1 / S.POINT_UNITS) and (colours == rgb).all()


@pytest.mark.parametrize("turn,covered", [(1.0, 1.0), (0.5, 0.5), (0.25, 0.25)])
def test_coverage_sectors(turn, covered):
    c, f, u = ring(64, turn=turn)
    centre, R, radius = coverage.frame(c, f, u, fallback=np.zeros(3))
    assert np.allclose(centre, 0, atol=1e-6) and radius == pytest.approx(np.sqrt(5))
    m = coverage.measure(c, centre, R, radius)
    assert m["covered"] == pytest.approx(covered, abs=1 / 16)  # sector edges fall anywhere
    assert m["elevation"] == [27, 27]  # atan(1/2): the cameras are above the object
    up = coverage.normalise(np.array([[0, 0, 1.0]]), centre, R, radius)
    assert np.allclose(up, [[0, 1 / radius, 0]])  # world up becomes +Y


def test_coverage_uses_fallback_while_rays_are_parallel():
    c = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0.0]])
    f = np.tile([0, 1.0, 0], (3, 1))
    centre, _, _ = coverage.frame(c, f, np.tile([0, 0, 1.0], (3, 1)), fallback=np.array([5, 5, 5]))
    assert (centre == 5).all()


class FakeSlam:
    def __init__(self):
        self.resets, self.placed = 0, 0

    def reset(self, first):
        self.resets += 1
        self.placed = 0

    def add(self, rgb):
        ok = rgb.shape == (240, 320, 3)
        self.placed += ok
        return ok

    def cameras(self):  # placed frames walk round a circle, 20 degrees apart
        c, f, u = ring(18)
        return c[:self.placed], f[:self.placed], u[:self.placed]

    def overlay(self, centre=None, reach=None):
        return b"\x02\x00\x01\x00\x03\x00"  # 2 x 1 cells: covered from 3 directions, none

    def points(self, limit):
        rng = np.random.default_rng(1)
        return rng.normal(0, 0.3, (50, 3)), rng.integers(0, 255, (50, 3)).astype(np.uint8)


def test_service_tracks_each_pass_and_ends(tmp_path):
    sess = tmp_path / "live" / CID
    S.write_json(sess / "control.json", {"pass": 1, "ended": False})
    arriving = [(p, i, jpeg()) for p, n in ((1, 6), (2, 4)) for i in range(n)]
    arriving.insert(7, (2, 99, b"not a jpeg"))

    def phone(_):  # the service waits: the phone sends its next frame, then says "end"
        if arriving:
            p, i, raw = arriving.pop(0)
            S.write_atomic(sess / "in" / S.frame_name(p, i), raw)
        else:
            S.write_json(sess / "control.json", {"pass": 2, "ended": True})

    slam = FakeSlam()
    final = service.run(sess, slam, sleep=phone)
    assert final["status"] == "ended" and slam.resets == 2
    assert (sess / "out" / "done").exists() and not (sess / "in").exists()
    state = json.loads((sess / "out" / "state.json").read_text())
    p1, p2 = state["passes"]["1"], state["passes"]["2"]
    assert p1["frames"] == p1["tracked"] == 6 and len(p1["sectors"]) == 16
    assert p2["frames"] == 4 and 0 < p2["covered"] <= 1 and state["pass"] == 2
    xyz, _ = S.unpack_points((sess / "out" / "points.bin").read_bytes())
    assert len(xyz) == 50
    assert (sess / "out" / "overlay.bin").read_bytes() == FakeSlam().overlay()


def test_service_skips_a_backlog(tmp_path):
    sess = tmp_path / "live" / CID
    for i in range(10):
        S.write_atomic(sess / "in" / S.frame_name(1, i), jpeg())
    S.write_json(sess / "control.json", {"pass": 1, "ended": True})
    final = service.run(sess, FakeSlam(), sleep=lambda _: None)
    assert final["passes"]["1"]["frames"] == 2  # the two newest; the rest were already old


def test_service_closes_stale_sessions(tmp_path):
    root = tmp_path / "live"
    S.write_json(root / "old" / "control.json", {"pass": 1})
    S.write_json(root / "new" / "control.json", {"pass": 1})
    assert service.waiting(root, clock=lambda: time.time() + 60) in (root / "old", root / "new")
    assert service.waiting(root, clock=lambda: time.time() + service.STALE + 5) is None
    assert (root / "old" / "out" / "done").exists()
    service.waiting(root, clock=lambda: time.time() + service.KEEP + 5)
    assert not (root / "old").exists()  # long finished: cleared away


# --- The capture site's WebSocket ------------------------------------------------------------

def start_capture(capture, token, cid=CID):
    body = {"id": cid, "mode": "object", "mime": "video/mp4", "consent": True}
    assert capture.post(f"/api/{token}/captures", json=body).status_code == 200


def test_live_socket_needs_a_valid_invite(capture, token):
    with pytest.raises(WebSocketDisconnect), capture.websocket_connect(f"/live/nope/{CID}") as ws:
        ws.receive_text()


def test_live_socket_writes_checked_frames_by_pass(capture, token, cfg):
    start_capture(capture, token)
    sess = S.session(cfg.data, CID)
    with capture.websocket_connect(f"/live/{token}/{CID}") as ws:
        ws.send_bytes(jpeg())
        time.sleep(0.15)
        ws.send_bytes(b"\xff\xd8\xff not really")  # refused: not a JPEG
        time.sleep(0.15)
        ws.send_bytes(jpeg(4000, 3000))  # refused: too big a picture
        time.sleep(0.15)
        ws.send_text(json.dumps({"pass": 2}))
        ws.send_bytes(jpeg())
        ws.send_bytes(jpeg())  # refused: too soon after the last
        time.sleep(0.3)
        assert sorted(p.name for p in (sess / "in").glob("*.jpg")) == ["1_000000.jpg", "2_000001.jpg"]
        assert S.read_json(sess / "control.json") == {"pass": 2, "ended": False, "mode": "object"}
        S.write_json(sess / "out" / "state.json", {"status": "tracking", "pass": 2})
        assert json.loads(ws.receive_text())["pass"] == 2
    time.sleep(0.2)
    assert S.read_json(sess / "control.json")["ended"] is True  # closing the socket ends it


def test_live_socket_one_session_at_a_time(capture, token):
    other = str(uuid.uuid4())
    start_capture(capture, token)
    start_capture(capture, token, other)
    with capture.websocket_connect(f"/live/{token}/{CID}"):
        time.sleep(0.1)
        with capture.websocket_connect(f"/live/{token}/{other}") as ws2:
            assert ws2.receive_json() == {"status": "busy", "why": "in_use"}


def test_no_live_view_while_a_model_is_being_made(capture, token, cfg):
    other = str(uuid.uuid4())
    start_capture(capture, token)
    start_capture(capture, token, other)
    with db.connect(cfg.db) as con:
        db.set_capture(con, other, status="processing")
    with capture.websocket_connect(f"/live/{token}/{CID}") as ws:
        assert ws.receive_json() == {"status": "busy", "why": "model"}


def test_reconstruction_waits_while_someone_films_live(cfg):
    from castlive.worker.loop import wait_for_live

    active = cfg.data / "live" / "active"
    S.write_atomic(active, CID.encode())
    (cfg.data / "live.heartbeat").touch()
    naps = []

    def sleep(s):
        naps.append(s)
        if len(naps) == 3:
            active.unlink()  # the live session ends

    wait_for_live(cfg, sleep=sleep)
    assert len(naps) == 3
    wait_for_live(cfg, sleep=sleep)  # no live session: no wait
    assert len(naps) == 3
    S.write_atomic(active, CID.encode())  # left by a live service that has stopped
    wait_for_live(cfg, sleep=sleep, clock=lambda: time.time() + 3600)
    assert len(naps) == 3


def test_studio_serves_the_live_state(studio, cfg):
    assert studio.get(f"/api/captures/{CID}/live").status_code == 404
    out = S.session(cfg.data, CID) / "out"
    S.write_json(out / "state.json", {"status": "tracking", "pass": 1})
    S.write_atomic(out / "points.bin", S.pack_points(np.zeros((2, 3)), np.zeros((2, 3), np.uint8)))
    assert studio.get(f"/api/captures/{CID}/live").json()["pass"] == 1
    assert len(studio.get(f"/api/captures/{CID}/live/points").content) == 4 + 2 * 9
    assert studio.get("/api/captures/../live").status_code == 404
    assert "wss://testserver" in studio.get("/live").headers["content-security-policy"]


def test_web_apps_do_not_need_numpy():
    """The web image has no numpy: capture and studio must import without it."""
    import subprocess
    import sys

    code = ("import sys; sys.modules['numpy'] = None\n"
            "import castlive.capture.app, castlive.studio.app")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_live_socket_drops_frames_when_the_live_service_is_behind(capture, token, cfg, monkeypatch):
    from castlive.capture import app

    monkeypatch.setattr(app, "LIVE_QUEUE", 2)
    start_capture(capture, token)
    with capture.websocket_connect(f"/live/{token}/{CID}") as ws:
        for _ in range(4):
            ws.send_bytes(jpeg())
            time.sleep(0.15)
        time.sleep(0.2)
    assert len(list((S.session(cfg.data, CID) / "in").glob("*.jpg"))) == 2
