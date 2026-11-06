import shutil
import subprocess
import uuid

import pytest

from castlive.common import db
from castlive.worker import loop

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def add_capture(cfg, token, content: bytes | None = None, video=None):
    cid = str(uuid.uuid4())
    d = cfg.inbox / cid
    d.mkdir(parents=True)
    if video:
        shutil.copy(video, d / "video.mp4")
    else:
        (d / "video.mp4").write_bytes(content)
    with db.connect(cfg.db) as con:
        con.execute(
            "INSERT INTO captures (id, project_id, invite, mode, status, mime, created, updated)"
            " VALUES (?, 1, ?, 'object', 'received', 'video/mp4', 0, 0)", (cid, token))
    return cid


def status(cfg, cid):
    with db.connect(cfg.db) as con:
        return con.execute("SELECT status, error, meta FROM captures WHERE id = ?", (cid,)).fetchone()


@needs_ffmpeg
def test_rejects_garbage(cfg, token):
    cid = add_capture(cfg, token, b"<html>not a video</html>")
    assert loop.step_check(cfg)
    s = status(cfg, cid)
    assert s["status"] == "rejected" and "could not be read" in s["error"]


@needs_ffmpeg
def test_accepts_video(cfg, token, tmp_path):
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30",
                    "-t", "6", "-pix_fmt", "yuv420p", str(src)], check=True)
    cid = add_capture(cfg, token, video=src)
    assert loop.step_check(cfg)
    s = status(cfg, cid)
    assert s["status"] == "checked", s["error"]
    assert '"width": 1280' in s["meta"]
    assert not loop.step_check(cfg)  # nothing left


@needs_ffmpeg
def test_webm_without_duration(tmp_path):
    # Live-muxed WebM (what MediaRecorder writes) carries no duration in its header.
    src = tmp_path / "v.webm"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30",
                    "-t", "6", "-c:v", "libvpx", "-deadline", "realtime", "-live", "1",
                    "-f", "webm", str(src)], check=True)
    assert loop.probe(src)["duration"] > 5


def test_worker_survives_unexpected_errors(cfg, token, monkeypatch):
    cid = add_capture(cfg, token, b"x")

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(loop, "probe", boom)
    assert loop.step_check(cfg)
    assert status(cfg, cid)["status"] == "rejected"
    assert not loop.step_check(cfg)  # not picked again


def test_drops_stale_uploads(cfg, token):
    cid = str(uuid.uuid4())
    (cfg.inbox / cid / "chunks").mkdir(parents=True)
    with db.connect(cfg.db) as con:
        con.execute(
            "INSERT INTO captures (id, project_id, invite, mode, status, mime, created, updated)"
            " VALUES (?, 1, ?, 'object', 'uploading', 'video/mp4', 0, 0)", (cid, token))
    loop.drop_stale(cfg)
    assert status(cfg, cid)["status"] == "abandoned"
    assert not (cfg.inbox / cid).exists()
