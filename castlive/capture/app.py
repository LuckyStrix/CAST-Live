"""The public capture site (behind Tailscale Funnel).

It takes a video (and, while filming, small live frames) from a phone that holds a valid invite
token. It has no admin routes and reads nothing back except a capture's own upload status and,
while filming, its own live coverage.
"""

import asyncio
import io
import json
import logging
import os
import shutil
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from castlive.common import config, db
from castlive.common.web import RateLimit, add_security_headers, client_ip
from castlive.live import session as live

log = logging.getLogger("uvicorn.error")
MAX_META_BYTES = 64 * 1024

# Live view: small JPEG frames from the phone while it films (see castlive/live/session.py).
LIVE_FRAME_BYTES = 300_000
LIVE_SIDE = 1024  # longest side of a live frame
LIVE_GAP = 0.12  # seconds between frames: at most ~8 per second; faster ones are dropped
LIVE_QUEUE = 30  # frames waiting for the live service; beyond this (it is down or behind), drop
IDLE_FINISH = 30  # after the phone says "end", seconds to wait for the final state
LIVE_WAIT = 20  # seconds to wait for the capture record (the uploader makes it as filming starts)

per_ip = RateLimit(count=900, seconds=60)  # ~15 requests/s: chunks plus status polls
new_capture = RateLimit(count=20, seconds=3600)  # per invite


def write_chunk(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def join_chunks(chunks: Path, count: int, out: Path) -> None:
    """Join the pieces in order. The output name is ours, never the phone's."""
    tmp = out.with_suffix(".joining")
    with open(tmp, "wb") as f:
        for i in range(count):
            with open(chunks / f"{i:06d}.part", "rb") as part:
                shutil.copyfileobj(part, f)
    os.replace(tmp, out)
    shutil.rmtree(chunks)


async def json_body(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > MAX_META_BYTES:
        raise HTTPException(413, "too much data")
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        raise HTTPException(400, "bad JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(400, "bad JSON")
    return body


def good_jpeg(raw: bytes) -> bool:
    """A real JPEG of a sensible size, checked here so the GPU side only sees sound files."""
    if len(raw) > LIVE_FRAME_BYTES or raw[:3] != b"\xff\xd8\xff":
        return False
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "JPEG" or max(im.size) > LIVE_SIDE or min(im.size) < 64:
                return False
            im.verify()
        return True
    except Exception:  # noqa: BLE001 - anything Pillow rejects is not a frame
        return False


def queued(sess: Path) -> int:
    try:
        with os.scandir(sess / "in") as it:
            return sum(1 for _ in it)
    except OSError:
        return 0


def create_app(cfg: config.Config | None = None) -> FastAPI:
    cfg = cfg or config.load()
    db.init(cfg.db)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.middleware("http")(add_security_headers)

    @app.middleware("http")
    async def limit(request: Request, call_next):
        if not per_ip.allow(client_ip(request)):
            return JSONResponse({"error": "slow down"}, status_code=429)
        return await call_next(request)

    web = config.WEB
    app.mount("/assets", StaticFiles(directory=web / "capture"), name="assets")
    app.mount("/i18n", StaticFiles(directory=web / "i18n"), name="i18n")

    @app.get("/")
    def no_link():
        return FileResponse(web / "capture" / "nolink.html")

    @app.get("/c/{token}")
    def capture_page(token: str):
        return FileResponse(web / "capture" / "index.html")

    # One lock per capture: chunk writes, finish and cancel for the same capture never overlap.
    # (One capture process, so an asyncio lock is enough.)
    locks: dict[str, asyncio.Lock] = {}

    def invite_or_404(con, token):
        inv = db.valid_invite(con, token)
        if inv is None:
            raise HTTPException(404, "This link is not valid. Ask your organiser for a new one.")
        return inv

    def capture_or_404(con, token, cid):
        row = None
        if db.is_capture_id(cid):
            row = con.execute(
                "SELECT * FROM captures WHERE id = ? AND invite = ?", (cid, token)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "unknown capture")
        return row

    def gone_unless_open(cap):
        """Closed captures can't take more data. The phone must not think it was sent."""
        if cap["status"] in ("abandoned", "cancelled"):  # (rejected/failed: the video arrived)
            raise HTTPException(410, "This upload was stopped. Please record again.")

    def chunk_dir(cid):
        return cfg.inbox / cid / "chunks"

    def have_chunks(cid) -> list[int]:
        d = chunk_dir(cid)
        if not d.is_dir():
            return []
        return sorted(int(p.stem) for p in d.glob("*.part"))

    @app.get("/api/{token}/info")
    def info(token: str):
        with db.connect(cfg.db) as con:
            inv = invite_or_404(con, token)
            project = con.execute(
                "SELECT name, mode FROM projects WHERE id = ?", (inv["project_id"],)
            ).fetchone()
        return {
            "project": project["name"],
            "mode": project["mode"],
            "chunk_bytes": cfg.chunk_bytes,
            "max_bytes": cfg.max_capture_bytes,
            "max_seconds": cfg.max_capture_seconds,
        }

    @app.post("/api/{token}/captures")
    async def create_capture(token: str, request: Request):
        body = await json_body(request)
        cid = str(body.get("id", ""))
        mode = body.get("mode")
        mime = str(body.get("mime", "")).split(";")[0].strip()
        if not body.get("consent"):
            raise HTTPException(400, "consent is required")
        if mode not in db.MODES or mime not in db.VIDEO_EXT:
            raise HTTPException(400, "bad mode or video type")
        if not db.is_capture_id(cid):
            raise HTTPException(400, "bad capture id")
        with db.connect(cfg.db) as con:
            inv = invite_or_404(con, token)
            row = con.execute("SELECT invite, status FROM captures WHERE id = ?", (cid,)).fetchone()
            if row is not None:  # a retry from the same phone: idempotent
                if row["invite"] != token:
                    raise HTTPException(409, "capture id in use")
                gone_unless_open(row)
                return {"id": cid, "have": have_chunks(cid)}
            if inv["max_captures"] is not None and db.captures_used(con, token) >= inv[
                "max_captures"
            ]:
                raise HTTPException(403, "This link has been used up. Ask your organiser.")
            if not new_capture.allow(token):
                raise HTTPException(429, "Too many captures from this link. Try later.")
            meta = {"device": str(body.get("device", ""))[:300]}
            now = time.time()
            con.execute(
                "INSERT INTO captures (id, project_id, invite, mode, status, mime, meta,"
                " created, updated) VALUES (?, ?, ?, ?, 'uploading', ?, ?, ?, ?)",
                (cid, inv["project_id"], token, mode, mime, json.dumps(meta), now, now),
            )
        return {"id": cid, "have": []}

    @app.put("/api/{token}/captures/{cid}/chunks/{n}")
    async def put_chunk(token: str, cid: str, n: int, request: Request):
        max_chunks = cfg.max_capture_bytes // cfg.chunk_bytes + 1
        if not 0 <= n < max_chunks:
            raise HTTPException(400, "bad chunk number")
        with db.connect(cfg.db) as con:
            invite_or_404(con, token)
            capture_or_404(con, token, cid)
        data = bytearray()
        async for piece in request.stream():
            data += piece
            if len(data) > cfg.chunk_bytes:
                log.warning("capture %s chunk %d over %d bytes", cid, n, cfg.chunk_bytes)
                raise HTTPException(413, f"piece {n} is over {cfg.chunk_bytes} bytes")
        async with locks.setdefault(cid, asyncio.Lock()):
            with db.connect(cfg.db) as con:
                cap = capture_or_404(con, token, cid)
                gone_unless_open(cap)
                if cap["status"] != "uploading":
                    return {"ok": True}  # already joined; a late retry is harmless
                path = chunk_dir(cid) / f"{n:06d}.part"
                old = path.stat().st_size if path.exists() else 0
                if cap["bytes"] - old + len(data) > cfg.max_capture_bytes:
                    log.warning("capture %s over the %d byte limit", cid, cfg.max_capture_bytes)
                    raise HTTPException(413, "The recording is too large.")
                await run_in_threadpool(write_chunk, path, bytes(data))
                con.execute(
                    "UPDATE captures SET bytes = bytes - ? + ?, updated = ? WHERE id = ?",
                    (old, len(data), time.time(), cid),
                )
        return {"ok": True}

    @app.post("/api/{token}/captures/{cid}/finish")
    async def finish(token: str, cid: str, request: Request):
        body = await json_body(request)
        count = body.get("chunks")
        async with locks.setdefault(cid, asyncio.Lock()):
            with db.connect(cfg.db) as con:
                invite_or_404(con, token)
                cap = capture_or_404(con, token, cid)
                gone_unless_open(cap)
                if cap["status"] != "uploading":
                    return {"status": cap["status"]}  # a retry after it was joined
                if not isinstance(count, int) or count < 1:
                    raise HTTPException(400, "bad chunk count")
                have = have_chunks(cid)
                missing = sorted(set(range(count)) - set(have))
                if missing or len(have) != count:
                    return JSONResponse({"missing": missing[:100]}, status_code=409)
                out = db.video_path(cfg.inbox, cid, cap["mime"])
                await run_in_threadpool(join_chunks, chunk_dir(cid), count, out)
                meta = json.loads(cap["meta"])
                meta["phone"] = body.get("meta", {})  # markers, sensor info; untrusted data
                db.set_capture(con, cid, status="received", chunks=count, meta=meta)
        locks.pop(cid, None)
        return {"status": "received"}

    @app.post("/api/{token}/captures/{cid}/cancel")
    async def cancel(token: str, cid: str):
        async with locks.setdefault(cid, asyncio.Lock()):
            with db.connect(cfg.db) as con:
                cap = capture_or_404(con, token, cid)
                if cap["status"] == "uploading":
                    db.set_capture(con, cid, status="cancelled", error="Cancelled on the phone.")
                    await run_in_threadpool(shutil.rmtree, cfg.inbox / cid, True)
        locks.pop(cid, None)
        return {"ok": True}

    @app.get("/api/{token}/captures/{cid}")
    def status(token: str, cid: str):
        with db.connect(cfg.db) as con:
            invite_or_404(con, token)
            cap = capture_or_404(con, token, cid)
        out = {"status": cap["status"], "error": cap["error"]}
        if cap["status"] == "uploading":
            out["have"] = have_chunks(cid)
        if cap["status"] == "processing":
            p = json.loads(cap["progress"] or "{}")
            out["progress"] = {k: p[k] for k in ("fraction", "eta_seconds") if k in p}
        return out

    live_now: set[str] = set()  # one live session at a time: there is one GPU

    @app.websocket("/live/{token}/{cid}")
    async def live_socket(ws: WebSocket, token: str, cid: str):
        """Phone -> JPEG frames (binary) and {"pass": n} / {"end": true} (text).
        Server -> the live state (text JSON) and preview points (binary)."""
        if not db.is_capture_id(cid) or not per_ip.allow(client_ip(ws)):
            await ws.close(code=1008)
            return
        with db.connect(cfg.db) as con:
            if db.valid_invite(con, token) is None:
                await ws.close(code=1008)
                return
        await ws.accept()
        for _ in range(LIVE_WAIT * 2):  # the capture record appears once uploading starts
            with db.connect(cfg.db) as con:
                cap = con.execute("SELECT status FROM captures WHERE id = ? AND invite = ?",
                                  (cid, token)).fetchone()
            if cap is not None:
                break
            await asyncio.sleep(0.5)
        sess = live.session(cfg.data, cid)
        if cap is None or cap["status"] != "uploading" or (sess / "control.json").exists():
            await ws.send_json({"status": "refused"})
            await ws.close()
            return
        if live_now:
            await ws.send_json({"status": "busy"})
            await ws.close()
            return
        live_now.add(cid)
        control = {"pass": 1, "ended": False}
        live.write_json(sess / "control.json", control)
        started = time.monotonic()
        limit = cfg.max_capture_seconds + 60

        async def send_updates():
            seen = {}
            while True:
                for name in ("state.json", "points.bin"):
                    f = sess / "out" / name
                    try:
                        mtime = f.stat().st_mtime
                    except OSError:
                        continue
                    if seen.get(name) != mtime:
                        seen[name] = mtime
                        data = await run_in_threadpool(f.read_bytes)
                        if name == "state.json":
                            await ws.send_text(data.decode())
                        else:
                            await ws.send_bytes(data)
                if (sess / "out" / "done").exists():
                    return
                await asyncio.sleep(0.4)

        sender = asyncio.create_task(send_updates())
        seq, last = 0, 0.0
        try:
            while time.monotonic() - started < limit:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    now = time.monotonic()
                    raw = msg["bytes"]
                    if now - last < LIVE_GAP or queued(sess) >= LIVE_QUEUE:
                        continue
                    if not await run_in_threadpool(good_jpeg, raw):
                        continue
                    last = now
                    name = live.frame_name(control["pass"], seq)
                    await run_in_threadpool(live.write_atomic, sess / "in" / name, raw)
                    seq += 1
                elif msg.get("text") is not None:
                    try:
                        body = json.loads(msg["text"])
                    except ValueError:
                        continue
                    if not isinstance(body, dict):
                        continue
                    if body.get("pass") == control["pass"] + 1 and body["pass"] <= 2:
                        control["pass"] = body["pass"]
                    if body.get("end"):
                        control["ended"] = True
                    live.write_json(sess / "control.json", control)
                    if control["ended"]:
                        await asyncio.wait_for(sender, timeout=IDLE_FINISH)
                        break
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
            pass
        finally:
            control["ended"] = True
            live.write_json(sess / "control.json", control)
            sender.cancel()
            live_now.discard(cid)
            try:
                await ws.close()
            except RuntimeError:
                pass

    return app


def main():
    uvicorn.run(
        create_app(),
        host=os.environ.get("CASTLIVE_CAPTURE_HOST", "127.0.0.1"),
        port=int(os.environ.get("CASTLIVE_CAPTURE_PORT", "8001")),
        log_level="info",
    )
