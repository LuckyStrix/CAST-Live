import uuid

from castlive.common import db

CID = str(uuid.uuid4())


def start(capture, token, cid=CID, **kw):
    body = {"id": cid, "mode": "object", "mime": "video/mp4;codecs=avc1", "consent": True, **kw}
    return capture.post(f"/api/{token}/captures", json=body)


def test_info_needs_valid_token(capture, token):
    assert capture.get(f"/api/{token}/info").json()["project"] == "Test"
    assert capture.get("/api/nope/info").status_code == 404


def test_upload_joins_chunks_in_order(capture, token, cfg):
    assert start(capture, token).json() == {"id": CID, "have": []}
    parts = [b"a" * 1024, b"b" * 1024, b"c" * 10]
    for n in (2, 0, 1):  # out of order, as a flaky phone would
        assert capture.put(f"/api/{token}/captures/{CID}/chunks/{n}", content=parts[n]).status_code == 200
    r = capture.post(f"/api/{token}/captures/{CID}/finish",
                     json={"chunks": 3, "meta": {"markers": [{"t": 0, "event": "start"}]}})
    assert r.json() == {"status": "received"}
    assert (cfg.inbox / CID / "video.mp4").read_bytes() == b"".join(parts)
    assert not (cfg.inbox / CID / "chunks").exists()
    assert capture.get(f"/api/{token}/captures/{CID}").json()["status"] == "received"


def test_create_is_idempotent_and_reports_chunks(capture, token):
    start(capture, token)
    capture.put(f"/api/{token}/captures/{CID}/chunks/0", content=b"x")
    assert start(capture, token).json()["have"] == [0]


def test_finish_reports_missing(capture, token):
    start(capture, token)
    capture.put(f"/api/{token}/captures/{CID}/chunks/0", content=b"x")
    r = capture.post(f"/api/{token}/captures/{CID}/finish", json={"chunks": 3})
    assert r.status_code == 409 and r.json()["missing"] == [1, 2]


def test_limits(capture, token):
    start(capture, token)
    url = f"/api/{token}/captures/{CID}/chunks"
    assert capture.put(f"{url}/0", content=b"x" * 1025).status_code == 413  # chunk too big
    assert capture.put(f"{url}/99", content=b"x").status_code == 400  # past the capture size
    for n in range(10):
        assert capture.put(f"{url}/{n}", content=b"x" * 1024).status_code == 200
    assert capture.put(f"{url}/10", content=b"x").status_code == 413  # whole capture too big
    # replacing a chunk does not count twice
    assert capture.put(f"{url}/9", content=b"y" * 1024).status_code == 200


def test_rejects_bad_input(capture, token):
    assert start(capture, token, consent=False).status_code == 400
    assert start(capture, token, mime="text/html").status_code == 400
    assert start(capture, token, cid="../../etc").status_code == 400
    assert start(capture, token, cid=CID.upper()).status_code == 400  # one spelling per id
    assert capture.get(f"/api/{token}/captures/..%2F..%2Fx").status_code == 404
    assert capture.post(f"/api/{token}/captures", content=b"not json").status_code == 400


def test_capture_belongs_to_its_invite(capture, token, cfg):
    start(capture, token)
    with db.connect(cfg.db) as con:
        other = db.create_invite(con, 1, "other")
    assert capture.get(f"/api/{other}/captures/{CID}").status_code == 404
    assert start(capture, other).status_code == 409


def test_revoked_expired_and_used_up(capture, token, cfg):
    with db.connect(cfg.db) as con:
        expired = db.create_invite(con, 1, days=-1)
        limited = db.create_invite(con, 1, max_captures=1)
        con.execute("UPDATE invites SET revoked = 1 WHERE token = ?", (token,))
    assert capture.get(f"/api/{token}/info").status_code == 404
    assert capture.get(f"/api/{expired}/info").status_code == 404
    assert start(capture, limited).status_code == 200
    assert start(capture, limited, cid=str(uuid.uuid4())).status_code == 403


def test_capture_site_has_no_studio_routes(capture, token):
    for path in ("/api/projects", "/api/status", f"/api/invites/{token}/qr.svg", "/docs",
                 "/openapi.json"):
        assert capture.get(path).status_code in (404, 405), path


def test_pages_and_headers(capture, token):
    r = capture.get(f"/c/{token}")
    assert r.status_code == 200 and "capture.js" in r.text
    assert r.headers["referrer-policy"] == "no-referrer"
    assert capture.get("/").status_code == 200
    assert capture.get("/i18n/en.json").json()["finish"]


def test_stopped_upload_is_refused_not_faked(capture, token, cfg):
    # An abandoned or cancelled upload must not answer "ok", or the phone deletes its copy.
    start(capture, token)
    with db.connect(cfg.db) as con:
        db.set_capture(con, CID, status="abandoned")
    assert capture.put(f"/api/{token}/captures/{CID}/chunks/0", content=b"x").status_code == 410
    assert capture.post(f"/api/{token}/captures/{CID}/finish", json={"chunks": 1}).status_code == 410
    assert start(capture, token).status_code == 410


def test_finish_retry_is_harmless(capture, token, cfg):
    start(capture, token)
    capture.put(f"/api/{token}/captures/{CID}/chunks/0", content=b"abc")
    assert capture.post(f"/api/{token}/captures/{CID}/finish", json={"chunks": 1}).status_code == 200
    r = capture.post(f"/api/{token}/captures/{CID}/finish", json={"chunks": 1})
    assert r.json() == {"status": "received"}
    assert (cfg.inbox / CID / "video.mp4").read_bytes() == b"abc"


def test_chunks_refresh_activity_time(capture, token, cfg):
    start(capture, token)
    with db.connect(cfg.db) as con:
        con.execute("UPDATE captures SET updated = 0 WHERE id = ?", (CID,))
    capture.put(f"/api/{token}/captures/{CID}/chunks/0", content=b"x")
    with db.connect(cfg.db) as con:
        assert con.execute("SELECT updated FROM captures WHERE id = ?", (CID,)).fetchone()[0] > 0


def test_cancel_frees_the_slot(capture, token, cfg):
    with db.connect(cfg.db) as con:
        once = db.create_invite(con, 1, max_captures=1)
    start(capture, once)
    capture.put(f"/api/{once}/captures/{CID}/chunks/0", content=b"x")
    assert capture.post(f"/api/{once}/captures/{CID}/cancel").status_code == 200
    assert not (cfg.inbox / CID).exists()
    # rejected and cancelled attempts don't use up the link
    assert start(capture, once, cid=str(uuid.uuid4())).status_code == 200


def test_rate_limit_key_is_last_forwarded_address():
    from starlette.requests import Request

    from castlive.common.web import RateLimit, client_ip

    req = Request({"type": "http", "headers": [(b"x-forwarded-for", b"6.6.6.6, 100.1.2.3")],
                   "client": ("172.18.0.2", 1)})
    assert client_ip(req) == "100.1.2.3"
    rl = RateLimit(count=1, seconds=60)
    for i in range(10_050):
        rl.allow(f"k{i}")
    assert len(rl.events) <= 10_050


def test_parallel_chunk_puts_are_serialised(cfg, token):
    import asyncio

    import httpx

    from castlive.capture.app import create_app

    async def go():
        app = create_app(cfg)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://t") as c:
            await c.post(f"/api/{token}/captures", json={"id": CID, "mode": "object",
                                                        "mime": "video/mp4", "consent": True})
            url = f"/api/{token}/captures/{CID}/chunks"
            # the same chunk twice at once, plus more chunks than the size cap allows
            rs = await asyncio.gather(*[c.put(f"{url}/{n}", content=b"x" * 1024)
                                        for n in [0, 0, *range(1, 11)]])
            return [r.status_code for r in rs]

    codes = asyncio.run(go())
    assert codes.count(413) == 1  # 11 distinct chunks of 1 KiB against a 10 KiB cap
    with db.connect(cfg.db) as con:
        assert con.execute("SELECT bytes FROM captures").fetchone()[0] == 10 * 1024
