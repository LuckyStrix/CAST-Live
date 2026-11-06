"""The studio site: projects, invite links, captures. Tailnet only.

`tailscale serve` adds Tailscale-User-Login to every tailnet request and Funnel traffic never
carries it. Studio sits on a docker network shared only with the tailscale container, so nothing
else (in particular not the public capture app) can reach it and forge that header.
"""

import json
import os
import shutil
import time
from pathlib import Path

import segno
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from castlive.common import config, db
from castlive.common.web import add_security_headers
from castlive.live import session as live


def create_app(cfg: config.Config | None = None) -> FastAPI:
    cfg = cfg or config.load()
    db.init(cfg.db)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.middleware("http")(add_security_headers)

    @app.middleware("http")
    async def tailnet_only(request: Request, call_next):
        if not cfg.trust_local and not request.headers.get("tailscale-user-login"):
            return JSONResponse({"error": "Studio is only reachable on your tailnet."}, 403)
        return await call_next(request)

    web = config.WEB
    app.mount("/assets", StaticFiles(directory=web / "studio"), name="assets")

    def public_base(request: Request) -> str:
        """Where phones reach the capture site: the same machine name on Funnel's port 443."""
        if os.environ.get("CASTLIVE_PUBLIC_URL"):
            return os.environ["CASTLIVE_PUBLIC_URL"].rstrip("/")
        return "https://" + request.url.hostname

    @app.get("/")
    def index():
        return FileResponse(web / "studio" / "index.html")

    @app.get("/view")
    def viewer():
        return FileResponse(web / "studio" / "view.html")

    @app.get("/live")
    def live_page():
        return FileResponse(web / "studio" / "live.html")

    @app.get("/api/status")
    def status(request: Request):
        def alive(name):
            beat = cfg.data / name
            return beat.exists() and time.time() - beat.stat().st_mtime < 30

        free = shutil.disk_usage(cfg.data).free
        return {
            "user": request.headers.get("tailscale-user-login", "local"),
            "capture_url": public_base(request),
            "worker_ok": alive("worker.heartbeat"),
            "reconstructor_ok": alive("reconstructor.heartbeat"),
            "disk_free_gb": round(free / 1e9, 1),
            "disk_ok": free > 20e9,
        }

    @app.get("/api/projects")
    def projects():
        with db.connect(cfg.db) as con:
            rows = con.execute(
                "SELECT p.*, (SELECT COUNT(*) FROM captures c WHERE c.project_id = p.id) AS n"
                " FROM projects p ORDER BY p.created DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    @app.post("/api/projects")
    async def new_project(request: Request):
        body = await request.json()
        name = str(body.get("name", "")).strip()[:200]
        if not name:
            raise HTTPException(400, "name is required")
        try:
            with db.connect(cfg.db) as con:
                pid = db.create_project(con, name, body.get("mode", "object"))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"id": pid}

    @app.get("/api/projects/{pid}")
    def project(pid: int, request: Request):
        with db.connect(cfg.db) as con:
            p = con.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
            if p is None:
                raise HTTPException(404, "no such project")
            invites = con.execute(
                "SELECT i.*, (SELECT COUNT(*) FROM captures c WHERE c.invite = i.token) AS used"
                " FROM invites i WHERE project_id = ? ORDER BY created DESC",
                (pid,),
            ).fetchall()
            caps = con.execute(
                "SELECT * FROM captures WHERE project_id = ? ORDER BY created DESC", (pid,)
            ).fetchall()
        base = public_base(request)
        return {
            "project": dict(p),
            "invites": [
                dict(i, url=f"{base}/c/{i['token']}", active=db.invite_active(i))
                for i in invites
            ],
            "captures": [dict(c, meta=json.loads(c["meta"]), progress=json.loads(c["progress"]),
                              live=(live.session(cfg.data, c["id"]) / "control.json").exists())
                         for c in caps],
        }

    @app.post("/api/projects/{pid}/invites")
    async def new_invite(pid: int, request: Request):
        body = await request.json()
        days = body.get("days", 30)
        max_caps = body.get("max_captures")
        with db.connect(cfg.db) as con:
            if con.execute("SELECT 1 FROM projects WHERE id = ?", (pid,)).fetchone() is None:
                raise HTTPException(404, "no such project")
            token = db.create_invite(
                con, pid, str(body.get("label", ""))[:200],
                days=float(days) if days else None,
                max_captures=int(max_caps) if max_caps else None,
            )
        return {"token": token, "url": f"{public_base(request)}/c/{token}"}

    @app.post("/api/invites/{token}/revoke")
    def revoke(token: str):
        with db.connect(cfg.db) as con:
            con.execute("UPDATE invites SET revoked = 1 WHERE token = ?", (token,))
        return {"ok": True}

    @app.get("/api/invites/{token}/qr.svg")
    def qr(token: str, request: Request):
        with db.connect(cfg.db) as con:
            if con.execute("SELECT 1 FROM invites WHERE token = ?", (token,)).fetchone() is None:
                raise HTTPException(404, "no such invite")
        svg = segno.make(f"{public_base(request)}/c/{token}", error="m").svg_inline(scale=6)
        return Response(svg, media_type="image/svg+xml")

    @app.get("/api/captures/{cid}/video")
    def video(cid: str):
        with db.connect(cfg.db) as con:
            cap = con.execute("SELECT mime FROM captures WHERE id = ?", (cid,)).fetchone()
        if cap is None:
            raise HTTPException(404, "no such capture")
        path = db.video_path(cfg.inbox, cid, cap["mime"])
        if not path.exists():
            raise HTTPException(404, "not uploaded yet")
        return FileResponse(path, media_type=cap["mime"])

    def run_file(cid: str, name: str) -> Path:
        if not db.is_capture_id(cid):
            raise HTTPException(404, "no such capture")
        path = cfg.runs / cid / name
        if not path.exists():
            raise HTTPException(404, "not made yet")
        return path

    def live_file(cid: str, name: str) -> Path:
        if not db.is_capture_id(cid):
            raise HTTPException(404, "no such capture")
        path = live.session(cfg.data, cid) / "out" / name
        if not path.exists():
            raise HTTPException(404, "no live view yet")
        return path

    @app.get("/api/captures/{cid}/live")
    def live_state(cid: str):
        """The live tracking state while (and after) the phone films."""
        return FileResponse(live_file(cid, "state.json"), media_type="application/json",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/captures/{cid}/live/points")
    def live_points(cid: str):
        return FileResponse(live_file(cid, "points.bin"), media_type="application/octet-stream",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/captures/{cid}/model.glb")
    def model(cid: str):
        return FileResponse(run_file(cid, "model.glb"), media_type="model/gltf-binary",
                            filename=f"castlive-{cid[:8]}.glb")

    @app.get("/api/captures/{cid}/log")
    def run_log(cid: str):
        """The pipeline's own log, then the end of COLMAP's output."""
        if not db.is_capture_id(cid):
            raise HTTPException(404, "no such capture")
        out = []
        steps = cfg.runs / cid / "run.log"
        out.append(steps.read_text(errors="replace") if steps.exists() else "(not started)\n")
        tool = cfg.runs / cid / "colmap.log"
        if tool.exists():
            with open(tool, "rb") as fh:
                fh.seek(max(0, tool.stat().st_size - 16_000))
                tail = fh.read().decode(errors="replace").splitlines()[1:][-60:]
            out.append("\n--- COLMAP, last lines ---\n" + "\n".join(tail))
        return Response("".join(out), media_type="text/plain; charset=utf-8")

    @app.get("/view-log")
    def log_page():
        return FileResponse(web / "studio" / "log.html")

    @app.get("/api/captures/{cid}/report")
    def report(cid: str):
        return FileResponse(run_file(cid, "report.json"), media_type="application/json")

    @app.post("/api/captures/{cid}/retry")
    def retry(cid: str):
        """Run reconstruction again: a failed run continues from the stage that failed, a
        finished one starts over (e.g. after the pipeline has improved)."""
        with db.connect(cfg.db) as con:
            cap = con.execute("SELECT status, meta FROM captures WHERE id = ?", (cid,)).fetchone()
            if cap is None:
                raise HTTPException(404, "no such capture")
            if cap["status"] not in ("failed", "done"):
                raise HTTPException(409, f"cannot retry a capture that is {cap['status']}")
            if cap["status"] == "done":
                shutil.rmtree(cfg.runs / cid, ignore_errors=True)
            meta = json.loads(cap["meta"])
            meta.pop("report", None)
            meta.pop("attempts", None)
            db.set_capture(con, cid, status="checked", error="", stage="", meta=meta, progress={})
        return {"ok": True}

    return app


def main():
    uvicorn.run(
        create_app(),
        host=os.environ.get("CASTLIVE_STUDIO_HOST", "127.0.0.1"),
        port=int(os.environ.get("CASTLIVE_STUDIO_PORT", "8002")),
        log_level="info",
    )
