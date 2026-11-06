"""The worker: one job at a time, found by polling SQLite.

Two roles, run as separate containers: check each received upload is one real video within
limits, and (GPU image, with COLMAP) reconstruct checked captures one at a time.
"""

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from castlive.common import config, db

log = logging.getLogger("castlive.worker")
STALE_DAYS = 14  # an upload not touched for this long is given up and its chunks deleted


def probe(path, timeout=60) -> dict:
    """ffprobe the file. Raises ValueError with a message a capturer can understand."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
             "-show_format", str(path)],
            capture_output=True, timeout=timeout, check=True, text=True,
        ).stdout
    except subprocess.CalledProcessError:
        raise ValueError("The video could not be read. Please record again.") from None
    except subprocess.TimeoutExpired:
        raise ValueError("The video could not be read in time. Please record again.") from None
    info = json.loads(out)
    video = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if len(video) != 1:
        raise ValueError("The upload is not a single video. Please record again.")
    v = video[0]
    duration = float(info.get("format", {}).get("duration") or v.get("duration") or 0)
    if not duration:  # browser-recorded WebM has no duration in its header: read the last frame time
        try:
            times = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "packet=pts_time", "-of", "csv=p=0", str(path)],
                check=False, capture_output=True, timeout=timeout * 5, text=True,
            ).stdout.split()
        except subprocess.TimeoutExpired:
            raise ValueError("The video could not be read in time. Please record again.") from None
        duration = max((float(t) for t in times if t.replace(".", "", 1).isdigit()), default=0)
    return {
        "codec": v.get("codec_name"),
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "fps": v.get("avg_frame_rate"),
        "duration": round(duration, 2),
        # WebM from a browser has no bitrate in its header: work it out from size and length
        "bitrate": int(info.get("format", {}).get("bit_rate") or 0)
        or (int(Path(path).stat().st_size * 8 / duration) if duration else 0),
    }


def check(cfg: config.Config, cap) -> None:
    path = db.video_path(cfg.inbox, cap["id"], cap["mime"])
    with db.connect(cfg.db) as con:
        try:
            p = probe(path)
            if min(p["width"], p["height"]) < 480:
                raise ValueError("The video is too small. Use the main camera at full quality.")
            if p["duration"] < 5:
                raise ValueError("The video is too short. Walk all the way round the object.")
            if p["duration"] > cfg.max_capture_seconds + 10:
                raise ValueError("The video is too long.")
        except ValueError as e:
            db.set_capture(con, cap["id"], status="rejected", error=str(e))
            return
        meta = json.loads(cap["meta"])
        meta["video"] = p
        db.set_capture(con, cap["id"], status="checked", meta=meta)


def drop_stale(cfg: config.Config) -> None:
    cutoff = time.time() - STALE_DAYS * 86400
    with db.connect(cfg.db) as con:
        rows = con.execute(
            "SELECT id FROM captures WHERE status = 'uploading' AND updated < ?", (cutoff,)
        ).fetchall()
        for r in rows:  # mark first, so a late chunk is refused rather than re-creating files
            db.set_capture(con, r["id"], status="abandoned", error="Upload never finished.")
            shutil.rmtree(cfg.inbox / r["id"], ignore_errors=True)


# Seconds per stage for a ~160-frame capture before this machine has finished any run of its
# own; after that, the median of its recent runs is used.
DEFAULT_SECONDS = {"frames": 60, "masks": 60, "poses": 150, "dense": 900, "mesh": 120}


def stage_estimates(cfg: config.Config) -> dict:
    """Expected seconds per stage, from the last few finished runs on this machine."""
    est = dict(DEFAULT_SECONDS)
    with db.connect(cfg.db) as con:
        rows = con.execute("SELECT meta FROM captures WHERE status = 'done'"
                           " ORDER BY updated DESC LIMIT 5").fetchall()
    seen: dict = {}
    for r in rows:
        for stage, secs in json.loads(r["meta"]).get("report", {}).get("seconds", {}).items():
            seen.setdefault(stage, []).append(secs)
    for stage, values in seen.items():
        est[stage] = sorted(values)[len(values) // 2]
    return est


class Tracker:
    """Turns (stage, fraction of that stage) into an overall fraction and seconds left."""

    def __init__(self, stages: list, estimates: dict, clock=time.time):
        self.stages, self.clock = stages, clock
        self.est = {s: max(1.0, float(estimates.get(s, 60))) for s in stages}
        self.stage, self.started = None, clock()

    def update(self, stage: str, fraction: float) -> dict:
        now = self.clock()
        if stage != self.stage:
            self.stage, self.started = stage, now
        fraction = min(max(fraction, 0.0), 1.0)
        elapsed = now - self.started
        expected = self.est.get(stage, 60.0)
        if fraction >= 0.1:  # measured pace beats the estimate once there is some
            left_here = elapsed * (1 - fraction) / fraction
        else:
            left_here = max(expected - elapsed, expected * (1 - fraction))
        i = self.stages.index(stage) if stage in self.stages else len(self.stages)
        later = self.stages[i + 1:]
        total = sum(self.est.values())
        before = sum(self.est[s] for s in self.stages[:i])
        return {"stage": stage, "stage_fraction": round(fraction, 3),
                "fraction": round(min(1.0, (before + expected * fraction) / total), 3),
                "eta_seconds": round(left_here + sum(self.est[s] for s in later)),
                "updated": round(now)}


LIVE_WAIT_MAX = 20 * 60  # longest a reconstruction waits for a live session (seconds)


def wait_for_live(cfg: config.Config, sleep=time.sleep, clock=time.time) -> None:
    """Someone is filming with the live view on: the GPU is theirs. Wait between stages."""
    active = cfg.data / "live" / "active"
    start = clock()
    if active.exists():
        log.info("waiting: live session %s is using the GPU", active.read_text(errors="replace"))
    while active.exists() and clock() - start < LIVE_WAIT_MAX:
        (cfg.data / "reconstructor.heartbeat").touch()
        sleep(2)


MAX_ATTEMPTS = 2  # a capture whose reconstruction kills the process (out of memory) twice fails
CRASHED = "The 3D model could not be made because of a problem on the server. The organiser can try again."


def reconstruct(cfg: config.Config, cap) -> None:
    from castlive.recon import PipelineError
    from castlive.recon.run import process

    cid = cap["id"]
    meta = json.loads(cap["meta"])
    meta["attempts"] = meta.get("attempts", 0) + 1
    meta.pop("report", None)
    with db.connect(cfg.db) as con:
        if meta["attempts"] > MAX_ATTEMPTS:
            log.error("capture %s: reconstruction stopped %d times; giving up", cid, MAX_ATTEMPTS)
            db.set_capture(con, cid, status="failed", stage="", error=CRASHED, meta=meta,
                           progress={})
            return
        db.set_capture(con, cid, status="processing", error="", meta=meta)

    from castlive.recon.run import stages_for

    tracker = Tracker(stages_for(cap["mode"]), stage_estimates(cfg))
    last = {"stage": None, "time": 0.0}

    def progress(stage, fraction):
        now = time.time()
        if stage == last["stage"] and now - last["time"] < 2:  # at most every 2 s
            return
        last.update(stage=stage, time=now)
        with db.connect(cfg.db) as con:
            db.set_capture(con, cid, stage=stage, progress=tracker.update(stage, fraction))

    try:
        report = process(db.video_path(cfg.inbox, cid, cap["mime"]), meta, cap["mode"],
                         cfg.runs / cid, progress, wait=lambda: wait_for_live(cfg))
    except PipelineError as e:
        with db.connect(cfg.db) as con:
            db.set_capture(con, cid, status="failed", stage="", error=str(e), progress={})
        return
    meta["report"] = report
    meta.pop("attempts")
    with db.connect(cfg.db) as con:
        db.set_capture(con, cid, status="done", stage="", meta=meta, progress={})


def step_check(cfg: config.Config) -> bool:
    """Check one received upload, if there is one. True if something was done."""
    with db.connect(cfg.db) as con:
        cap = con.execute(
            "SELECT * FROM captures WHERE status = 'received' ORDER BY updated LIMIT 1"
        ).fetchone()
    if cap is None:
        return False
    try:
        check(cfg, cap)
    except Exception:  # never let one bad job crash-loop the worker and block the queue
        log.exception("checking capture %s failed", cap["id"])
        with db.connect(cfg.db) as con:
            db.set_capture(con, cap["id"], status="rejected",
                           error="The server could not check this video. Please record again.")
    return True


def step_reconstruct(cfg: config.Config) -> bool:
    """Reconstruct one capture, if there is one. True if something was done."""
    with db.connect(cfg.db) as con:
        # 'processing' first: the job this worker was doing when it last stopped
        cap = con.execute(
            "SELECT * FROM captures WHERE status IN ('processing', 'checked')"
            " ORDER BY status = 'processing' DESC, updated LIMIT 1"
        ).fetchone()
    if cap is None:
        return False
    try:
        reconstruct(cfg, cap)
    except Exception:  # a bug, not the capture's fault: keep it retryable from studio
        log.exception("reconstructing capture %s failed", cap["id"])
        with db.connect(cfg.db) as con:
            db.set_capture(con, cap["id"], status="failed", stage="", error=CRASHED, progress={})
    return True


def heartbeat(path, every=10.0) -> None:
    """Touch `path` from a background thread, so a long job still shows as alive."""

    def beat():
        while True:
            path.touch()
            time.sleep(every)

    threading.Thread(target=beat, daemon=True).start()


def main(argv=None):
    """`castlive-worker` checks uploads; `castlive-worker --reconstruct` (GPU image) builds models.
    They run as separate containers, so a long reconstruction never delays a check."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load()
    db.init(cfg.db)
    argv = sys.argv[1:] if argv is None else argv
    if "--reconstruct" in argv:
        from castlive.recon import colmap

        if not colmap.available():
            log.error("COLMAP not found: reconstruction needs the GPU worker image")
            while True:
                time.sleep(3600)
        heartbeat(cfg.data / "reconstructor.heartbeat")
        log.info("reconstructor started")
        while True:
            if not step_reconstruct(cfg):
                time.sleep(3)
    heartbeat(cfg.data / "worker.heartbeat")
    log.info("upload checker started")
    last_sweep = 0.0
    while True:
        if time.time() - last_sweep > 3600:
            drop_stale(cfg)
            last_sweep = time.time()
        if not step_check(cfg):
            time.sleep(2)
