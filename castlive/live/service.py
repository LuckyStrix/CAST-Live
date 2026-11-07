"""The live service (GPU, no network): tracks the frames a phone sends while filming and writes
back what has been covered, for the phone and the studio. One session at a time.

Each pass is tracked on its own: turning the object over breaks tracking on purpose, and the two
passes are joined later by the full reconstruction.
"""

import logging
import shutil
import time
from pathlib import Path

import numpy as np

from castlive.common import config
from castlive.live import coverage
from castlive.live import session as S

log = logging.getLogger("castlive.live")
IDLE_END = 20  # seconds without a frame: the phone has gone; end the session
STALE = 15 * 60  # a session nobody finished or tracked within this long is dropped
POINTS = 4000  # preview points sent to the phone
CAMERAS = 300  # recent camera positions sent
WRITE_EVERY = 0.7  # seconds between updates


def decode(path: Path) -> np.ndarray | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except (OSError, ValueError):
        return None


class Pass:
    """What one pass has tracked so far."""

    def __init__(self, number: int, started: float):
        self.number, self.started, self.frames, self.tracked = number, started, 0, 0
        self.placed = False  # was the last frame placed? (False: tracking is lost)


def summary(p: Pass, slam) -> tuple[dict, bytes | None]:
    """State for this pass, and the preview points (None until there is enough to show)."""
    out = {"pass": p.number, "frames": p.frames, "tracked": p.tracked, "placed": p.placed}
    c, f, u = slam.cameras()
    if len(c) < 3:
        return out, None
    xyz, rgb = slam.points(POINTS)
    if not len(xyz):
        return out, None
    centre, R, radius = coverage.frame(c, f, u, fallback=np.median(xyz, axis=0))
    out.update(coverage.measure(c, centre, R, radius))
    out["cameras"] = np.round(coverage.normalise(c[-CAMERAS:], centre, R, radius), 3).tolist()
    return out, S.pack_points(coverage.normalise(xyz, centre, R, radius), rgb)


def run(sess: Path, slam, clock=time.time, sleep=time.sleep) -> dict:
    """Track one session until the phone ends it or goes quiet. Returns the final state."""
    out = sess / "out"
    state = {"status": "tracking", "passes": {}, "started": clock()}
    p, last_frame, last_write = None, clock(), 0.0
    while True:
        control = S.read_json(sess / "control.json", {})
        frames = S.pending(sess)
        while len(frames) > 2 and frames[0][0] == frames[1][0]:  # behind: skip to the newest
            frames.pop(0)[2].unlink(missing_ok=True)
        if frames:
            number, _, path = frames[0]
            img = decode(path)
            path.unlink(missing_ok=True)
            last_frame = clock()
            if img is not None:
                if p is None or number != p.number:
                    if p is not None:  # keep the finished pass's final numbers
                        state["passes"][str(p.number)] = summary(p, slam)[0]
                    p = Pass(number, clock())
                    slam.reset(img)
                p.frames += 1
                p.placed = slam.add(img)
                if p.placed:
                    p.tracked += 1
        else:
            sleep(0.05)
        ended = control.get("ended") and not frames or clock() - last_frame > IDLE_END
        if p is not None and (ended or clock() - last_write > WRITE_EVERY):
            info, points = summary(p, slam)
            state["passes"][str(p.number)] = info
            state.update({"pass": p.number, "updated": clock(),
                          "fps": round(p.frames / max(1e-6, clock() - p.started), 1)})
            if points is not None:
                S.write_atomic(out / "points.bin", points)
            S.write_json(out / "state.json", state)
            last_write = clock()
        if ended:
            state["status"] = "ended"
            S.write_json(out / "state.json", state)
            (out / "done").touch()
            shutil.rmtree(sess / "in", ignore_errors=True)
            return state


def waiting(root: Path, clock=time.time) -> Path | None:
    """The oldest session that has started and isn't done. Stale ones are closed."""
    found = []
    for control in root.glob("*/control.json"):
        sess = control.parent
        if (sess / "out" / "done").exists():
            continue
        age = clock() - control.stat().st_mtime
        if age > STALE:
            (sess / "out").mkdir(parents=True, exist_ok=True)
            (sess / "out" / "done").touch()
            shutil.rmtree(sess / "in", ignore_errors=True)
            continue
        found.append((control.stat().st_mtime, sess))
    return min(found)[1] if found else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load()
    root = S.root(cfg.data)
    root.mkdir(parents=True, exist_ok=True)
    from castlive.live.slam import Slam

    # Load the model before anyone films: it takes from half a minute to a few minutes.
    log.info("live: loading MASt3R-SLAM")
    t = time.time()
    slam = Slam()
    log.info("live: ready in %.0f s; waiting for sessions in %s", time.time() - t, root)
    while True:
        (cfg.data / "live.heartbeat").touch()
        sess = waiting(root)
        if sess is None:
            time.sleep(0.3)
            continue
        log.info("live: session %s", sess.name)
        S.write_atomic(root / "active", sess.name.encode())
        try:
            final = run(sess, slam)
            log.info("live: session %s ended: %s", sess.name,
                     {k: (v["frames"], v["tracked"], v.get("covered")) for k, v in
                      final["passes"].items()})
        except Exception as e:
            log.exception("live: session %s failed", sess.name)
            S.write_json(sess / "out" / "state.json", {"status": "failed", "error": str(e)[:300]})
            (sess / "out" / "done").touch()
        finally:
            (root / "active").unlink(missing_ok=True)
