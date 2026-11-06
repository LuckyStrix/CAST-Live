"""Video -> sharp, evenly spread still frames, one set per pass.

The phone records markers: in `object` mode the turn-over sits between `pass1_end` and
`pass2_start`, and those seconds (hands, a moving object) are left out. Frames are named
`p1_000001.jpg`, `p2_000001.jpg`, so later stages know which pass a frame came from.

More candidates than needed are extracted and the sharpest in each time window is kept: a
hand-held orbit blurs in bursts. (Approach from FleshForge, same author, MIT.)
"""

import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np

from castlive.recon import PipelineError

CANDIDATES_PER_FRAME = 3
FLIP_MARGIN = 0.7  # seconds trimmed either side of the turn-over: hands come in before and after
MIN_SEGMENT = 3.0  # a pass shorter than this is not used


def passes(markers: list, duration: float, mode: str) -> list[tuple[int, float, float]]:
    """(pass number, start, end) of each pass to use. The number is the real pass (1 or 2),
    even when the other pass was too short to keep: later stages rely on p1 = the first side."""
    t = {m.get("event"): float(m.get("t", 0)) for m in markers or []}
    end = min(t.get("stop", duration), duration)
    if "pass1_end" in t and "pass2_start" in t:
        # object: the turn-over (hands, moving object); in the ground: the pause between
        # loops (reading the card, phone lowered). Either way, not frames to use.
        margin = FLIP_MARGIN if mode == "object" else 0.3
        segs = [(0.0, t["pass1_end"] - margin), (t["pass2_start"] + margin, end)]
    else:
        segs = [(0.0, end)]  # one pass
    return [(k, round(a, 2), round(b, 2)) for k, (a, b) in enumerate(segs, 1)
            if b - a >= MIN_SEGMENT]


def split_count(segs, total: int) -> list[int]:
    """Frames to keep per segment, in proportion to its length (at least 20 each)."""
    lengths = np.array([b - a for _, a, b in segs], float)
    return [max(20, round(total * n / lengths.sum())) for n in lengths]


def sharpness(path: Path) -> float:
    """Variance of the Laplacian at a fixed width: higher is sharper."""
    import cv2

    grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if grey is None:
        return 0.0
    h, w = grey.shape
    if w > 640:
        grey = cv2.resize(grey, (640, max(1, round(h * 640 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def select_sharpest(scores, count: int) -> list[int]:
    """Index of the sharpest entry in each of `count` equal windows, in order."""
    scores = np.asarray(scores, float)
    n = len(scores)
    if n <= count:
        return list(range(n))
    edges = np.linspace(0, n, count + 1).round().astype(int)
    return [int(lo + np.argmax(scores[lo:hi])) for lo, hi in itertools.pairwise(edges) if hi > lo]


def extract(video: Path, start: float, end: float, count: int, max_size: int, out: Path) -> list:
    out.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(gt(iw,ih),min({max_size},iw),-2)':'if(gt(iw,ih),-2,min({max_size},ih))'"
    fps = count / (end - start)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
           "-i", str(video), "-vf", f"fps={fps:.5f},{scale}", "-q:v", "2", str(out / "%06d.jpg")]
    done = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        raise PipelineError(f"The video could not be decoded: {done.stderr.strip()[-300:]}")
    return sorted(out.glob("*.jpg"))


def make_frames(video: Path, markers: list, duration: float, mode: str, out: Path,
                total: int = 160, max_size: int = 1600, progress=None) -> dict:
    """Fill `out` with the chosen frames. Returns numbers for the report."""
    segs = passes(markers, duration, mode)
    if not segs:
        raise PipelineError("The recording is too short to use. Please record again.")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    report = {"passes": []}
    span = sum(b - a for _, a, b in segs)
    done = 0.0
    for (k, a, b), count in zip(segs, split_count(segs, total)):
        scratch = out.parent / f"candidates_{k}"
        if scratch.exists():
            shutil.rmtree(scratch)
        cands = extract(video, a, b, count * CANDIDATES_PER_FRAME, max_size, scratch)
        share = (b - a) / span
        if progress:
            progress(done + 0.6 * share)  # decoding is most of the time
        scores = [sharpness(p) for p in cands]
        done += share
        if progress:
            progress(done)
        keep = select_sharpest(scores, count)
        for n, i in enumerate(keep, 1):
            shutil.copyfile(cands[i], out / f"p{k}_{n:06d}.jpg")
        shutil.rmtree(scratch)
        report["passes"].append({"pass": k, "start": a, "end": b, "candidates": len(cands),
                                 "kept": len(keep)})
    report["frames"] = sum(p["kept"] for p in report["passes"])
    if report["frames"] < 20:
        raise PipelineError("Too few usable frames. Please record a slower, longer walk round.")
    return report
