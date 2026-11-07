"""COLMAP, run as a subprocess (it is in the worker image): camera poses, then a dense cloud.

Checked against COLMAP 4.2.1 (colmap/colmap image). Option names that changed between releases
are looked up in the command's own help. Pose reading and pair lists come from FleshForge.
"""

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from castlive.recon import PipelineError

MODELS = Path(os.environ.get("CASTLIVE_MODELS", "/opt/castlive/models"))
EXHAUSTIVE_MAX = 250  # above this many frames, match a chosen list of pairs
PAIR_WINDOW, PAIR_STRIDE = 10, 8
MIN_FLIP_ANGLE = 120  # degrees between the passes' average camera "up" after a turn-over
MIN_REGISTERED = 0.8  # fraction of frames that must get a camera pose

def available() -> bool:
    return shutil.which("colmap") is not None


def log_error(text: str) -> None:
    import logging

    logging.getLogger("castlive.recon").exception(text)


_COUNT = re.compile(r"\[(\d+)/(\d+)\]")  # "Processed file [12/160]", "Fusing image [3/160]"
_BLOCK = re.compile(r"\[(\d+)/(\d+), (\d+)/(\d+)\]")  # exhaustive "Matching block [1/4, 2/4]"
_VIEW = re.compile(r"Processing view (\d+) / (\d+)")  # patch_match_stereo
_REGISTER = re.compile(r"Registering image #\d+ \((\d+)\)")  # incremental mapper


def line_fraction(line: str, total_images: int = 0) -> float | None:
    """How far a COLMAP command is, from one line of its output, or None."""
    if m := _BLOCK.search(line):
        a, b, c, d = map(int, m.groups())
        return ((a - 1) * d + c) / (b * d)
    if m := _VIEW.search(line):
        return int(m.group(1)) / int(m.group(2))
    if m := _COUNT.search(line):
        return int(m.group(1)) / int(m.group(2))
    if total_images and (m := _REGISTER.search(line)):
        return min(1.0, int(m.group(1)) / total_images)
    return None


def run(command: str, args: list, log: Path, progress=None, total_images: int = 0,
        passes: int = 1, retries: int = 1) -> float:
    """Run `colmap command args`, appending its output to `log`. `progress(fraction)` is called
    as COLMAP reports how far it is. `passes`: how many times the command goes through every
    image (patch_match_stereo with geometric consistency: 2). A crash (killed by a signal, e.g.
    the occasional "CUDA failure 999" abort under WSL) is run again up to `retries` times."""
    for attempt in range(retries + 1):
        try:
            return _run_once(command, args, log, progress, total_images, passes)
        except Crashed:
            if attempt == retries:
                raise
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== colmap {command} crashed; trying again\n")


class Crashed(RuntimeError):
    """COLMAP was killed by a signal rather than exiting with an error."""


def _run_once(command, args, log, progress, total_images, passes) -> float:
    cmd = ["colmap", command, *[str(a) for a in args]]
    start = time.time()
    done_passes, last, shown = 0, 0.0, 0.0
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n===== {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                fh.write(line)
                f = line_fraction(line, total_images) if progress else None
                if f is None:
                    continue
                if f + 0.5 < last and done_passes + 1 < passes:  # counting started over
                    done_passes += 1
                last = f
                # never backwards: a mapper that starts a second sub-model counts from 0 again
                shown = max(shown, min(1.0, (done_passes + f) / passes))
                try:
                    progress(shown)
                except Exception:  # noqa: BLE001 - progress is for display; never stop COLMAP
                    log_error("progress update failed")
            code = proc.wait()
        finally:
            if proc.poll() is None:  # leaving early (an error here): don't orphan COLMAP
                proc.kill()
                proc.wait()
    if code != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-8:])
        error = Crashed if code < 0 else RuntimeError
        raise error(f"colmap {command} failed (exit {code}):\n{tail}")
    return time.time() - start


def match_pairs(names: list, window: int = PAIR_WINDOW, stride: int = PAIR_STRIDE) -> list:
    """Neighbours in order plus every `stride`-th frame: links laps and both passes."""
    n, keep = len(names), set()
    for i in range(n):
        for j in range(i + 1, min(n, i + window + 1)):
            keep.add((i, j))
        for j in range(0, n, stride):
            if j != i:
                keep.add((min(i, j), max(i, j)))
    return [(names[i], names[j]) for i, j in sorted(keep)]


@dataclass
class Model:
    folder: Path
    names: list  # registered image names
    centres: np.ndarray  # (N, 3) camera centres
    ups: np.ndarray  # (N, 3) camera up vectors (world)
    forwards: np.ndarray  # (N, 3) viewing directions (world)
    points: int
    error: float
    overlap: dict  # 3D points by which passes saw them: {"p1": n, "p2": n, "shared": n}


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    w, x, y, z = np.array([qw, qx, qy, qz], float) / np.linalg.norm([qw, qx, qy, qz])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def iter_images(text: str):
    """images.txt entries as (header tokens, image name, observation tokens). Pose lines
    alternate with observation lines, which may be empty (and the last one may be missing)."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) % 2:
        lines.append("")
    for header, obs in zip(lines[0::2], lines[1::2]):
        tok = header.split()
        if len(tok) >= 10:
            yield tok, " ".join(tok[9:]), obs.split()


def parse_images_txt(text: str):
    names, centres, ups, fwds = [], [], [], []
    for tok, _, _ in iter_images(text):
        R = quat_to_rot(*map(float, tok[1:5]))
        t = np.array(list(map(float, tok[5:8])))
        names.append(" ".join(tok[9:]))
        centres.append(-R.T @ t)
        ups.append(-R[1])  # COLMAP camera: +y down, +z forward
        fwds.append(R[2])
    return names, np.array(centres), np.array(ups), np.array(fwds)


def join_quality(xyz: np.ndarray, kinds: list) -> dict:
    """Where the points seen from both passes lie. A real join has them all round the object;
    a false one has a few, bunched along one edge or line.

    spread: size of the shared points' footprint on the object's two main axes, relative to
    all points' (about 0.1-0.4 for the false joins seen, 0.7-1.0 for true ones).
    sectors: of 16 directions round the object's centre, how many hold shared points."""
    xyz = np.asarray(xyz, float)
    shared = np.array([k == "shared" for k in kinds])
    if len(xyz) < 10 or shared.sum() < 3:
        return {"spread": 0.0, "sectors": 0}
    d = np.linalg.norm(xyz - np.median(xyz, 0), axis=1)
    keep = d < np.quantile(d, 0.95)  # far strays would stretch the axes
    xyz, shared = xyz[keep], shared[keep]
    if shared.sum() < 3:
        return {"spread": 0.0, "sectors": 0}
    mean = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - mean, full_matrices=False)
    q = (xyz - mean) @ vt.T
    spread = float(np.sqrt(np.prod(q[shared, :2].std(0)) / np.prod(q[:, :2].std(0))))
    ang = np.arctan2(q[shared, 1], q[shared, 0])
    sectors = len(set(((ang + np.pi) / (2 * np.pi) * 16).astype(int) % 16))
    return {"spread": round(spread, 2), "sectors": sectors}


def flip_check(names: list, centres: np.ndarray, ups: np.ndarray, xyz: np.ndarray) -> dict:
    """Was the object really turned over between pass 1 and pass 2?

    Pass 1 films the object from above; after turning it over, pass 2 films what was its
    underside. So, seen from the object, the two passes' cameras must be on opposite sides of
    its thinnest axis, and their average "up" must point nearly opposite ways. A box or tablet
    looks much like itself flipped, so matching can find a tidy mirror solution with pass 2
    laid onto pass 1's side (both camera sets on one side, "up" within a few degrees)."""
    p1 = np.array([n.startswith("p1_") for n in names])
    p2 = np.array([n.startswith("p2_") for n in names])
    if not p1.any() or not p2.any() or len(xyz) < 10:
        return {"turned_over": False, "up_angle": 0.0, "opposite_sides": False}
    centre = np.median(xyz, 0)
    axis = np.linalg.svd(xyz - centre, full_matrices=False)[2][2]
    h = (centres - centre) @ axis
    u1, u2 = ups[p1].mean(0), ups[p2].mean(0)
    cos = float(u1 @ u2 / (np.linalg.norm(u1) * np.linalg.norm(u2)))
    angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    opposite = bool(np.sign(np.median(h[p1])) != np.sign(np.median(h[p2])))
    return {"turned_over": opposite and angle >= MIN_FLIP_ANGLE, "up_angle": round(angle),
            "opposite_sides": opposite}


def pass_overlap(images_text: str, points_text: str) -> dict:
    """How many 3D points each pass saw alone, and how many were seen from both ("shared"),
    plus the points' count and mean reprojection error ("_points", "_error"), in one read.

    This is the real test of whether two passes joined: frames can all get a position while
    the two sides' positions only rest on a few (possibly wrong) common points."""
    side = {tok[0]: name.split("_")[0] for tok, name, _ in iter_images(images_text)}
    out, xyz, kinds, errors = {"shared": 0}, [], [], []
    for ln in points_text.splitlines():
        if not ln or ln.startswith("#"):
            continue
        tok = ln.split()
        errors.append(float(tok[7]))
        seen = {side.get(i) for i in tok[8::2]} - {None}
        key = "shared" if len(seen) > 1 else next(iter(seen), None)
        if key:
            out[key] = out.get(key, 0) + 1
            xyz.append([float(v) for v in tok[1:4]])
            kinds.append(key)
    if len({k for k in kinds if k != "shared"}) > 1 or out["shared"]:
        out.update(join_quality(np.array(xyz).reshape(-1, 3), kinds))
    out["_xyz"] = np.array(xyz).reshape(-1, 3)
    out["_points"] = len(errors)
    out["_error"] = float(np.mean(errors)) if errors else float("nan")
    return out


def to_text(folder: Path, log: Path) -> Path:
    txt = folder.parent / (folder.name + "_txt")
    shutil.rmtree(txt, ignore_errors=True)
    txt.mkdir()
    run("model_converter", ["--input_path", folder, "--output_path", txt, "--output_type", "TXT"],
        log)
    return txt


def model_from_text(folder: Path, txt: Path) -> Model:
    images_text = (txt / "images.txt").read_text(errors="replace")
    overlap = pass_overlap(images_text, (txt / "points3D.txt").read_text(errors="replace"))
    names, centres, ups, fwds = parse_images_txt(images_text)
    xyz = overlap.pop("_xyz")
    if any(n.startswith("p2_") for n in names):
        overlap["flip"] = flip_check(names, centres, ups, xyz)
    return Model(folder, names, centres, ups, fwds, overlap.pop("_points"),
                 overlap.pop("_error"), overlap)


def read_model(folder: Path, log: Path) -> Model:
    txt = to_text(folder, log)
    try:
        return model_from_text(folder, txt)
    finally:
        shutil.rmtree(txt)


def largest(sparse: Path, log: Path) -> Model | None:
    """The sub-model with the most frames, read in full (the others are only counted)."""
    best, best_n, texts = None, -1, []
    try:
        for folder in sorted(p for p in sparse.iterdir() if (p / "images.bin").exists()):
            txt = to_text(folder, log)
            texts.append(txt)
            n = sum(1 for _ in iter_images((txt / "images.txt").read_text(errors="replace")))
            if n > best_n:
                best, best_n = (folder, txt), n
        return model_from_text(*best) if best else None
    finally:
        for txt in texts:
            shutil.rmtree(txt, ignore_errors=True)


def span(progress, lo: float, hi: float):
    """A progress callback for one part (lo..hi) of a stage."""
    return (lambda f: progress(lo + (hi - lo) * f)) if progress else None


# Feature setups. ALIKED + LightGlue (COLMAP's ONNX models, BSD-3 / Apache-2.0) matches more
# across the turn-over; SIFT is COLMAP's classic.
SETUPS = {
    "aliked": (["--FeatureExtraction.type", "ALIKED_N16ROT",
                "--FeatureExtraction.max_image_size", "2048",
                "--AlikedExtraction.max_num_features", "8192",
                "--AlikedExtraction.n16rot_model_path", MODELS / "aliked-n16rot.onnx"],
               ["--FeatureMatching.type", "ALIKED_LIGHTGLUE",
                "--AlikedMatching.lightglue_model_path", MODELS / "aliked-lightglue.onnx",
                "--AlikedMatching.bruteforce_model_path", MODELS / "bruteforce-matcher.onnx"]),
    "sift": (["--FeatureExtraction.type", "SIFT", "--FeatureExtraction.max_image_size", "3200",
              "--SiftExtraction.max_num_features", "16384"],
             ["--FeatureMatching.type", "SIFT_BRUTEFORCE"]),
}
# Tried in order until one passes `accept`; symmetric objects can fool any one setup, so
# several are tried and checked. SIFT + incremental mapper first: on both turned-over box scans
# it joined the sides right in ~100 s (the global mapper took 13-24 min for the same cameras,
# to within 0.05% of the orbit), and the in-the-ground mouse took 8 min instead of 41 for
# ALIKED; lower reprojection error each time. CAST's box needed the fallbacks.
CANDIDATES = [("sift", "mapper"), ("aliked", "mapper"), ("sift", "global_mapper"),
              ("aliked", "global_mapper")]


def features(images: Path, masks: Path | None, db: Path, setup: str, log: Path,
             progress=None) -> dict:
    """Extract and match one feature setup into `db`. Returns timings."""
    for old in db.parent.glob(db.name + "*"):
        old.unlink()
    extract, match = SETUPS[setup]
    names = sorted(p.name for p in images.glob("*.jpg"))
    args = ["--database_path", db, "--image_path", images,
            "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--FeatureExtraction.use_gpu", "1", *extract]
    if masks is not None:
        args += ["--ImageReader.mask_path", masks]
    secs = {"features": run("feature_extractor", args, log, span(progress, 0, 0.4))}
    if len(names) <= EXHAUSTIVE_MAX:
        matcher, margs = "exhaustive_matcher", []
    else:
        pairs = db.parent / "pairs.txt"
        pairs.write_text("".join(f"{a} {b}\n" for a, b in match_pairs(names)))
        matcher, margs = "matches_importer", ["--match_list_path", pairs, "--match_type", "pairs"]
    secs["matching"] = run(matcher, ["--database_path", db, "--FeatureMatching.use_gpu", "1",
                                     *match, *margs], log, span(progress, 0.4, 1.0))
    return secs


def poses(images: Path, masks: Path | None, work: Path, log: Path, progress=None,
          accept=lambda model: True, candidates=None) -> tuple[Model, dict, list]:
    """Camera poses: each (feature setup, mapper) candidate in turn until `accept(model)` says
    one is good. Returns (the accepted model, or the largest if none was, timings, one summary
    per candidate tried). Each candidate has an equal share of the progress bar, which jumps to
the end when one is accepted."""
    work.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in images.glob("*.jpg"))
    candidates = candidates or CANDIDATES
    secs, best, tried, done, broken = {}, None, [], set(), set()
    for i, (setup, mapper) in enumerate(candidates):
        p = span(progress, i / len(candidates), (i + 1) / len(candidates))
        db = work / f"{setup}.db"
        if setup in broken:
            continue
        if setup not in done:
            try:
                for k, v in features(images, masks, db, setup, log, span(p, 0, 0.45)).items():
                    secs[f"{setup}_{k}"] = v
            except RuntimeError as e:  # e.g. a GPU crash in one setup: the others may still work
                broken.add(setup)
                tried.append({"setup": setup, "mapper": mapper, "placed": 0, "overlap": {},
                              "accepted": False, "error": str(e).splitlines()[0]})
                continue
            done.add(setup)
        out = work / f"{setup}_{mapper}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        try:
            secs[f"{setup}_{mapper}"] = run(mapper, ["--database_path", db, "--image_path",
                                                     images, "--output_path", out], log,
                                            span(p, 0.45, 1.0), total_images=len(names))
            model = largest(out, log)
        except RuntimeError:
            model = None
        ok = bool(model) and accept(model)
        tried.append({"setup": setup, "mapper": mapper,
                      "placed": len(model.names) if model else 0,
                      "overlap": model.overlap if model else {}, "accepted": ok})
        if ok:
            return model, secs, tried
        if model and (best is None or len(model.names) > len(best.names)):
            best = model
    if best is None:
        raise PipelineError("No camera positions could be found. The object needs more texture "
                            "or light, or the walk round was too fast.")
    return best, secs, tried


def undistort(images: Path, model_dir: Path, work: Path, log: Path, max_size: int,
              progress=None) -> float:
    shutil.rmtree(work, ignore_errors=True)
    return run("image_undistorter", [
        "--image_path", images, "--input_path", model_dir, "--output_path", work,
        "--output_type", "COLMAP", "--max_image_size", max_size], log, progress)


def crop_cameras(cameras_text: str, images_text: str, boxes: dict) -> tuple[str, str]:
    """Rewrite an undistorted (PINHOLE) model so each image has its own camera cropped to
    `boxes[name] = (x0, y0, x1, y1)`: size from the box, principal point moved by its corner,
    2D observations moved likewise. Images without a box keep their full frame."""
    cams = {}
    for ln in cameras_text.splitlines():
        if ln and not ln.startswith("#"):
            t = ln.split()
            if t[1] != "PINHOLE":
                raise RuntimeError(f"undistorted camera model is {t[1]}, expected PINHOLE")
            cams[t[0]] = [float(v) for v in t[2:8]]  # w h fx fy cx cy
    out_cams, out_images = [], []
    for t, name, p in iter_images(images_text):
        image_id = t[0]
        w, h, fx, fy, cx, cy = cams[t[8]]
        x0, y0, x1, y1 = boxes.get(name, (0, 0, int(w), int(h)))
        out_cams.append(f"{image_id} PINHOLE {x1 - x0} {y1 - y0} {fx!r} {fy!r} "
                        f"{cx - x0!r} {cy - y0!r}")
        t[8] = image_id
        out_images.append(" ".join(t))
        for i in range(0, len(p) - 2, 3):  # x y point3D_id
            p[i], p[i + 1] = repr(float(p[i]) - x0), repr(float(p[i + 1]) - y0)
        out_images.append(" ".join(p))
    return "\n".join(out_cams) + "\n", "\n".join(out_images) + "\n"


def mask_box(mask, margin: float, min_side: int):
    """(x0, y0, x1, y1) round the mask's object plus a margin, or None when there is no
    object or the box would be too small to be useful."""
    if mask is None or not mask.any():
        return None
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pad = margin * max(xs.max() - xs.min(), ys.max() - ys.min())
    x0, y0 = max(0, int(xs.min() - pad)), max(0, int(ys.min() - pad))
    x1, y1 = min(w, int(xs.max() + pad) + 1), min(h, int(ys.max() + pad) + 1)
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return x0, y0, x1, y1


def crop_to_masks(work: Path, mask_dir: Path, log: Path, margin: float = 0.12,
                  min_side: int = 256) -> dict:
    """Crop every undistorted image (and its mask) to the object and give each its own camera.
    Depth maps then spend their pixels on the object, not the table."""
    import cv2

    boxes, areas = {}, []
    for img_path in sorted((work / "images").iterdir()):
        mask_path = mask_dir / (img_path.name + ".png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        box = mask_box(mask, margin, min_side)
        if box is None:
            areas.append(1.0)
            continue
        x0, y0, x1, y1 = box
        image = cv2.imread(str(img_path))
        cv2.imwrite(str(img_path), image[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mask_path), mask[y0:y1, x0:x1])
        boxes[img_path.name] = box
        areas.append((x1 - x0) * (y1 - y0) / (mask.shape[0] * mask.shape[1]))

    txt = work / "sparse_txt"
    shutil.rmtree(txt, ignore_errors=True)
    txt.mkdir()
    run("model_converter", ["--input_path", work / "sparse", "--output_path", txt,
                            "--output_type", "TXT"], log)
    cameras, images = crop_cameras((txt / "cameras.txt").read_text(),
                                   (txt / "images.txt").read_text(), boxes)
    sparse = work / "sparse"
    for old in sparse.iterdir():
        old.unlink()
    (sparse / "cameras.txt").write_text(cameras)
    (sparse / "images.txt").write_text(images)
    shutil.copy(txt / "points3D.txt", sparse / "points3D.txt")
    shutil.rmtree(txt)
    return {"images": len(areas), "cropped": len(boxes),
            "median_area_kept": round(float(np.median(areas)), 3) if areas else 1.0}


def stereo(work: Path, log: Path, max_size: int = 2000, progress=None) -> float:
    """A depth and normal map per image (GPU), with geometric consistency (two passes)."""
    return run("patch_match_stereo", [
        "--workspace_path", work, "--workspace_format", "COLMAP",
        "--PatchMatchStereo.max_image_size", max_size,
        "--PatchMatchStereo.geom_consistency", "1"], log, progress, passes=2)


def fuse(work: Path, out_ply: Path, masks: Path | None, log: Path, progress=None) -> float:
    args = ["--workspace_path", work, "--workspace_format", "COLMAP",
            "--input_type", "geometric", "--output_path", out_ply]
    if masks is not None:
        args += ["--StereoFusion.mask_path", masks]
    return run("stereo_fusion", args, log, progress)
