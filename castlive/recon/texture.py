"""Colour from the photos. texrecon (mvs-texturing, BSD-3) picks, for each part of the mesh, the
sharpest head-on frame, evens out the seams and packs the pieces into texture images, so ink,
text and barcodes stay as sharp as the video. Point colours (the fallback) are far coarser.
"""

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from castlive.recon import colmap
from castlive.recon.colmap import iter_images, quat_to_rot

TEXRECON = "texrecon"
GAIN_SAMPLES = 20_000  # surface points used to compare the photos' brightness
GAIN_SCALE = 4  # photos are read at a quarter size for that (colour statistics only)
GAIN_MIN_SEEN = 200  # a photo seeing fewer of the points keeps its own brightness
EVEN = False  # even out the photos' brightness before texturing (being tested)
OUTLIERS = "none"  # texrecon's outlier removal: none, gauss_damping, gauss_clamping


def exposure_gains(images: Path, views, vertices: np.ndarray, faces: np.ndarray,
                   seed: int = 0) -> dict[str, np.ndarray]:
    """One colour multiplier per photo, so the same surface looks the same in all of them (the
    phone's exposure and white balance drift as it moves round). Points spread over the mesh
    are looked up in every photo that sees them face-on; per photo and colour channel, the log
    brightness offset from the points' common value is solved for (alternating means, robust
    to shadows and highlights by trimming). Applied to whole photos: nothing is blurred.
    Returns {image name: rgb multiplier}."""
    import open3d as o3d
    from PIL import Image

    rng = np.random.default_rng(seed)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    area = np.linalg.norm(normal, axis=1)
    pick = rng.choice(len(faces), GAIN_SAMPLES, p=area / area.sum())
    u, v = rng.random((2, GAIN_SAMPLES))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[pick] + u[:, None] * (b - a)[pick] + v[:, None] * (c - a)[pick]
    nrm = normal[pick] / np.maximum(area[pick], 1e-12)[:, None]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices.astype(np.float32), faces.astype(np.uint32))
    size = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))

    obs_view, obs_pt, obs_rgb = [], [], []
    for k, view in enumerate(views):
        centre = -view.w2c[:3, :3].T @ view.w2c[:3, 3]
        to_cam = centre - pts
        dist = np.linalg.norm(to_cam, axis=1)
        facing = (nrm * to_cam).sum(1) / dist > 0.3  # face-on enough (outward normals)
        cam = pts @ view.w2c[:3, :3].T + view.w2c[:3, 3]
        px = cam @ view.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = px[:, 0] / px[:, 2], px[:, 1] / px[:, 2]
        ok = facing & (cam[:, 2] > 0) & (x >= 0) & (x < view.w) & (y >= 0) & (y < view.h)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        rays = np.concatenate([np.repeat(centre[None], len(idx), 0), -to_cam[idx] / dist[idx, None]], 1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
        idx = idx[np.abs(hit - dist[idx]) < 0.003 * size]  # not hidden behind other surface
        with Image.open(images / view.name) as im:
            small = np.asarray(im.convert("RGB").reduce(GAIN_SCALE), dtype=np.float32) / 255
        sx = np.clip((x[idx] / GAIN_SCALE).astype(int), 0, small.shape[1] - 1)
        sy = np.clip((y[idx] / GAIN_SCALE).astype(int), 0, small.shape[0] - 1)
        rgb = small[sy, sx]
        good = (rgb.min(1) > 0.03) & (rgb.max(1) < 0.97)  # not black or clipped
        obs_view.append(np.full(good.sum(), k))
        obs_pt.append(idx[good])
        obs_rgb.append(rgb[good])
    if not obs_view:
        return {}
    vi, pi, L = np.concatenate(obs_view), np.concatenate(obs_pt), np.log(np.concatenate(obs_rgb))
    n_views = len(views)
    gain, surface = np.zeros((n_views, 3)), np.zeros((GAIN_SAMPLES, 3))
    keep = np.ones(len(vi), bool)
    for it in range(20):
        if it == 10:  # drop what no single brightness explains: shadows, highlights
            resid = np.abs(L - gain[vi] - surface[pi]).max(1)
            keep = resid < max(3 * np.median(resid), 0.05)
        cnt_p = np.bincount(pi[keep], minlength=GAIN_SAMPLES)[:, None]
        surface = np.stack([np.bincount(pi[keep], (L - gain[vi])[keep, ch], GAIN_SAMPLES)
                            for ch in range(3)], 1) / np.maximum(cnt_p, 1)
        cnt_v = np.bincount(vi[keep], minlength=n_views)[:, None]
        gain = np.stack([np.bincount(vi[keep], (L - surface[pi])[keep, ch], n_views)
                         for ch in range(3)], 1) / np.maximum(cnt_v, 1)
        gain[cnt_v[:, 0] < GAIN_MIN_SEEN] = 0
        gain -= np.median(gain[cnt_v[:, 0] >= GAIN_MIN_SEEN], axis=0)  # keep the typical photo
    return {view.name: np.exp(-gain[k]) for k, view in enumerate(views)}


def even_out(images: Path, gains: dict[str, np.ndarray], out: Path) -> None:
    """Copies of the photos with their gains applied (into texrecon's scene folder)."""
    from PIL import Image

    for name, g in gains.items():
        with Image.open(images / name) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) * g
        Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8)).save(out / name, quality=95)


def write_scene(images: Path, cameras_txt: str, images_txt: str, out: Path) -> int:
    """texrecon's scene: each image next to a `.cam` file (world-to-camera pose, then focal
    length and principal point as fractions of the image size). Cameras must be PINHOLE
    (undistorted). Returns the number of views written."""
    cams = {}
    for ln in cameras_txt.splitlines():
        if ln and not ln.startswith("#"):
            t = ln.split()
            if t[1] != "PINHOLE":
                raise RuntimeError(f"camera model is {t[1]}, expected PINHOLE")
            cams[t[0]] = [float(v) for v in t[2:8]]  # w h fx fy cx cy
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for tok, _, _ in iter_images(images_txt):
        name = " ".join(tok[9:])
        if not (images / name).exists():
            continue
        w, h, fx, fy, cx, cy = cams[tok[8]]
        R = quat_to_rot(*map(float, tok[1:5]))
        t = [float(v) for v in tok[5:8]]
        stem = Path(name).stem
        (out / f"{stem}.cam").write_text(
            " ".join(f"{v:.10g}" for v in [*t, *R.reshape(-1)]) + "\n"
            + f"{fx / max(w, h):.10g} 0 0 {fy / fx:.10g} {cx / w:.10g} {cy / h:.10g}\n")
        link = out / name
        if not link.exists():
            link.symlink_to((images / name).resolve())
        n += 1
    return n


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    v = np.ascontiguousarray(vertices, dtype="<f4")
    f = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    f["n"], f["i"] = 3, faces
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  f"element face {len(f)}\nproperty list uchar int vertex_indices\n"
                  "end_header\n").encode())
        fh.write(v.tobytes())
        fh.write(f.tobytes())


def run_texrecon(scene: Path, ply: Path, out_prefix: Path, log: Path, outliers: str = "none") -> Path:
    """Returns the .obj written (with its .mtl and texture images next to it)."""
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n===== {TEXRECON} {scene} {ply} {out_prefix}\n")
        fh.flush()
        code = subprocess.run([TEXRECON, "--keep_unseen_faces", f"--outlier_removal={outliers}",
                               scene, ply, out_prefix],
                              stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
    obj = out_prefix.with_suffix(".obj")
    if code or not obj.exists():
        raise RuntimeError(f"texrecon failed (exit {code})")
    return obj


def read_obj(obj: Path) -> list:
    """texrecon's OBJ -> parts `(vertices, uvs, faces, texture path or None)`, one per material,
    with vertices split wherever one corner has two UVs. UVs are flipped to glTF's top-left
    origin."""
    v, vt, groups, current = [], [], {}, None
    with open(obj) as fh:
        for ln in fh:
            if ln.startswith("v "):
                v.append(ln.split()[1:4])
            elif ln.startswith("vt "):
                vt.append(ln.split()[1:3])
            elif ln.startswith("usemtl "):
                current = ln.split()[1]
            elif ln.startswith("f "):
                groups.setdefault(current, []).append(
                    [c.split("/")[:2] for c in ln.split()[1:4]])
    v = np.array(v, dtype=np.float64)
    vt = np.array(vt, dtype=np.float64) if vt else np.zeros((1, 2))
    maps, mtl = {}, obj.with_suffix(".mtl")
    if mtl.exists():
        name = None
        for ln in mtl.read_text().splitlines():
            if ln.startswith("newmtl "):
                name = ln.split()[1]
            elif ln.startswith("map_Kd ") and name:
                maps[name] = obj.parent / ln.split(maxsplit=1)[1].strip()
    parts = []
    for name, corners in groups.items():
        c = np.array([[[int(i or 0) for i in k] for k in f] for f in corners]).reshape(-1, 2) - 1
        image = maps.get(name)
        if image is None:
            c[:, 1] = -1  # untextured: only positions matter
        pairs, inverse = np.unique(c, axis=0, return_inverse=True)
        uvs = vt[np.maximum(pairs[:, 1], 0)].copy()
        uvs[:, 1] = 1 - uvs[:, 1]
        parts.append((v[pairs[:, 0]], uvs, inverse.reshape(-1, 3), image))
    return parts


def make(dense: Path, vertices: np.ndarray, faces: np.ndarray, work: Path, log: Path,
         even: bool = EVEN, outliers: str = OUTLIERS):
    """Texture the mesh (model coordinates) from the undistorted images in `dense`. Returns
    (parts as in `read_obj` with texture paths, report)."""
    t = time.time()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    sparse = dense / "sparse"
    txt = sparse if (sparse / "images.txt").exists() else colmap.to_text(sparse, log)
    views = write_scene(dense / "images", (txt / "cameras.txt").read_text(),
                        (txt / "images.txt").read_text(), work / "scene")
    write_ply(work / "mesh.ply", vertices, faces)
    report = {}
    if even:  # replace the scene's links to the photos with evened-out copies
        from castlive.recon import splat

        g = exposure_gains(dense / "images", splat.read_views(txt), vertices, faces)
        for name in g:
            (work / "scene" / name).unlink(missing_ok=True)
        even_out(dense / "images", g, work / "scene")
        spread = np.abs(np.log(np.array(list(g.values())))).max(1) if g else np.zeros(1)
        report["exposure_change_p90"] = round(float(np.exp(np.quantile(spread, 0.9))) - 1, 3)
    try:
        obj = run_texrecon(work / "scene", work / "mesh.ply", work / "textured", log, outliers)
    finally:  # texrecon's working files (hundreds of MB), whether it worked or not
        for junk in [*work.glob("*.spt"), *work.glob("*.vec"), work / "mesh.ply"]:
            junk.unlink(missing_ok=True)
        shutil.rmtree(work / "scene", ignore_errors=True)
    parts = read_obj(obj)
    return parts, {"views": views, "parts": len(parts),
                   "textured": sum(1 for p in parts if p[3] is not None),
                   "seconds": round(time.time() - t), **report}
