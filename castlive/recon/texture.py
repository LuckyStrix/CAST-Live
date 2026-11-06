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


def run_texrecon(scene: Path, ply: Path, out_prefix: Path, log: Path) -> Path:
    """Returns the .obj written (with its .mtl and texture images next to it)."""
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n===== {TEXRECON} {scene} {ply} {out_prefix}\n")
        fh.flush()
        code = subprocess.run([TEXRECON, "--keep_unseen_faces", scene, ply, out_prefix],
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


def make(dense: Path, vertices: np.ndarray, faces: np.ndarray, work: Path, log: Path):
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
    try:
        obj = run_texrecon(work / "scene", work / "mesh.ply", work / "textured", log)
    finally:  # texrecon's working files (hundreds of MB), whether it worked or not
        for junk in [*work.glob("*.spt"), *work.glob("*.vec"), work / "mesh.ply"]:
            junk.unlink(missing_ok=True)
        shutil.rmtree(work / "scene", ignore_errors=True)
    parts = read_obj(obj)
    return parts, {"views": views, "parts": len(parts),
                   "textured": sum(1 for p in parts if p[3] is not None),
                   "seconds": round(time.time() - t)}
