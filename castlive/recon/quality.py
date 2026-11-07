"""How good is a mesh? Two checks that need nothing but the capture itself:

- outline: rendered from every camera, how much of the object (its mask) the mesh fails to
  cover (holes, missing parts) and how much it covers outside it (blobs, extra parts);
- accuracy: how far COLMAP's best 3D points (seen from 3+ photos, small reprojection error)
  lie from the surface, as a fraction of the object's size.
"""

from pathlib import Path

import numpy as np

from castlive.recon import splat

SCALE = 4  # render the outline at a quarter of the photo size
MIN_TRACK = 3
MAX_ERROR_PX = 1.5


def good_points(sparse: Path) -> np.ndarray:
    pts = []
    for ln in (sparse / "points3D.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            t = ln.split()
            if float(t[7]) <= MAX_ERROR_PX and (len(t) - 8) // 2 >= MIN_TRACK:
                pts.append([float(v) for v in t[1:4]])
    return np.array(pts).reshape(-1, 3)


def measure(dense: Path, vertices: np.ndarray, faces: np.ndarray, use_masks: bool) -> dict:
    import open3d as o3d
    from PIL import Image

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32))
    mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    lo, hi = vertices.min(0), vertices.max(0)
    size = float(np.linalg.norm(hi - lo))
    out = {}

    if use_masks:
        missed, extra = [], []
        for v in splat.read_views(dense / "sparse"):
            w, h = max(1, v.w // SCALE), max(1, v.h // SCALE)
            K = v.K.copy()
            K[0] *= w / v.w
            K[1] *= h / v.h
            rays = scene.create_rays_pinhole(intrinsic_matrix=K, extrinsic_matrix=v.w2c,
                                             width_px=w, height_px=h)
            hit = np.isfinite(scene.cast_rays(rays)["t_hit"].numpy())
            with Image.open(dense / "masks" / (v.name + ".png")) as m:
                mask = np.asarray(m.convert("L").resize((w, h))) > 127
            if mask.sum():
                missed.append((mask & ~hit).sum() / mask.sum())
                extra.append((hit & ~mask).sum() / mask.sum())
        if missed:
            out["outline_missed"] = round(float(np.mean(missed)), 4)  # holes: 0 is best
            out["outline_missed_worst"] = round(float(np.max(missed)), 4)
            out["outline_extra"] = round(float(np.mean(extra)), 4)

    pts = good_points(dense / "sparse")
    near = pts[np.all((pts >= lo - 0.02 * size) & (pts <= hi + 0.02 * size), axis=1)]
    if len(near) >= 20:
        d = np.abs(scene.compute_distance(o3d.core.Tensor(near.astype(np.float32))).numpy())
        out["points_checked"] = len(near)
        out["point_distance_median"] = round(float(np.median(d)) / size, 5)  # of the size
        out["point_distance_p90"] = round(float(np.quantile(d, 0.9)) / size, 5)
    return out
