"""How good is a mesh? Two checks that need nothing but the capture itself:

- outline: rendered from every camera, how much of the object (its mask) the mesh fails to
  cover (holes, missing parts) and how much it covers outside it (blobs, extra parts). Mask
  patches the mesh doesn't touch at all are other things the masking model caught (seen: spools
  behind a box) and are left out; a missing part of the object is joined to the rest of it;
- accuracy: how far COLMAP's best 3D points (seen from 3+ photos, small reprojection error;
  for objects, only points inside the object's outline in 90% of the photos framing them)
  lie from the surface, as a fraction of its size.
"""

from pathlib import Path

import numpy as np

from castlive.recon import splat

SCALE = 4  # render the outline at a quarter of the photo size


def good_points(sparse: Path) -> np.ndarray:
    return np.array(list(splat.good_points(sparse).values())).reshape(-1, 3)


def measure(dense: Path, vertices: np.ndarray, faces: np.ndarray, use_masks: bool) -> dict:
    import open3d as o3d
    from PIL import Image
    from scipy.ndimage import label

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32))
    mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    lo, hi = vertices.min(0), vertices.max(0)
    size = float(np.linalg.norm(hi - lo))
    out = {}

    views = splat.read_views(splat.text_model(dense))
    pts = good_points(splat.text_model(dense))
    on_object = np.zeros(len(pts), int)  # how many photos see each point on the object
    in_view = np.zeros(len(pts), int)

    if use_masks:
        missed, extra = [], []
        for v in views:
            w, h = max(1, v.w // SCALE), max(1, v.h // SCALE)
            K = v.K.copy()
            K[0] *= w / v.w
            K[1] *= h / v.h
            rays = scene.create_rays_pinhole(intrinsic_matrix=K, extrinsic_matrix=v.w2c,
                                             width_px=w, height_px=h)
            hit = np.isfinite(scene.cast_rays(rays)["t_hit"].numpy())
            with Image.open(dense / "masks" / (v.name + ".png")) as m:
                full = np.asarray(m.convert("L").resize((w, h))) > 127
            cam = pts @ v.w2c[:3, :3].T + v.w2c[:3, 3]
            px = cam @ K.T
            with np.errstate(divide="ignore", invalid="ignore"):
                x, y = px[:, 0] / px[:, 2], px[:, 1] / px[:, 2]
            ok = (cam[:, 2] > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            in_view += ok
            on_object[ok] += full[y[ok].astype(int), x[ok].astype(int)]
            parts, _ = label(full)
            touched = np.unique(parts[hit & full])
            mask = full
            if (touched > 0).any():  # the mesh missing the object entirely still counts
                mask = np.isin(parts, touched[touched > 0])
            if mask.sum():
                missed.append((mask & ~hit).sum() / mask.sum())
                extra.append((hit & ~mask).sum() / mask.sum())
        if missed:
            out["outline_missed"] = round(float(np.mean(missed)), 4)  # holes: 0 is best
            out["outline_missed_median"] = round(float(np.median(missed)), 4)
            out["outline_missed_worst"] = round(float(np.max(missed)), 4)
            out["outline_extra"] = round(float(np.mean(extra)), 4)
        # points on the object: inside its outline in (nearly) every photo that frames them;
        # floor points beside it fall outside from most directions
        pts = pts[(in_view > 0) & (on_object >= 0.9 * in_view)]
    else:
        pts = pts[np.all((pts >= lo - 0.02 * size) & (pts <= hi + 0.02 * size), axis=1)]

    if len(pts) >= 20:
        d = np.abs(scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy())
        out["points_checked"] = len(pts)
        out["point_distance_median"] = round(float(np.median(d)) / size, 5)  # of the size
        out["point_distance_p90"] = round(float(np.quantile(d, 0.9)) / size, 5)
    return out
