"""Gaussian splat -> surface. Replaces COLMAP's dense stereo, which leaves holes on plain faces.

A 2D gaussian splat (gsplat, Apache-2.0; 2DGS: flat discs that sit on the surface) is trained on
the undistorted (and, for objects, masked and cropped) photos from the dense workspace. Its
depth, rendered from every camera, is fused into a signed-distance grid on the GPU, and marching
cubes gives the mesh. Off-object pixels count as empty space, so the object's outline from every
view carves the shape too. Parts no camera saw stay open (honest output, as in CAST).

Run on its own (experiments): python -m castlive.recon.splat <dense dir> <out dir> [--steps N]
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from castlive.recon import PipelineError
from castlive.recon.colmap import iter_images, quat_to_rot

STEPS = 7_000  # training steps (one photo each): more gave no better surface on the box
HOLD_OUT = 12  # every n-th photo is kept out of training, to check the splat on unseen views
GRID = 448  # fusion cells along the longest side of the object's box
SH0 = 0.28209479177387814  # spherical harmonics: colour = SH0 * coefficient + 0.5
NORMAL_FROM = 0.2  # fraction of training after which surface normals must agree with depth
DISTORT_FROM = 0.1
MULTIVIEW_FROM = 0.3  # then: patches must match between neighbouring photos (PGSR's idea)
MULTIVIEW_PIXELS = 4096  # patch centres checked per step
NCC_WEIGHT, GEO_WEIGHT = 0.15, 0.03
PATCH_STEP = 1  # pixels between patch samples (7x7 samples)
FLAT_STD = 0.01  # patches with less grey contrast than this carry no evidence
MIN_TRACK, MAX_ERROR_PX = 3, 1.5  # COLMAP points trusted as measurements


@dataclass
class View:
    name: str
    w2c: np.ndarray  # 4x4 world -> camera
    K: np.ndarray  # 3x3
    w: int
    h: int


def text_model(dense: Path) -> Path:
    """The dense workspace's camera model as text (COLMAP writes binary; the pipeline converts
    it, CAST runs keep a `sparse_txt` copy)."""
    for name in ("sparse", "sparse_txt"):
        if (dense / name / "cameras.txt").exists():
            return dense / name
    raise RuntimeError(f"no text camera model in {dense}")


def read_views(sparse: Path) -> list[View]:
    """Cameras of an undistorted COLMAP text model (PINHOLE)."""
    cams = {}
    for ln in (sparse / "cameras.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            t = ln.split()
            if t[1] != "PINHOLE":
                raise RuntimeError(f"camera model is {t[1]}, expected PINHOLE")
            cams[t[0]] = (int(t[2]), int(t[3]), *map(float, t[4:8]))
    views = []
    for tok, name, _ in iter_images((sparse / "images.txt").read_text()):
        w2c = np.eye(4)
        w2c[:3, :3] = quat_to_rot(*map(float, tok[1:5]))
        w2c[:3, 3] = [float(v) for v in tok[5:8]]
        w, h, fx, fy, cx, cy = cams[tok[8]]
        views.append(View(name, w2c, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), w, h))
    return sorted(views, key=lambda v: v.name)


def read_points(sparse: Path) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP's sparse points (start positions for the gaussians): xyz, rgb 0..1."""
    xyz, rgb = [], []
    for ln in (sparse / "points3D.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            t = ln.split()
            xyz.append([float(v) for v in t[1:4]])
            rgb.append([int(v) / 255 for v in t[4:7]])
    return np.array(xyz).reshape(-1, 3), np.array(rgb).reshape(-1, 3)


def good_points(sparse: Path) -> dict[int, np.ndarray]:
    """COLMAP's well-measured points: seen in 3+ photos, small reprojection error. id -> xyz."""
    out = {}
    for ln in (sparse / "points3D.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            t = ln.split()
            if float(t[7]) <= MAX_ERROR_PX and (len(t) - 8) // 2 >= MIN_TRACK:
                out[int(t[0])] = np.array([float(v) for v in t[1:4]])
    return out


def centres_of(views: list[View]) -> np.ndarray:
    return np.array([-v.w2c[:3, :3].T @ v.w2c[:3, 3] for v in views])


def ssim(a, b):
    """Structural similarity of two [3, H, W] images (11 px gaussian window)."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()

    def blur(x):
        return F.conv2d(x[None], win, padding=5, groups=3)[0]

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
    return (((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean()


class Photos:
    """The photos (uint8, CPU) and object masks (bool, or None in the ground)."""

    def __init__(self, dense: Path, views: list[View], use_masks: bool):
        from PIL import Image

        self.images, self.masks = [], []
        for v in views:
            with Image.open(dense / "images" / v.name) as im:
                self.images.append(np.asarray(im.convert("RGB")))
            if use_masks:
                with Image.open(dense / "masks" / (v.name + ".png")) as m:
                    self.masks.append(np.asarray(m.convert("L")) > 127)
        if not use_masks:
            self.masks = [None] * len(views)


def render(params, view_t, sh_degree, background=None, distloss=False):
    """(colours+expected depth, alphas, normals, normals from depth, distortion, median depth,
    meta) for one camera."""
    import torch
    from gsplat import rasterization_2dgs

    viewmat, K, w, h = view_t
    return rasterization_2dgs(
        params["means"], params["quats"], torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), torch.cat([params["sh0"], params["shN"]], 1),
        viewmat[None], K[None], w, h, sh_degree=sh_degree, render_mode="RGB+ED",
        distloss=distloss, backgrounds=background)


def neighbours(views: list[View], k: int = 6) -> list[list[int]]:
    """For each photo, up to k others looking the same way (3-45 degrees apart), closest first."""
    f = np.array([v.w2c[2, :3] for v in views])  # viewing directions
    out = []
    for i in range(len(views)):
        angle = np.degrees(np.arccos(np.clip(f @ f[i], -1, 1)))
        ok = np.flatnonzero((angle > 3) & (angle < 45))
        out.append(ok[np.argsort(angle[ok])][:k].tolist())
    return out


def grey(image):
    import torch

    return (torch.from_numpy(image).cuda().float() @ torch.tensor([0.299, 0.587, 0.114], device="cuda")) / 255


def multiview(colours, normals, alphas, cam, cam_n, grey_r, grey_n, depth_n, mask, rng,
              step=PATCH_STEP, flat_std=FLAT_STD):
    """Multi-view consistency of the rendered surface (as in PGSR): each sampled pixel's depth
    and normal give a local plane; a 7x7 patch warped through it into a neighbouring photo
    should look the same there (normalised cross-correlation), and the point should land where
    the neighbour's own depth puts it (forward-backward reprojection, in pixels). Pixels that
    fail that check (hidden in the neighbour) are left out of the patch term."""
    import torch
    import torch.nn.functional as F

    viewmat, K, w, _ = cam
    viewmat_n, K_n, w_n, h_n = cam_n
    depth = colours[0, ..., 3]
    ok = (alphas[0, ..., 0] > 0.5) & (depth > 0)
    if mask is not None:
        ok &= mask
    b = 3 * step + 1
    ok[:b], ok[-b:], ok[:, :b], ok[:, -b:] = False, False, False, False
    idx = ok.reshape(-1).nonzero()[:, 0]
    if len(idx) < 100:
        return None
    idx = idx[torch.randint(len(idx), (MULTIVIEW_PIXELS,), device=idx.device, generator=rng)]
    v, u = idx // w, idx % w
    one = torch.ones_like(u, dtype=torch.float32)
    pix = torch.stack([u + 0.5, v + 0.5, one], -1)
    X = (pix @ torch.linalg.inv(K).T) * depth.reshape(-1)[idx, None]  # camera frame
    n = F.normalize(normals[0].reshape(-1, 3)[idx] @ viewmat[:3, :3].T, dim=-1)
    n = torch.where((n * X).sum(-1, keepdim=True) > 0, -n, n)  # facing the camera
    d = (n * X).sum(-1)  # plane: n.X = d (d < 0)
    T = viewmat_n @ torch.linalg.inv(viewmat)
    R, t = T[:3, :3], T[:3, 3]

    # forward-backward reprojection through the neighbour's depth
    Xn = X @ R.T + t
    qn = Xn @ K_n.T
    qn = qn[:, :2] / qn[:, 2:].clamp_min(1e-6)
    inside = (Xn[:, 2] > 0) & (qn[:, 0] > 0) & (qn[:, 0] < w_n) & (qn[:, 1] > 0) & (qn[:, 1] < h_n)
    norm = torch.stack([qn[:, 0] / w_n * 2 - 1, qn[:, 1] / h_n * 2 - 1], -1)
    dn = F.grid_sample(depth_n[None, None], norm[None, None], mode="nearest",
                       align_corners=False)[0, 0, 0]
    back = torch.cat([qn, one[:, None]], -1) @ torch.linalg.inv(K_n).T * dn[:, None]
    back = (back - t) @ R  # into the reference camera
    back = back @ K.T
    back = back[:, :2] / back[:, 2:].clamp_min(1e-6)
    err = (back - pix[:, :2]).norm(dim=-1)
    seen = inside & (dn > 0) & (err < 1.0)
    geo = err[inside & (dn > 0)].clamp(max=5).mean() if (inside & (dn > 0)).any() else None

    # patches through the plane's homography
    H = K_n @ (R[None] + t[None, :, None] * n[:, None, :] / d[:, None, None]) @ torch.linalg.inv(K)
    if step > 1:  # spaced samples: average the pixels between them (no aliasing of noise)
        grey_r = F.avg_pool2d(grey_r[None, None], step, 1, step // 2)[0, 0, :grey_r.shape[0], :grey_r.shape[1]]
        grey_n = F.avg_pool2d(grey_n[None, None], step, 1, step // 2)[0, 0, :grey_n.shape[0], :grey_n.shape[1]]
    r = torch.arange(-3, 4, device=u.device) * step
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    pu, pv = u[:, None] + dx.reshape(-1), v[:, None] + dy.reshape(-1)  # [N, 49]
    ref = grey_r[pv, pu]
    pts = torch.stack([pu + 0.5, pv + 0.5, torch.ones_like(pu, dtype=torch.float32)], -1)
    q = torch.einsum("nij,npj->npi", H, pts)
    q = q[..., :2] / q[..., 2:].clamp_min(1e-6)
    grid = torch.stack([q[..., 0] / w_n * 2 - 1, q[..., 1] / h_n * 2 - 1], -1)
    nb = F.grid_sample(grey_n[None, None], grid[None], align_corners=False)[0, 0]
    ref_c, nb_c = ref - ref.mean(1, keepdim=True), nb - nb.mean(1, keepdim=True)
    var_r, var_n = (ref_c ** 2).sum(1), (nb_c ** 2).sum(1)
    ncc = (ref_c * nb_c).sum(1) / (var_r * var_n + 1e-8).sqrt()
    use = seen & (var_r > 49 * flat_std ** 2)  # flat patches carry no evidence
    photo = ((1 - ncc) / 2)[use].mean() if use.any() else None
    return photo, geo


def train(dense: Path, use_masks: bool, steps: int = STEPS, progress=lambda f: None,
          normal_weight: float = 0.05, seed: int = 0, use_multiview: bool = True,
          ncc_weight: float = NCC_WEIGHT,
          patch_step: int = PATCH_STEP, flat_std: float = FLAT_STD):
    """Train the splat. Returns (params, photos, cameras, report)."""
    import torch
    from gsplat.strategy import DefaultStrategy

    dev = "cuda"
    torch.manual_seed(seed)
    views = read_views(text_model(dense))
    xyz, rgb = read_points(text_model(dense))
    photos = Photos(dense, views, use_masks)
    centres = centres_of(views)
    middle = centres.mean(0)
    scene = float(np.linalg.norm(centres - middle, axis=1).max()) * 1.1
    keep = np.linalg.norm(xyz - middle, axis=1) < scene  # stray points far outside the ring
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) < 100:
        raise PipelineError("Too few points were found to start the model. Record closer to "
                            "the object, in more light, and walk slower.")
    held = [i for i in range(len(views)) if i % HOLD_OUT == HOLD_OUT // 2]
    trained = [i for i in range(len(views)) if i not in held]

    pts = torch.tensor(xyz, dtype=torch.float32, device=dev)
    nn = torch.cdist(pts, pts).topk(4, largest=False).values[:, 1:].mean(1).clamp_min(1e-7)
    n = len(pts)
    params = torch.nn.ParameterDict({
        "means": pts,
        "scales": torch.log(nn)[:, None].repeat(1, 3),
        "quats": torch.rand(n, 4, device=dev),
        "opacities": torch.logit(torch.full((n,), 0.1, device=dev)),
        "sh0": ((torch.tensor(rgb, dtype=torch.float32, device=dev) - 0.5) / SH0)[:, None],
        "shN": torch.zeros(n, 15, 3, device=dev),
    }).requires_grad_(True)
    rates = {"means": 1.6e-4 * scene, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
             "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = {k: torch.optim.Adam([{"params": params[k], "lr": r, "name": k}], eps=1e-15)
           for k, r in rates.items()}
    decay = torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1 / steps))
    strategy = DefaultStrategy(key_for_gradient="gradient_2dgs", refine_stop_iter=steps // 2,
                               reset_every=3000, verbose=False)
    strategy.check_sanity(params, opt)
    sstate = strategy.initialize_state(scene_scale=scene)

    cams = cameras(views)
    near = neighbours(views)
    near = [[j for j in js if j not in held] for js in near]  # held-out photos stay unseen
    trng = torch.Generator(device=dev).manual_seed(seed)
    report_mv = {"photo": [], "geo": []}
    rng = np.random.default_rng(seed)
    order: list[int] = []
    start = time.time()
    for step in range(steps):
        if not order:
            order = list(rng.permutation(trained))
        i = order.pop()
        img = torch.from_numpy(photos.images[i]).to(dev, non_blocking=True).float() / 255
        bg = torch.rand(3, device=dev)
        if photos.masks[i] is not None:  # off the object: see-through (the random background)
            m = torch.from_numpy(photos.masks[i]).to(dev)[..., None].float()
            img = img * m + bg * (1 - m)
        colours, alphas, normals, normals_d, distort, _, info = render(
            params, cams[i], min(3, step // 1000), bg[None], distloss=step >= DISTORT_FROM * steps)
        out = colours[0, ..., :3]
        strategy.step_pre_backward(params, opt, sstate, step, info)
        l1 = (out - img).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(out.permute(2, 0, 1), img.permute(2, 0, 1)))
        if step >= NORMAL_FROM * steps:
            nd = normals_d.reshape(cams[i][3], cams[i][2], 3) * alphas[0].detach()
            loss = loss + normal_weight * (1 - (normals[0] * nd).sum(-1)).mean()
        if step >= DISTORT_FROM * steps:
            loss = loss + 1e-2 * distort.mean()
        if use_multiview and step >= MULTIVIEW_FROM * steps and near[i]:
            j = near[i][rng.integers(len(near[i]))]
            with torch.no_grad():
                depth_n = render(params, cams[j], 0)[0][0, ..., 3]
            mask = None if photos.masks[i] is None else torch.from_numpy(photos.masks[i]).to(dev)
            mv = multiview(colours, normals, alphas, cams[i], cams[j], grey(photos.images[i]),
                           grey(photos.images[j]), depth_n, mask, trng, patch_step, flat_std)
            if mv is not None:
                photo, geo = mv
                if photo is not None:
                    loss = loss + ncc_weight * photo
                    report_mv["photo"].append(float(photo))
                if geo is not None:
                    loss = loss + GEO_WEIGHT * geo
                    report_mv["geo"].append(float(geo))
        loss.backward()
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        decay.step()
        strategy.step_post_backward(params, opt, sstate, step, info, packed=False)
        if step % 200 == 0:
            progress(step / steps)
    report = {"views": len(views), "held_out": len(held), "gaussians": len(params["means"]),
              "steps": steps, "train_seconds": round(time.time() - start),
              "gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
    for k, vals in report_mv.items():  # how well the last tenth of training agreed across views
        if vals:
            report[f"multiview_{k}"] = round(float(np.mean(vals[-max(1, len(vals) // 10):])), 4)
    with torch.no_grad():
        report["psnr_held_out"] = round(psnr(params, [cams[i] for i in held],
                                             [photos.images[i] for i in held],
                                             [photos.masks[i] for i in held]), 2)
    return params, photos, cams, report


def psnr(params, cams, images, masks) -> float:
    """Mean PSNR (dB) of the splat against photos, on the object only when there are masks."""
    import torch

    out = []
    for cam, image, mask in zip(cams, images, masks):
        colours = render(params, cam, 3, torch.zeros(1, 3, device="cuda"))[0][0, ..., :3]
        img = torch.from_numpy(image).cuda().float() / 255
        if mask is not None:
            m = torch.from_numpy(mask).cuda()[..., None].float()
            img, colours = img * m, colours * m
        out.append(float(-10 * torch.log10(((colours.clamp(0, 1) - img) ** 2).mean())))
    return float(np.mean(out)) if out else 0.0


def depths(params, cams, photos, kind: str = "median"):
    """Depth per camera where the splat is solid (and on the object), 0 elsewhere: the median
    (the first surface crossed) or the expected (opacity-weighted mean) depth."""
    import torch

    out = []
    with torch.no_grad():
        for cam, mask in zip(cams, photos.masks):
            colours, alphas, _, _, _, median, _ = render(params, cam, 0)
            ok = alphas[0, ..., 0] > 0.5
            if mask is not None:
                ok &= torch.from_numpy(mask).cuda()
            d = median[0, ..., 0] if kind == "median" else colours[0, ..., 3]
            out.append(torch.where(ok, d, 0))
    return out


def cameras(views: list[View]):
    import torch

    return [(torch.tensor(v.w2c, dtype=torch.float32, device="cuda"),
             torch.tensor(v.K, dtype=torch.float32, device="cuda"), v.w, v.h) for v in views]


def load_ply(path: Path):
    """A splat written by save_ply, as training parameters."""
    import torch

    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    names = [ln.split()[-1] for ln in raw[:end].decode().splitlines() if ln.startswith("property")]
    data = np.frombuffer(raw[end:], dtype=[(n, "<f4") for n in names])

    def col(prefix):
        return np.stack([data[n] for n in names if n.startswith(prefix)], 1)

    n = len(data)
    rest = col("f_rest_").reshape(n, 3, -1).transpose(0, 2, 1)
    t = {"means": np.stack([data["x"], data["y"], data["z"]], 1),
         "scales": col("scale_"), "quats": col("rot_"), "opacities": data["opacity"].copy(),
         "sh0": col("f_dc_")[:, None], "shN": rest}
    return {k: torch.tensor(np.ascontiguousarray(v), device="cuda") for k, v in t.items()}


def fuse(cams, depth_maps, masks, lo, hi, cells=GRID):
    """Average signed distances from every depth map on a grid over [lo, hi] (GPU), then
    marching cubes. Off-object pixels (masks) count as empty space. Returns vertices, faces."""
    import torch
    from skimage.measure import marching_cubes

    cell = float((hi - lo).max()) / cells
    dims = [max(2, int(np.ceil(float(x) / cell))) for x in (hi - lo)]
    axes = [lo[i] + (torch.arange(dims[i], device=lo.device) + 0.5) * cell for i in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    tsdf = torch.zeros(len(grid), device=lo.device)
    weight = torch.zeros(len(grid), device=lo.device)
    band = 4 * cell
    for (viewmat, K, w, h), depth, mask in zip(cams, depth_maps, masks):
        pc = grid @ viewmat[:3, :3].T + viewmat[:3, 3]
        z = pc[:, 2]
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).floor().long()
        v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).floor().long()
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = v.clamp(0, h - 1) * w + u.clamp(0, w - 1)
        d = depth.reshape(-1)[idx]
        sdf = ((d - z) / band).clamp(-1, 1)
        seen = (d > 0) & (d - z > -band)  # in front of the surface or just behind it
        if mask is not None:
            off = ~torch.from_numpy(mask).cuda().reshape(-1)[idx]
            sdf = torch.where(off, torch.ones_like(sdf), sdf)
            seen |= off
        use = inside & seen
        tsdf = torch.where(use, (tsdf * weight + sdf) / (weight + 1), tsdf)
        weight += use.float()
    seen = (weight > 0).reshape(dims).cpu().numpy()
    vol = torch.where(weight > 0, tsdf, torch.ones_like(tsdf)).reshape(dims).cpu().numpy()
    if vol.min() >= 0 or vol.max() <= 0:
        raise PipelineError("No surface could be made from the model.")
    # Surface only between cells some camera saw: where the seen band behind a surface meets
    # unseen space there is no evidence of anything, so no second (phantom) surface is made.
    verts, faces, inward, _ = marching_cubes(vol, 0.0, mask=seen)  # normals: to lower values
    return verts * cell + (lo.cpu().numpy() + 0.5 * cell), outward(verts, faces, inward)


def outward(verts: np.ndarray, faces: np.ndarray, inward: np.ndarray) -> np.ndarray:
    """Faces wound so their normals point out of the object (texturing picks the photos facing
    a face; viewers light the front). `inward`: per-vertex normals into the object (the
    distance field's descent), which hold for open surfaces too."""
    a, b, c = (verts[faces[:, i]] for i in range(3))
    agree = np.einsum("ij,ij->i", np.cross(b - a, c - a), inward[faces].sum(1)) > 0
    return faces[:, ::-1].copy() if agree.mean() > 0.5 else faces


def bounds(params, centre: np.ndarray, radius: float):
    """The box round the solid part of the splat, within `radius` of `centre`."""
    import torch

    c = torch.tensor(centre, dtype=torch.float32, device="cuda")
    pts = params["means"].detach()
    solid = (torch.sigmoid(params["opacities"].detach()) > 0.5) & ((pts - c).norm(dim=1) < radius)
    pts = pts[solid]
    if len(pts) < 100:
        raise PipelineError("No surface could be made from the model.")
    lo, hi = torch.quantile(pts, 0.002, 0), torch.quantile(pts, 0.998, 0)
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def colours_at(params, vertices: np.ndarray) -> np.ndarray:
    """Each vertex takes the base colour of the nearest solid gaussian (texture fallback)."""
    import torch
    from scipy.spatial import cKDTree

    solid = (torch.sigmoid(params["opacities"]) > 0.3).cpu().numpy()
    means = params["means"].detach().cpu().numpy()[solid]
    base = (params["sh0"].detach().cpu().numpy()[solid, 0] * SH0 + 0.5).clip(0, 1)
    return base[cKDTree(means).query(vertices)[1]]


def save_ply(params, path: Path) -> None:
    """The splat in the usual 3DGS .ply layout (x y z, nx ny nz, f_dc, f_rest, opacity, scale,
    rot), readable by common splat viewers. 2DGS discs keep a tiny third scale."""
    p = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in params.items()}
    n = len(p["means"])
    rest = p["shN"].transpose(0, 2, 1).reshape(n, -1)  # per colour channel, as 3DGS stores them
    scales = p["scales"].copy()
    scales[:, 2] = np.log(1e-4) + np.log(np.exp(scales[:, :2]).min(1))  # flat disc
    q = p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
    cols = ([("x", p["means"][:, 0]), ("y", p["means"][:, 1]), ("z", p["means"][:, 2])]
            + [(f"n{a}", np.zeros(n, np.float32)) for a in "xyz"]
            + [(f"f_dc_{i}", p["sh0"][:, 0, i]) for i in range(3)]
            + [(f"f_rest_{i}", rest[:, i]) for i in range(rest.shape[1])]
            + [("opacity", p["opacities"])]
            + [(f"scale_{i}", scales[:, i]) for i in range(3)]
            + [(f"rot_{i}", q[:, i]) for i in range(4)])
    data = np.empty(n, dtype=[(name, "<f4") for name, _ in cols])
    for name, values in cols:
        data[name] = values
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n" + "".join(f"property float {name}\n" for name, _ in cols)
              + "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(data.tobytes())


def make(dense: Path, out: Path, use_masks: bool, centre: np.ndarray, radius: float,
         steps: int = STEPS, progress=lambda f: None, depth: str = "median", refuse: bool = False,
         **options):
    """Train, save the splat (out/splat.ply) and fuse the surface. Returns (vertices, faces,
    colours, report). `refuse`: skip training, fuse the splat already in `out`."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    if refuse:
        views = read_views(text_model(dense))
        params, photos, cams, report = (load_ply(out / "splat.ply"), Photos(dense, views, use_masks),
                                        cameras(views), {})
    else:
        params, photos, cams, report = train(
            dense, use_masks, steps, progress=lambda f: progress(0.9 * f), **options)
        save_ply(params, out / "splat.ply")
    t = time.time()
    lo, hi = bounds(params, centre, radius)
    verts, faces = fuse(cams, depths(params, cams, photos, depth), photos.masks, lo, hi)
    report["fuse_seconds"] = round(time.time() - t)
    colours = colours_at(params, verts)
    torch.cuda.empty_cache()
    progress(1.0)
    return verts, faces, colours, report


def main():
    import argparse
    import sys

    from castlive.recon import mesh as M

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dense", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--centre", help="x,y,z of what was filmed (default: where cameras look)")
    ap.add_argument("--radius", type=float, help="how far round it to make the surface")
    ap.add_argument("--progress", action="store_true", help="print 'progress <fraction>' lines")
    ap.add_argument("--check", action="store_true", help="measure the mesh (quality.py)")
    ap.add_argument("--normal-weight", type=float, default=0.05)
    ap.add_argument("--no-multiview", action="store_true")
    ap.add_argument("--ncc-weight", type=float, default=NCC_WEIGHT)
    ap.add_argument("--patch-step", type=int, default=PATCH_STEP)
    ap.add_argument("--flat-std", type=float, default=FLAT_STD)
    ap.add_argument("--depth", choices=["median", "expected"], default="median")
    ap.add_argument("--refuse", action="store_true", help="fuse the splat already in <out> again")
    ap.add_argument("--mesh-name", default="mesh.ply")
    a = ap.parse_args()
    views = read_views(text_model(a.dense))
    c = centres_of(views)
    centre = (np.array([float(x) for x in a.centre.split(",")]) if a.centre
              else M.look_at_point(c, np.array([v.w2c[2, :3] for v in views])))
    radius = a.radius or 0.8 * float(np.median(np.linalg.norm(c - centre, axis=1)))
    show = (lambda f: print(f"progress {f:.4f}", flush=True)) if a.progress else (lambda f: None)
    t = time.time()
    try:
        v, f, col, report = make(a.dense, a.out, not a.no_masks, centre, radius, a.steps, show,
                                 depth=a.depth, refuse=a.refuse, normal_weight=a.normal_weight,
                                 use_multiview=not a.no_multiview,
                                 ncc_weight=a.ncc_weight, patch_step=a.patch_step,
                                 flat_std=a.flat_std)
        m = M.from_arrays(v, f, col)
    except PipelineError as e:
        print(f"PipelineError: {e}", flush=True)
        sys.exit(2)
    M.save_ply(m, a.out / a.mesh_name)
    report["mesh"] = m.stats
    if a.check:
        from castlive.recon import quality

        report["quality"] = quality.measure(a.dense, m.vertices, m.faces, not a.no_masks)
    report["total_seconds"] = round(time.time() - t)
    (a.out / (Path(a.mesh_name).stem + ".json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
