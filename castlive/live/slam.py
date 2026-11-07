"""MASt3R-SLAM (CC BY-NC-SA 4.0, research use) fed one frame at a time.

Its WSL branch runs tracking and the backend optimisation in one process. We drive the same
steps as its main.py, calling its own functions (`main.run_backend` uses module globals, which we
set), so none of its code is copied here. Needs the `live` image (docker/Dockerfile.live).
"""

import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CASTLIVE_MAST3R_SLAM", "/opt/m3slam"))
CONF_MIN = 1.5  # MASt3R-SLAM's own default for saving points
SIZE = 512
OVERLAY_STEP = 4  # one overlay cell per 4x4 pixels of the live frame


class SeenFrom:
    """Which directions each bit of surface has been filmed from: the space is cut into small
    cubes, the directions into 32 even bins, and the (cube, bin) pairs seen are kept, sorted,
    on the GPU. Coverage of a point = how many bins its cube has been seen from."""

    BITS = 19  # per axis: +-262k cubes

    def __init__(self, torch, device, cube: float):
        self.torch, self.cube = torch, cube
        n = 32  # directions: a Fibonacci sphere
        i = torch.arange(n, dtype=torch.float32, device=device) + 0.5
        z, a = 1 - 2 * i / n, i * np.pi * (3 - 5 ** 0.5)
        r = (1 - z * z).sqrt()
        self.dirs = torch.stack([r * a.cos(), r * a.sin(), z], 1)
        self.pairs = torch.zeros(0, dtype=torch.int64, device=device)

    def _keys(self, pts):
        off = 1 << (self.BITS - 1)
        q = (pts / self.cube).floor().long().clamp(-off, off - 1) + off
        return (q[:, 0] << (2 * self.BITS)) | (q[:, 1] << self.BITS) | q[:, 2]

    def add(self, pts, centre) -> None:
        """Points (N, 3) seen from a camera at `centre` (3,)."""
        d = pts - centre
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        b = (d @ self.dirs.T).argmax(1)
        new = self.torch.unique(self._keys(pts) * 32 + b)
        self.pairs = self.torch.unique(self.torch.cat([self.pairs, new]))

    def count(self, pts):
        k = self._keys(pts) * 32
        return (self.torch.searchsorted(self.pairs, k + 32)
                - self.torch.searchsorted(self.pairs, k))


class Slam:
    def __init__(self, device: str = "cuda:0"):
        sys.path.insert(0, str(ROOT))
        os.chdir(ROOT)  # its checkpoints are found relative to the working folder
        import main as m3  # MASt3R-SLAM's main.py (guarded by __main__)
        import torch
        import torch.multiprocessing as mp
        from mast3r_slam.config import load_config
        from mast3r_slam.mast3r_utils import load_mast3r

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        load_config(str(ROOT / "config" / "base.yaml"))
        self.torch, self.m3, self.device = torch, m3, device
        self.manager = mp.Manager()
        self.model = load_mast3r(device=device)
        self.n = 0

    def reset(self, first: np.ndarray) -> None:
        """A new pass: forget everything tracked so far. `first`: an RGB uint8 frame."""
        from mast3r_slam.frame import SharedKeyframes, SharedStates
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import load_retriever, resize_img
        from mast3r_slam.tracker import FrameTracker

        h, w = resize_img(first.astype(np.float32) / 255, SIZE)["img"][0].shape[1:]
        self.shape = first.shape
        self.keyframes = SharedKeyframes(self.manager, h, w)
        self.states = SharedStates(self.manager, h, w)
        self.tracker = FrameTracker(self.model, self.keyframes, self.device)
        self.m3.factor_graph = FactorGraph(self.model, self.keyframes, None, self.device)
        self.m3.retrieval_database = load_retriever(self.model)
        self.n = 0
        self.track: list = []  # per placed frame: (its keyframe, pose relative to it)
        self.seen = None  # SeenFrom, made with the first frame (it sets the cube size)
        self.last = None  # the last placed frame

    def add(self, rgb: np.ndarray) -> bool:
        """Track one RGB uint8 frame (same size as the first). True if it was placed."""
        import lietorch
        from mast3r_slam.frame import Mode, create_frame
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        if rgb.shape != self.shape:
            return False
        img = rgb.astype(np.float32) / 255
        states, keyframes = self.states, self.keyframes
        mode = states.get_mode()
        T_WC = (lietorch.Sim3.Identity(1, device=self.device) if self.n == 0
                else states.get_frame().T_WC)
        frame = create_frame(self.n, img, T_WC, img_size=SIZE, device=self.device)
        self.n += 1
        placed, new_kf = True, False
        ref = len(keyframes) - 1  # the tracker places a frame against the last keyframe
        if mode == Mode.INIT:
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            self.track.append((0, lietorch.Sim3.Identity(1, device=self.device).data))
            self._seen(frame)
            return True
        if mode == Mode.TRACKING:
            new_kf, _, try_reloc = self.tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
                placed = False
            states.set_frame(frame)
        else:  # RELOC: lost; try to find our place again among the keyframes
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            placed = False
        if placed:  # remember it relative to its keyframe, which the backend keeps correcting
            kf = lietorch.Sim3(keyframes.T_WC[ref])
            self.track.append((ref, (kf.inv() * frame.T_WC).data))
            self._seen(frame)
        if new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
        self.m3.run_backend(states, keyframes)
        return placed and states.get_mode() == Mode.TRACKING

    def _world(self, frame, step: int):
        """The frame's confident points in the world, every `step` pixels: (points, mask)."""
        h, w = (int(v) for v in frame.img_shape.flatten()[:2])
        X = frame.X_canon.reshape(h, w, 3)[::step, ::step]
        ok = (frame.get_average_conf().reshape(h, w)[::step, ::step] > CONF_MIN)
        pts = frame.T_WC.act(X.reshape(-1, 3)).reshape(-1, 3)
        return pts, ok.reshape(-1), X.shape[:2]

    def _seen(self, frame) -> None:
        self.last = frame
        pts, ok, _ = self._world(frame, 2)
        if self.seen is None:
            depth = frame.X_canon.norm(dim=-1).median().item()
            self.seen = SeenFrom(self.torch, self.device, cube=depth / 80)
        centre = frame.T_WC.data.reshape(-1)[:3]
        self.seen.add(pts[ok], centre)

    def overlay(self) -> bytes:
        """How well each part of the last placed frame has been covered: uint16 width, height,
        then one byte per cell (row by row): 0 = no surface found, else directions seen from."""
        if self.last is None:
            return struct.pack("<HH", 0, 0)
        pts, ok, (gh, gw) = self._world(self.last, OVERLAY_STEP)
        n = self.seen.count(pts).clamp(0, 255) * ok
        return struct.pack("<HH", gw, gh) + n.to(self.torch.uint8).cpu().numpy().tobytes()

    def cameras(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every placed frame's centre, viewing direction and image-up direction, from its
        keyframe's latest (optimised) pose: positions as tracked drift until corrected."""
        import lietorch
        from scipy.spatial.transform import Rotation

        if not self.track:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
        torch = self.torch
        refs = torch.tensor([r for r, _ in self.track], device=self.device)
        rel = lietorch.Sim3(torch.cat([d.reshape(1, -1) for _, d in self.track]))
        with self.keyframes.lock:
            kf = lietorch.Sim3(self.keyframes.T_WC[refs].reshape(len(self.track), -1))
        d = (kf * rel).data.detach().cpu().numpy()
        R = Rotation.from_quat(d[:, 3:7]).as_matrix()
        return d[:, :3], R[:, :, 2], -R[:, :, 1]

    def points(self, limit: int) -> tuple[np.ndarray, np.ndarray]:
        """Up to `limit` confident points from the keyframes (world, RGB uint8)."""
        xyz, rgb = [], []
        with self.keyframes.lock:
            for i in range(len(self.keyframes)):
                kf = self.keyframes[i]
                ok = (kf.get_average_conf().reshape(-1) > CONF_MIN).cpu().numpy()
                xyz.append(kf.T_WC.act(kf.X_canon).cpu().numpy().reshape(-1, 3)[ok])
                rgb.append((kf.uimg.cpu().numpy().reshape(-1, 3)[ok] * 255).astype(np.uint8))
        if not xyz:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
        xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
        if len(xyz) > limit:
            keep = np.random.default_rng(0).choice(len(xyz), limit, replace=False)
            xyz, rgb = xyz[keep], rgb[keep]
        return xyz, rgb
