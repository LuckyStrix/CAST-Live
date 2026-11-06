"""The live hand-off between the public capture site and the GPU `live` service.

They share files only (the capture site has no network path to the GPU containers):

    /data/live/<capture id>/
        in/<pass>_<seq>.jpg   frames from the phone (capture writes, live reads then deletes)
        control.json          {"pass": n, "ended": bool}  (capture writes)
        out/state.json        tracking state and coverage  (live writes)
        out/points.bin        preview point cloud, see `pack_points`  (live writes)
        out/done              the session is over  (live writes)
    /data/live/active         the capture being tracked; the reconstructor waits while it exists
"""

import json
import os
import struct
from pathlib import Path

import numpy as np

POINT_UNITS = 8000  # int16 units per orbit radius: points up to 4 radii from the centre fit


def root(data: Path) -> Path:
    return data / "live"


def session(data: Path, cid: str) -> Path:
    return root(data) / cid


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    write_atomic(path, json.dumps(obj, separators=(",", ":")).encode())


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def frame_name(pass_: int, seq: int) -> str:
    return f"{pass_}_{seq:06d}.jpg"


def pending(sess: Path) -> list[tuple[int, int, Path]]:
    """Frames waiting, oldest first: (pass, seq, path)."""
    out = []
    for p in (sess / "in").glob("*.jpg"):
        try:
            a, b = p.stem.split("_")
            out.append((int(a), int(b), p))
        except ValueError:
            continue
    return sorted(out)


def pack_points(xyz: np.ndarray, rgb: np.ndarray) -> bytes:
    """uint32 count, then count*3 int16 positions (orbit radii * POINT_UNITS), then count*3
    uint8 colours. Little-endian; easy to read in a browser with typed arrays."""
    q = np.clip(np.round(np.asarray(xyz, np.float64) * POINT_UNITS), -32767, 32767).astype("<i2")
    c = np.asarray(rgb, np.uint8).reshape(-1, 3)
    return struct.pack("<I", len(q)) + q.tobytes() + c.tobytes()


def unpack_points(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    (n,) = struct.unpack_from("<I", raw)
    xyz = np.frombuffer(raw, "<i2", 3 * n, 4).reshape(-1, 3) / POINT_UNITS
    rgb = np.frombuffer(raw, np.uint8, 3 * n, 4 + 6 * n).reshape(-1, 3)
    return xyz, rgb
