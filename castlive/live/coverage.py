"""Which directions round the object have been filmed: 16 sectors of a circle round the
upright axis, plus the range of heights. Shown on the phone while filming."""

import numpy as np

from castlive.recon.mesh import look_at_point, up_rotation

SECTORS = 16


def frame(centres: np.ndarray, forwards: np.ndarray, ups: np.ndarray, fallback: np.ndarray):
    """(centre, R, radius): the filmed point, a rotation taking the cameras' average up to +Y,
    and the median camera distance. `fallback` is used for the centre while the cameras'
    viewing rays don't yet cross well (all looking the same way)."""
    d = forwards / np.linalg.norm(forwards, axis=1, keepdims=True)
    spread = np.linalg.eigvalsh(sum(np.eye(3) - np.outer(v, v) for v in d) / len(d))[0]
    centre = look_at_point(centres, d) if spread > 0.02 else fallback  # ~8 degrees of spread
    up = ups.mean(axis=0)
    radius = float(np.median(np.linalg.norm(centres - centre, axis=1))) or 1.0
    return centre, up_rotation(up / np.linalg.norm(up)), radius


def measure(centres: np.ndarray, centre: np.ndarray, R: np.ndarray, radius: float) -> dict:
    """Sector counts and height range of the cameras, seen from the object."""
    v = (centres - centre) @ R.T / radius
    az = np.degrees(np.arctan2(v[:, 2], v[:, 0])) % 360
    el = np.degrees(np.arctan2(v[:, 1], np.linalg.norm(v[:, [0, 2]], axis=1)))
    counts = np.bincount((az // (360 / SECTORS)).astype(int) % SECTORS, minlength=SECTORS)
    return {"sectors": counts.tolist(), "covered": round(float((counts > 0).mean()), 3),
            "elevation": [round(float(el.min())), round(float(el.max()))]}


def normalise(points: np.ndarray, centre: np.ndarray, R: np.ndarray, radius: float) -> np.ndarray:
    """Points in the object's frame: centred, upright (+Y), one orbit radius = 1."""
    return (points - centre) @ R.T / radius
