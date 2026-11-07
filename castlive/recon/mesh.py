"""Surfaces: the mesh from the splat (`from_arrays`), or, from a dense point cloud, a trimmed
Poisson surface (`build`, CAST's way), and how either is placed for viewing.

Poisson reconstruction closes every surface, inventing shape where there was no data. CAST
trims those parts (low Poisson density = few points nearby) so unseen areas stay open instead
of being made up (PLAN.md: honest output).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from castlive.recon import PipelineError

MAX_POINTS = 3_000_000
MIN_POINTS = 20_000
TRIM_QUANTILE = 0.06  # drop the least supported 6% of the Poisson surface
TRIANGLES = 500_000
SIZE = 0.2  # metres: the longest side until a scale marker gives the real size (Phase 5)


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray  # sRGB 0..1
    normals: np.ndarray
    stats: dict


def look_at_point(centres: np.ndarray, forwards: np.ndarray) -> np.ndarray:
    """The point closest (least squares) to every camera's viewing ray: what was filmed."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(centres, forwards):
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.lstsq(A, b, rcond=None)[0]


def up_rotation(up: np.ndarray) -> np.ndarray:
    """A rotation taking `up` to +Y (glTF's up)."""
    u = up / np.linalg.norm(up)
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(u, y), float(u @ y)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def build(cloud: Path, centre: np.ndarray, radius: float, depth: int = 10) -> Mesh:
    """The trimmed surface, in the camera model's coordinates (so photos can be projected onto
    it); `placement` stands it upright for viewing."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pcd = o3d.io.read_point_cloud(str(cloud))
    n_fused = len(pcd.points)
    keep = np.flatnonzero(np.linalg.norm(np.asarray(pcd.points) - centre, axis=1) <= radius)
    pcd = pcd.select_by_index(keep)
    if len(pcd.points) > MAX_POINTS:
        pcd = pcd.random_down_sample(MAX_POINTS / len(pcd.points))
    if len(pcd.points) >= 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
    n_clean = len(pcd.points)
    if n_clean < MIN_POINTS:
        raise PipelineError(f"Only {n_clean:,} surface points were found (need {MIN_POINTS:,}). "
                            "Record closer to the object, in more light, and walk slower.")
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        pcd.orient_normals_consistent_tangent_plane(20)

    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, TRIM_QUANTILE))
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    if len(counts) > 1:  # keep the main piece: small shells form round stray points
        mesh.remove_triangles_by_mask(clusters != counts.argmax())
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(TRIANGLES)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise PipelineError("No surface could be made from the points.")

    v = np.asarray(mesh.vertices)
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    _, idx = cKDTree(pts).query(v, k=min(6, len(pts)))
    colors = cols[idx].mean(axis=1) if idx.ndim == 2 else cols[idx]

    mesh.compute_vertex_normals()
    edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    stats = {"points_fused": n_fused, "points_used": n_clean,
             "vertices": len(v), "triangles": len(mesh.triangles),
             "open": len(np.asarray(edges)) > 0, "scale_known": False}
    return Mesh(v, np.asarray(mesh.triangles), colors, np.asarray(mesh.vertex_normals), stats)


def from_arrays(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> Mesh:
    """A Mesh from raw triangles (e.g. marching cubes): the main piece only, at most TRIANGLES."""
    import open3d as o3d

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    raw = len(mesh.triangles)
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    if len(counts) > 1:  # keep the main piece: specks form round stray gaussians
        mesh.remove_triangles_by_mask(clusters != counts.argmax())
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(TRIANGLES)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise PipelineError("No surface could be made from the model.")
    mesh.compute_vertex_normals()
    edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    v = np.asarray(mesh.vertices)
    stats = {"triangles_raw": raw, "pieces": len(counts), "vertices": len(v),
             "triangles": len(mesh.triangles), "open": len(np.asarray(edges)) > 0,
             "scale_known": False}
    return Mesh(v, np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors),
                np.asarray(mesh.vertex_normals), stats)


def save_ply(m: Mesh, path: Path) -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                     o3d.utility.Vector3iVector(m.faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(m.colors)
    mesh.vertex_normals = o3d.utility.Vector3dVector(m.normals)
    o3d.io.write_triangle_mesh(str(path), mesh)


def load_ply(path: Path) -> Mesh:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    return Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                np.asarray(mesh.vertex_colors), np.asarray(mesh.vertex_normals), {})


def normals_for(vertices: np.ndarray, m: Mesh) -> np.ndarray:
    """Normals of the whole mesh `m` at `vertices` (copies of its vertices, e.g. the textured
    parts), so shading runs smoothly across the seams between parts."""
    from scipy.spatial import cKDTree

    return m.normals[cKDTree(m.vertices).query(vertices)[1]]


def placement(vertices: np.ndarray, centre: np.ndarray, up: np.ndarray):
    """Model coordinates -> upright, centred, a nominal size (the real scale is unknown until a
    marker card is used). Returns `place(vertices, normals) -> (vertices, normals)`, the same
    move for every part of the model."""
    R = up_rotation(up)
    v = (vertices - centre) @ R.T
    lo, hi = v.min(axis=0), v.max(axis=0)
    mid, scale = (lo + hi) / 2, SIZE / float((hi - lo).max())

    def place(vs, normals):
        return ((vs - centre) @ R.T - mid) * scale, normals @ R.T

    return place
