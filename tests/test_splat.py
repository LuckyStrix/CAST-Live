"""The splat stage's plain parts (no GPU): reading the model, and how the pipeline runs it."""

import numpy as np
import pytest

from castlive.recon import PipelineError, quality, run, splat

CAMERAS = "# c\n1 PINHOLE 100 80 90 90 50 40\n2 PINHOLE 60 50 90 90 30 25\n"
IMAGES = ("# i\n"
          "2 1 0 0 0 0 0 5 2 p1_000002.jpg\n10 20 7 30 40 8\n"
          "1 1 0 0 0 1 2 3 1 p1_000001.jpg\n\n")  # empty observations line
POINTS = ("# p\n"
          "7 0.1 0.2 0.3 255 0 0 0.5 1 0 2 0 3 1\n"  # seen 3 times, small error: good
          "8 1 1 1 0 255 0 2.5 1 0 2 0 3 1\n"  # error too big
          "9 2 2 2 0 0 255 0.2 1 0 2 0\n"  # seen twice
          "10 0 0 1 9 9 9 0.1 1 1 2 1 3 2\n")  # good


@pytest.fixture
def sparse(tmp_path):
    d = tmp_path / "sparse"
    d.mkdir()
    (d / "cameras.txt").write_text(CAMERAS)
    (d / "images.txt").write_text(IMAGES)
    (d / "points3D.txt").write_text(POINTS)
    return d


def test_reads_cameras_in_name_order(sparse):
    views = splat.read_views(sparse)
    assert [v.name for v in views] == ["p1_000001.jpg", "p1_000002.jpg"]
    a, b = views
    assert (a.w, a.h, b.w, b.h) == (100, 80, 60, 50)
    assert a.K[0, 2] == 50 and b.K[1, 2] == 25
    assert np.allclose(a.w2c[:3, 3], [1, 2, 3])
    assert np.allclose(splat.centres_of(views)[0], [-1, -2, -3])  # identity rotation


def test_reads_start_points_and_picks_the_good_ones(sparse):
    xyz, rgb = splat.read_points(sparse)
    assert xyz.shape == (4, 3) and np.allclose(rgb[0], [1, 0, 0])
    assert list(splat.good_points(sparse)) == [7, 10]
    assert np.allclose(quality.good_points(sparse), [[0.1, 0.2, 0.3], [0, 0, 1]])


def fake_python(tmp_path, monkeypatch, body: str):
    exe = tmp_path / "python"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    monkeypatch.setattr(run.sys, "executable", str(exe))


def test_run_splat_arguments_parse_with_a_negative_centre(tmp_path, monkeypatch):
    args = tmp_path / "args"
    fake_python(tmp_path, monkeypatch, f'printf "%s\\n" "$@" > {args}\necho "{{}}"\n')
    run.run_splat(tmp_path, tmp_path / "s", True, [-0.13, 0.4, -0.08], 2.5, tmp_path / "log")
    a = splat.parser().parse_args(args.read_text().splitlines()[2:])  # after -m <module>
    assert a.centre == "-0.13,0.4,-0.08" and a.radius == 2.5 and not a.no_masks


def test_run_splat_reports_progress_and_result(tmp_path, monkeypatch):
    fake_python(tmp_path, monkeypatch, "echo 'progress 0.5'\necho 'training'\n"
                                       "echo '{\"gaussians\": 5}'\necho 'UserWarning: x'\n")
    seen = []
    out = run.run_splat(tmp_path, tmp_path / "s", True, [0, 0, 0], 1.0, tmp_path / "log",
                        seen.append)
    assert out == {"gaussians": 5} and seen == [0.5]
    assert "training" in (tmp_path / "log").read_text()


def test_run_splat_passes_on_errors_the_user_can_act_on(tmp_path, monkeypatch):
    fake_python(tmp_path, monkeypatch, "echo 'PipelineError: Too few points'\nexit 2\n")
    with pytest.raises(PipelineError, match="Too few points"):
        run.run_splat(tmp_path, tmp_path / "s", False, [0, 0, 0], 1.0, tmp_path / "log")
    fake_python(tmp_path, monkeypatch, "echo 'Traceback: CUDA out of memory'\nexit 1\n")
    with pytest.raises(RuntimeError, match="exit 1"):
        run.run_splat(tmp_path, tmp_path / "s", False, [0, 0, 0], 1.0, tmp_path / "log")


def test_faces_point_outward():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    out = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])  # a tetrahedron, outward
    inward = 0.25 - v  # towards its middle
    assert (splat.outward(v, out, inward) == out).all()
    assert (splat.outward(v, out[:, ::-1], inward) == out).all()
    one = out[3:]  # a single open face
    assert (splat.outward(v, one[:, ::-1], inward) == one).all()


def test_viewer_file_has_placed_discs(tmp_path):
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(45)] + ["opacity", "scale_0", "scale_1", "scale_2"]
             + [f"rot_{i}" for i in range(4)])
    rows = np.zeros((2, len(names)), np.float32)
    col = {n: i for i, n in enumerate(names)}
    rows[:, col["x"]] = [1, 5]
    rows[:, col["opacity"]] = [3, -6]  # the second is nearly transparent: left out
    rows[0, col["scale_0"]], rows[0, col["scale_1"]] = np.log(0.2), np.log(0.1)
    rows[0, col["rot_3"]] = rows[0, col["rot_0"]] = np.sqrt(0.5)  # 90 degrees about z
    rows[0, col["f_dc_0"]] = 0.5 / splat.SH0  # red channel 1.0
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
              + "".join(f"property float {n}\n" for n in names) + "end_header\n")
    ply = tmp_path / "splat.ply"
    ply.write_bytes(header.encode() + rows.tobytes())

    def place(v, n):  # move up by 10, double the size
        return (v + [0, 0, 10]) * 2, n

    assert splat.write_view(ply, tmp_path / "splat.bin", place) == 1
    d = np.frombuffer((tmp_path / "splat.bin").read_bytes(), np.float32).reshape(-1, 16)
    assert np.allclose(d[0, 0:3], [2, 0, 20]) and d[0, 3] > 0.9
    assert np.allclose(d[0, 4:7], [0, 0.4, 0], atol=1e-6)  # x axis turned onto y, doubled
    assert np.allclose(d[0, 8:11], [-0.2, 0, 0], atol=1e-6)
    assert np.isclose(d[0, 7], 1.0) and np.isclose(d[0, 11], 0.5)


def test_exposure_gains_even_out_the_photos(tmp_path):
    pytest.importorskip("open3d")
    from PIL import Image

    from castlive.recon import texture

    verts = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])  # a square facing +z
    views = []
    for k, grey in enumerate([100, 150, 120]):  # the same surface, three exposures
        w2c = np.diag([1.0, -1.0, -1.0, 1.0])  # above the square, looking down
        w2c[:3, 3] = [0.1 * k, 0, 3]
        views.append(splat.View(f"v{k}.jpg", w2c, np.array([[200, 0, 200], [0, 200, 200], [0, 0, 1]], float), 400, 400))
        Image.new("RGB", (400, 400), (grey, grey, grey)).save(tmp_path / f"v{k}.jpg")
    g = texture.exposure_gains(tmp_path, views, verts, faces)
    assert np.allclose([g[f"v{k}.jpg"][0] for k in range(3)], [1.2, 0.8, 1.0], atol=0.02)
