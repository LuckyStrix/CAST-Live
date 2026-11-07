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
          "9 2 2 2 0 0 255 0.2 1 0 2 0\n")  # seen twice


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
    assert xyz.shape == (3, 3) and np.allclose(rgb[0], [1, 0, 0])
    assert np.allclose(quality.good_points(sparse), [[0.1, 0.2, 0.3]])


def fake_python(tmp_path, monkeypatch, body: str):
    exe = tmp_path / "python"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    monkeypatch.setattr(run.sys, "executable", str(exe))


def test_run_splat_reports_progress_and_result(tmp_path, monkeypatch):
    fake_python(tmp_path, monkeypatch, "echo 'progress 0.5'\necho 'training'\n"
                                       "echo '{\"gaussians\": 5}'\n")
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
