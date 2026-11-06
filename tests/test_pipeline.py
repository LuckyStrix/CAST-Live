import json
import struct

import numpy as np
import pytest

from castlive.common import db
from castlive.recon import PipelineError, colmap, frames, glb, mesh
from castlive.worker import loop


def test_passes_skip_the_turn_over():
    markers = [{"t": 0, "event": "start"}, {"t": 30, "event": "pass1_end"},
               {"t": 40, "event": "pass2_start"}, {"t": 70, "event": "stop"}]
    assert frames.passes(markers, 70.5, "object") == [(1, 0, 29.3), (2, 40.7, 70)]
    # in the ground: the pause between loops is left out too, with a smaller margin
    assert frames.passes(markers, 70.5, "in-situ") == [(1, 0, 29.7), (2, 40.3, 70)]
    # no markers (e.g. an old capture): the whole video
    assert frames.passes([], 12, "object") == [(1, 0, 12)]
    # a pass too short to use is dropped, and the other keeps its real number
    short = [{"t": 2, "event": "pass1_end"}, {"t": 5, "event": "pass2_start"}]
    assert frames.passes(short, 30, "object") == [(2, 5.7, 30)]


def test_split_and_select():
    assert frames.split_count([(1, 0, 30), (2, 40, 70)], 160) == [80, 80]
    assert frames.split_count([(1, 0, 10), (2, 10, 100)], 100) == [20, 90]  # at least 20 each
    assert frames.select_sharpest([1, 9, 2, 3, 8, 1], 2) == [1, 4]
    assert frames.select_sharpest([1, 2], 5) == [0, 1]


def test_pairs_link_far_frames():
    names = [f"f{i}" for i in range(40)]
    pairs = set(colmap.match_pairs(names, window=2, stride=8))
    assert ("f1", "f2") in pairs and ("f1", "f3") in pairs and ("f1", "f4") not in pairs
    assert ("f5", "f32") in pairs  # every 8th frame links distant parts


def test_parse_images_txt():
    text = ("# header\n"
            "1 1 0 0 0 0 0 5 1 p1_000001.jpg\n\n"
            "2 0 0 1 0 1 0 0 1 p2_000001.jpg\n1 2 3\n")
    names, centres, ups, fwds = colmap.parse_images_txt(text)
    assert names == ["p1_000001.jpg", "p2_000001.jpg"]
    np.testing.assert_allclose(centres[0], [0, 0, -5])
    np.testing.assert_allclose(ups[0], [0, -1, 0])
    np.testing.assert_allclose(fwds[0], [0, 0, 1])


def test_look_at_and_up():
    # cameras on a circle all looking at (1, 2, 3)
    target = np.array([1.0, 2.0, 3.0])
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    centres = target + np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], 1) * 2
    np.testing.assert_allclose(mesh.look_at_point(centres, target - centres), target, atol=1e-9)
    for up in ([0, 0, 1], [1, 1, 0], [0, -1, 0], [0, 1, 0]):
        R = mesh.up_rotation(np.array(up, float))
        np.testing.assert_allclose(R @ (np.array(up) / np.linalg.norm(up)), [0, 1, 0], atol=1e-9)
        assert abs(np.linalg.det(R) - 1) < 1e-9


def test_glb_layout(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    path = tmp_path / "t.glb"
    glb.write_glb(path, v, np.array([[0, 1, 2]]), colors_srgb=np.full((3, 3), 0.5))
    data = path.read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    jlen = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20:20 + jlen])
    assert doc["accessors"][0]["max"] == [1, 1, 0]
    with pytest.raises(ValueError):
        glb.write_glb(path, v, np.array([[0, 1, 5]]))


def _checked(cfg, token):
    cid = "6f1c2c1e-6b1f-4d7e-9a43-0b8c1f2d3e4a"
    with db.connect(cfg.db) as con:
        con.execute(
            "INSERT INTO captures (id, project_id, invite, mode, status, mime, meta, created,"
            " updated) VALUES (?, 1, ?, 'object', 'checked', 'video/mp4', ?, 0, 0)",
            (cid, token, json.dumps({"video": {"duration": 30}})))
    return cid


def _row(cfg, cid):
    with db.connect(cfg.db) as con:
        return con.execute("SELECT * FROM captures WHERE id = ?", (cid,)).fetchone()


def test_worker_reconstructs_and_reports(cfg, token, monkeypatch):
    cid = _checked(cfg, token)
    stages = []

    def fake(video, meta, mode, run_dir, progress, wait):
        progress("frames", 0.5)
        row = _row(cfg, cid)
        stages.append(row["stage"])
        p = json.loads(row["progress"])
        assert p["stage"] == "frames" and 0 < p["fraction"] < 0.1 and p["eta_seconds"] > 0
        return {"poses": {"placed": 10}}

    monkeypatch.setattr("castlive.recon.run.process", fake)
    assert not loop.step_check(cfg)  # the upload checker leaves checked captures alone
    assert loop.step_reconstruct(cfg)
    row = _row(cfg, cid)
    assert stages == ["frames"] and row["status"] == "done"
    assert json.loads(row["meta"])["report"]["poses"]["placed"] == 10


def test_worker_reports_pipeline_errors(cfg, token, monkeypatch):
    cid = _checked(cfg, token)

    def fail(*a, **k):
        raise PipelineError("The two sides could not be joined.")

    monkeypatch.setattr("castlive.recon.run.process", fail)
    loop.step_reconstruct(cfg)
    row = _row(cfg, cid)
    assert row["status"] == "failed" and "joined" in row["error"]


def test_retry(studio, cfg, token):
    cid = _checked(cfg, token)
    assert studio.post(f"/api/captures/{cid}/retry").status_code == 409  # not finished yet
    with db.connect(cfg.db) as con:
        db.set_capture(con, cid, status="failed", error="x")
    assert studio.post(f"/api/captures/{cid}/retry").status_code == 200
    assert _row(cfg, cid)["status"] == "checked"
    assert studio.get(f"/api/captures/{cid}/model.glb").status_code == 404
    assert studio.get("/api/captures/../../etc/model.glb").status_code == 404


def test_process_killing_capture_is_not_retried_forever(cfg, token, monkeypatch):
    # The process dies mid-run (e.g. out of memory): the row stays 'processing' and the
    # restarted worker picks it up again. After MAX_ATTEMPTS starts it must give up.
    cid = _checked(cfg, token)

    def killed(*a, **k):
        raise SystemExit  # stands in for SIGKILL: nothing after this line runs

    monkeypatch.setattr("castlive.recon.run.process", killed)
    for _ in range(loop.MAX_ATTEMPTS):
        with pytest.raises(SystemExit):
            loop.step_reconstruct(cfg)
        assert _row(cfg, cid)["status"] == "processing"
    loop.step_reconstruct(cfg)
    row = _row(cfg, cid)
    assert row["status"] == "failed" and row["error"] == loop.CRASHED
    assert not loop.step_reconstruct(cfg)  # the queue moves on


def test_heartbeat_thread(tmp_path):
    import time

    beat = tmp_path / "hb"
    loop.heartbeat(beat, every=0.05)
    time.sleep(0.2)
    assert beat.exists()


def test_tracker_eta():
    now = [0.0]
    t = loop.Tracker(["frames", "poses", "dense"], {"frames": 10, "poses": 20, "dense": 70},
                     clock=lambda: now[0])
    p = t.update("frames", 0.0)
    assert p["fraction"] == 0 and p["eta_seconds"] == 100
    now[0] = 40  # poses: half done in 30 s, so about 30 s left here, plus 70 for dense
    t.update("poses", 0.0)
    now[0] = 70
    p = t.update("poses", 0.5)
    assert p["eta_seconds"] == 100 and p["fraction"] == 0.2
    p = t.update("dense", 0.0)  # a new stage starts from its estimate
    assert p["eta_seconds"] == 70


def test_estimates_learn_from_finished_runs(cfg, token):
    cid = _checked(cfg, token)
    with db.connect(cfg.db) as con:
        db.set_capture(con, cid, status="done",
                       meta={"report": {"seconds": {"dense": 300, "frames": 20}}})
    est = loop.stage_estimates(cfg)
    assert est["dense"] == 300 and est["frames"] == 20 and est["mesh"] == loop.DEFAULT_SECONDS["mesh"]


def test_colmap_progress_lines():
    from castlive.recon.colmap import line_fraction

    assert line_fraction("I1105 Processed file [40/160]") == 0.25
    assert line_fraction("Matching block [2/4, 1/4]") == 5 / 16
    assert line_fraction("Processing view 80 / 160 for p1_000080.jpg") == 0.5
    assert line_fraction("Registering image #12 (30)", total_images=60) == 0.5
    assert line_fraction("Registering image #12 (30)") is None
    assert line_fraction("nothing here") is None


def test_crop_cameras_moves_principal_point_and_points():
    cams = "# c\n1 PINHOLE 3200 1800 2500.0 2500.0 1600.0 900.0\n"
    imgs = ("# i\n"
            "1 1 0 0 0 0 0 0 1 p1_000001.jpg\n100.0 200.0 7 1700.5 950.25 -1\n"
            "2 1 0 0 0 0 0 0 1 p1_000002.jpg\n")
    c, i = colmap.crop_cameras(cams, imgs, {"p1_000001.jpg": (1000, 500, 2200, 1400)})
    c_lines, i_lines = c.splitlines(), i.splitlines()
    assert c_lines[0] == "1 PINHOLE 1200 900 2500.0 2500.0 600.0 400.0"
    assert c_lines[1] == "2 PINHOLE 3200 1800 2500.0 2500.0 1600.0 900.0"  # no box: whole
    assert i_lines[0].split()[8] == "1" and i_lines[2].split()[8] == "2"  # own camera each
    assert i_lines[1].split() == ["-900.0", "-300.0", "7", "700.5", "450.25", "-1"]
    assert i_lines[3] == ""


def test_mask_box():
    m = np.zeros((1000, 2000), np.uint8)
    m[400:600, 900:1100] = 255
    assert colmap.mask_box(m, 0.1, 100) == (880, 380, 1119, 619)  # object 900..1099 + 19.9
    assert colmap.mask_box(m, 0.1, 300) is None  # too small: keep the whole image
    assert colmap.mask_box(np.zeros((10, 10), np.uint8), 0.1, 1) is None


def test_pass_overlap_counts_shared_points():
    images = ("1 1 0 0 0 0 0 0 1 p1_000001.jpg\n\n2 1 0 0 0 0 0 0 1 p1_000002.jpg\n\n"
              "3 1 0 0 0 0 0 0 1 p2_000001.jpg\n\n")
    points = ("# id x y z r g b err track\n"
              "1 0 0 0 0 0 0 0.5 1 0 2 0\n"  # side 1 only
              "2 0 0 0 0 0 0 0.5 1 1 3 4\n"  # both sides
              "3 0 0 0 0 0 0 0.5 3 9\n")  # side 2 only
    o = colmap.pass_overlap(images, points)
    assert (o["shared"], o["p1"], o["p2"]) == (1, 1, 1)
    assert o["spread"] == 0.0  # far too few points to judge: counts as not joined


def test_join_quality_tells_real_joins_from_false_ones():
    rng = np.random.default_rng(0)
    # a box-like object: points on the rim of a 4 x 3 rectangle
    t = rng.uniform(0, 14, 2000)
    rim = np.stack([np.where(t < 4, t, np.where(t < 7, 4, np.where(t < 11, 11 - t, 0))),
                    np.where(t < 4, 0, np.where(t < 7, t - 4, np.where(t < 11, 3, 14 - t))),
                    rng.normal(0, 0.05, 2000)], 1)
    kinds = ["p1" if i % 2 else "p2" for i in range(2000)]
    everywhere = [("shared" if i % 7 == 0 else k) for i, k in enumerate(kinds)]
    real = colmap.join_quality(rim, everywhere)
    assert real["spread"] > 0.7 and real["sectors"] >= 14
    one_edge = [("shared" if (rim[i, 1] == 0 and i % 3 == 0) else k) for i, k in enumerate(kinds)]
    false = colmap.join_quality(rim, one_edge)
    assert false["spread"] < 0.4 and false["sectors"] <= 6
    assert colmap.join_quality(rim[:5], kinds[:5]) == {"spread": 0.0, "sectors": 0}


@pytest.fixture
def fake_colmap(tmp_path, monkeypatch):
    """A `colmap` on PATH that prints the given lines, then exits with the given code."""
    def make(lines, code=0):
        exe = tmp_path / "bin" / "colmap"
        exe.parent.mkdir(exist_ok=True)
        body = "".join(f"echo '{ln}'\n" for ln in lines)
        exe.write_text(f"#!/bin/sh\n{body}exit {code}\n")
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", f"{exe.parent}:" + __import__("os").environ["PATH"])
    return make


def test_run_progress_never_goes_back_and_errors_dont_stop_it(fake_colmap, tmp_path):
    fake_colmap(["Registering image #1 (50)", "Registering image #2 (90)",
                 "Registering image #3 (10)", "done"])  # a second sub-model counts from 0
    seen = []

    def progress(f):
        seen.append(f)
        if len(seen) == 2:
            raise RuntimeError("database is locked")  # display only: must not stop COLMAP

    colmap.run("mapper", [], tmp_path / "log.txt", progress, total_images=100)
    assert seen == [0.5, 0.9, 0.9]
    assert "done" in (tmp_path / "log.txt").read_text()


def test_run_raises_on_failure(fake_colmap, tmp_path):
    fake_colmap(["boom"], code=3)
    with pytest.raises(RuntimeError, match="exit 3"):
        colmap.run("mapper", [], tmp_path / "log.txt")


def test_run_retries_a_crash_once(fake_colmap, tmp_path):
    fake_colmap(["CUDA failure 999"], code=0)
    exe = next((tmp_path / "bin").glob("colmap"))
    # first call aborts (killed by a signal), the second succeeds
    exe.write_text(f"#!/bin/sh\nif [ -e {tmp_path}/once ]; then echo ok; exit 0; fi\n"
                   f"touch {tmp_path}/once\nkill -ABRT $$\n")
    colmap.run("feature_extractor", [], tmp_path / "log.txt")
    assert "crashed; trying again" in (tmp_path / "log.txt").read_text()
    (tmp_path / "once").unlink()
    with pytest.raises(colmap.Crashed):
        colmap.run("feature_extractor", [], tmp_path / "log.txt", retries=0)


def _flip_scene(turned: bool):
    """Points on a flat slab; pass 1 cameras above it, pass 2 below (turned over) or above."""
    rng = np.random.default_rng(1)
    xyz = np.c_[rng.uniform(-2, 2, 500), rng.uniform(-1.5, 1.5, 500), rng.normal(0, 0.1, 500)]
    names = [f"p1_{i:06d}.jpg" for i in range(10)] + [f"p2_{i:06d}.jpg" for i in range(10)]
    ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    ring = np.c_[3 * np.cos(ang), 3 * np.sin(ang)]
    z2 = -2 if turned else 2
    centres = np.r_[np.c_[ring, np.full(10, 2.0)], np.c_[ring, np.full(10, z2)]]
    ups = np.r_[np.tile([0, 0, 1.0], (10, 1)), np.tile([0, 0, -1.0 if turned else 1.0], (10, 1))]
    return names, centres, ups, xyz


def test_flip_check_rejects_mirror_solutions():
    good = colmap.flip_check(*_flip_scene(turned=True))
    assert good["turned_over"] and good["opposite_sides"] and good["up_angle"] == 180
    mirror = colmap.flip_check(*_flip_scene(turned=False))  # pass 2 laid onto pass 1's side
    assert not mirror["turned_over"] and not mirror["opposite_sides"] and mirror["up_angle"] == 0


def test_poses_tries_candidates_until_one_is_accepted(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(colmap, "features", lambda *a, **k: calls.append(("features", a[3])) or {})
    monkeypatch.setattr(colmap, "run", lambda cmd, *a, **k: calls.append((cmd,)) or 1.0)
    models = iter([None, "second"])

    def fake_largest(out, log):
        m = next(models)
        return m and colmap.Model(out, ["p1_1.jpg"], np.zeros((1, 3)), np.zeros((1, 3)),
                                  np.zeros((1, 3)), 1, 0.1, {"tag": m})

    monkeypatch.setattr(colmap, "largest", fake_largest)
    (tmp_path / "img").mkdir()
    model, _, tried = colmap.poses(tmp_path / "img", None, tmp_path / "w", tmp_path / "log",
                                   accept=lambda m: m.overlap["tag"] == "second")
    assert model.overlap["tag"] == "second"
    assert [t["accepted"] for t in tried] == [False, True]
    assert [t["setup"] for t in tried] == ["aliked", "sift"]
    assert calls[0] == ("features", "aliked") and ("features", "sift") in calls


def test_poses_skips_a_setup_whose_features_crash(monkeypatch, tmp_path):
    def features(images, masks, db, setup, log, progress=None):
        if setup == "aliked":
            raise colmap.Crashed("colmap feature_extractor failed (exit -6):\nCUDA failure 999")
        return {}

    monkeypatch.setattr(colmap, "features", features)
    monkeypatch.setattr(colmap, "run", lambda cmd, *a, **k: 1.0)
    monkeypatch.setattr(colmap, "largest", lambda out, log: colmap.Model(
        out, ["p1_1.jpg"], np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), 1, 0.1, {}))
    (tmp_path / "img").mkdir()
    model, _, tried = colmap.poses(tmp_path / "img", None, tmp_path / "w", tmp_path / "log")
    assert model.folder.name == "sift_global_mapper"
    assert tried[0]["setup"] == "aliked" and "exit -6" in tried[0]["error"]
    assert [t["setup"] for t in tried] == ["aliked", "sift"]  # aliked's second mapper skipped


def test_texture_scene_cam_files(tmp_path):
    from castlive.recon import texture
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "p1_000001.jpg").write_bytes(b"jpg")
    cameras = "1 PINHOLE 1000 800 1200 1260 480 410\n"
    images = "1 1 0 0 0 0.5 -0.25 2 1 p1_000001.jpg\n\n2 1 0 0 0 0 0 0 1 gone.jpg\n\n"
    assert texture.write_scene(tmp_path / "img", cameras, images, tmp_path / "scene") == 1
    ext, intr = (tmp_path / "scene" / "p1_000001.cam").read_text().splitlines()
    assert [float(x) for x in ext.split()] == [0.5, -0.25, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert [float(x) for x in intr.split()] == pytest.approx([1.2, 0, 0, 1.05, 0.48, 0.5125])
    assert (tmp_path / "scene" / "p1_000001.jpg").read_bytes() == b"jpg"


def test_read_obj_splits_corners_with_two_uvs(tmp_path):
    from castlive.recon import texture
    (tmp_path / "t.mtl").write_text("newmtl a\nmap_Kd a.png\nnewmtl b\n")
    (tmp_path / "t.obj").write_text(
        "mtllib t.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvt 0.5 0.5\n"
        "vn 0 0 1\nusemtl a\nf 1/1/1 2/2/1 3/3/1\nf 2/4/1 4/2/1 3/3/1\nusemtl b\nf 1//1 2//1 4//1\n")
    (a, uv, f, img), (b, _, fb, none) = texture.read_obj(tmp_path / "t.obj")
    assert img == tmp_path / "a.png" and none is None
    assert len(a) == 5  # vertex 2 has two UVs, so it is split
    assert np.allclose(a[f], [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [0, 1, 0]]])
    assert np.allclose(uv[f[0]], [[0, 1], [1, 1], [0, 0]])  # v flipped to top-left origin
    assert len(b) == 3 and fb.shape == (1, 3)


def test_textured_glb_layout(tmp_path):
    import json as _json
    import struct
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    png = b"\x89PNG\r\n\x1a\nfake"
    parts = [(v, np.tile([0, 0, 1.0], (3, 1)), np.zeros((3, 2)), np.array([[0, 1, 2]]), png),
             (v, None, None, np.array([[0, 1, 2]]), None)]
    glb.write_textured_glb(tmp_path / "t.glb", parts)
    raw = (tmp_path / "t.glb").read_bytes()
    n = struct.unpack("<I", raw[12:16])[0]
    doc = _json.loads(raw[20:20 + n])
    p0, p1 = doc["meshes"][0]["primitives"]
    assert "TEXCOORD_0" in p0["attributes"] and "TEXCOORD_0" not in p1["attributes"]
    assert doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] == {"index": 0}
    assert doc["images"][0]["mimeType"] == "image/png"
    assert struct.unpack("<I", raw[:4])[0] == 0x46546C67 and len(raw) % 4 == 0


def test_placement_is_shared_by_parts():
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], float)
    place = mesh.placement(v, np.zeros(3), np.array([0, 1.0, 0]))
    whole, _ = place(v, np.zeros((3, 3)))
    part, _ = place(v[1:], np.zeros((2, 3)))
    assert np.allclose(whole[1:], part) and np.ptp(whole, axis=0).max() == pytest.approx(mesh.SIZE)


def test_part_normals_come_from_the_whole_mesh():
    """A seam vertex copied into two parts gets the whole mesh's normal in both, not each
    part's own (which would show a false crease under raking light)."""
    import numpy as np
    import pytest

    pytest.importorskip("scipy")
    from castlive.recon import mesh

    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.5]], float)
    n = np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1], [0.2, 0.2, 1]], float)
    m = mesh.Mesh(v, np.array([[0, 1, 2], [1, 3, 2]]), None, n, {})
    part_a, part_b = v[[0, 1, 2]], v[[1, 3, 2]]
    assert np.allclose(mesh.normals_for(part_a, m)[1:], mesh.normals_for(part_b, m)[[0, 2]])
    assert np.allclose(mesh.normals_for(part_b, m), n[[1, 3, 2]])
