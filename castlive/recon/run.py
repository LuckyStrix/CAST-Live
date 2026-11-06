"""One capture through every stage. Writes everything to /data/runs/<capture id>/.

Each finished stage is recorded in state.json, so a retry continues from the stage that
failed. `progress(stage, fraction)` is called as stages start and advance, for studio; run.log
gets a line per stage, colmap.log everything COLMAP prints.
"""

import json
import time
from pathlib import Path

import numpy as np

from castlive.recon import PipelineError, colmap, frames, glb, masks, mesh, texture

STAGES = ["frames", "masks", "poses", "dense", "mesh"]
# Two passes are joined only if enough 3D points were seen from both, spread round the object.
# Frames all getting a position proves nothing: the first box run placed 158/160 frames on 42
# shared points bunched on two lines, and its halves were far apart. Measured on the box:
# false joins had spread 0.1-0.4 and 4-5 of 16 sectors; true joins 0.74-0.99 and 12-16.
MIN_SHARED_POINTS = 100
MIN_SPREAD = 0.5
MIN_SECTORS = 10
FRAME_SIZE = 3200  # long side of extracted frames: small objects need the pixels


def stages_for(mode: str) -> list[str]:
    return [s for s in STAGES if s != "masks" or mode == "object"]


def process(video: Path, meta: dict, mode: str, run_dir: Path,
            progress=lambda stage, fraction: None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": []}
    report = state.setdefault("report", {"mode": mode, "seconds": {}})
    log = run_dir / "colmap.log"
    images, mask_dir = run_dir / "frames", run_dir / "masks"
    dense_dir, cloud = run_dir / "dense", run_dir / "dense" / "fused.ply"
    use_masks = mode == "object"

    def note(text):
        with open(run_dir / "run.log", "a") as fh:
            fh.write(time.strftime("%H:%M:%S ") + text + "\n")

    def stage(name, fn):
        if name in state["done"]:
            return
        progress(name, 0.0)
        note(f"{name}: started")
        start = time.time()
        try:
            fn(lambda f: progress(name, f))
        except Exception as e:
            note(f"{name}: FAILED: {e}")
            raise
        report["seconds"][name] = round(time.time() - start)
        note(f"{name}: done in {report['seconds'][name]} s")
        state["done"].append(name)
        state_path.write_text(json.dumps(state, indent=1))

    def do_frames(p):
        markers = meta.get("phone", {}).get("markers", [])
        report["frames"] = frames.make_frames(video, markers, meta["video"]["duration"], mode,
                                              images, max_size=FRAME_SIZE, progress=p)
        note(f"frames: {report['frames']['frames']} kept")

    def do_masks(p):
        report["masks"] = masks.make_masks(images, mask_dir, masks.GROW_FOR_POSES, progress=p)

    def do_poses(p):
        total = len(list(images.glob("*.jpg")))
        frames_per_pass = {k: len(list(images.glob(f"{k}_*.jpg")))
                           for k in sorted({f.name.split("_")[0] for f in images.glob("*.jpg")})}
        two_sides = use_masks and len(frames_per_pass) > 1

        def placed(model):
            return {k: {"frames": n, "placed": sum(x.startswith(k + "_") for x in model.names)}
                    for k, n in frames_per_pass.items()}

        def joined(model):
            o = model.overlap
            return (o.get("shared", 0) >= MIN_SHARED_POINTS and o.get("spread", 0) >= MIN_SPREAD
                    and o.get("sectors", 0) >= MIN_SECTORS
                    and all(v["placed"] >= 0.5 * v["frames"] for v in placed(model).values()))

        def good(model):
            if len(model.names) < colmap.MIN_REGISTERED * total:
                return False
            return not two_sides or (joined(model) and model.overlap["flip"]["turned_over"])

        model, secs, tried = colmap.poses(images, mask_dir if use_masks else None,
                                          run_dir / "sfm", log, progress=p, accept=good)
        for t in tried:
            if "error" in t:
                note(f"poses: {t['setup']} features failed, skipping it: {t['error']}")
                continue
            o, f = t["overlap"], t["overlap"].get("flip", {})
            note(f"poses: {t['setup']} + {t['mapper']}: {t['placed']}/{total} placed, {o.get('shared', 0)} "
                 f"shared (spread {o.get('spread')}, {o.get('sectors')}/16), up angle "
                 f"{f.get('up_angle')}, opposite sides {f.get('opposite_sides')} -> "
                 f"{'accepted' if t['accepted'] else 'rejected'}")
        per_pass = placed(model)
        report["poses"] = {"placed": len(model.names), "frames": total, "passes": per_pass,
                           "points": model.points, "error_px": round(model.error, 3),
                           "overlap": model.overlap, "tried": tried,
                           "colmap_seconds": {k: round(v) for k, v in secs.items()}}
        state["model"] = str(model.folder)
        state["cameras"] = {"names": model.names, "centres": model.centres.tolist(),
                            "ups": model.ups.tolist(), "forwards": model.forwards.tolist()}
        note(f"poses: {len(model.names)}/{total} frames placed, by pass {per_pass}")
        if len(model.names) < colmap.MIN_REGISTERED * total:
            raise PipelineError(f"Only {len(model.names)} of {total} frames could be placed. "
                                "Walk slower and keep the whole object in view.")
        if two_sides:  # in the ground nothing moved: no join to check
            report["poses"]["joined"] = joined(model) and model.overlap["flip"]["turned_over"]
            if not report["poses"]["joined"]:
                raise PipelineError("The two sides could not be joined into one model. Film "
                                    "the edges slowly, before and after turning the object over, "
                                    "and keep the object filling the box on screen.")

    def do_dense(p):
        # Object: undistort at full size, mask, crop each image to the object. In the ground:
        # the whole scene matters, so whole images at a moderate size.
        secs = {"undistort": colmap.undistort(images, Path(state["model"]), dense_dir, log,
                                              FRAME_SIZE if use_masks else 2000,
                                              colmap.span(p, 0, 0.03))}
        fusion_masks = None
        if use_masks:  # fusion and cropping read the undistorted images, so mask those
            fusion_masks = dense_dir / "masks"
            t = time.time()
            masks.make_masks(dense_dir / "images", fusion_masks, masks.GROW_FOR_FUSION,
                             progress=colmap.span(p, 0.03, 0.1))
            report["crop"] = colmap.crop_to_masks(dense_dir, fusion_masks, log)
            secs["masks_and_crop"] = time.time() - t
        secs["stereo"] = colmap.stereo(dense_dir, log, progress=colmap.span(p, 0.1, 0.95))
        secs["fusion"] = colmap.fuse(dense_dir, cloud, fusion_masks, log,
                                     progress=colmap.span(p, 0.95, 1.0))
        report["dense_seconds"] = {k: round(v) for k, v in secs.items()}

    def do_mesh(p):
        cams = state["cameras"]
        centres, ups = np.array(cams["centres"]), np.array(cams["ups"])
        centre = mesh.look_at_point(centres, np.array(cams["forwards"]))
        distance = float(np.median(np.linalg.norm(centres - centre, axis=1)))
        first = np.array([n.startswith("p1_") for n in cams["names"]])
        up = ups[first].mean(axis=0) if first.any() else ups.mean(axis=0)
        m = mesh.build(cloud, centre, radius=0.8 * distance)
        report["mesh"] = m.stats
        note(f"mesh: {m.stats['triangles']:,} triangles")
        place = mesh.placement(m.vertices, centre, up)
        try:  # colour from the photos; point colours if that fails
            parts, report["texture"] = texture.make(dense_dir, m.vertices, m.faces,
                                                    run_dir / "texture", log)
        except Exception as e:  # noqa: BLE001
            note(f"mesh: texturing failed ({e}); using point colours")
            parts = None
        if parts:
            out = []
            for v, uv, f, image in parts:
                pv, pn = place(v, mesh.normals_for(v, m))
                out.append((pv, pn, uv, f, image.read_bytes() if image else None))
            glb.write_textured_glb(run_dir / "model.glb", out)
            note(f"mesh: textured from {report['texture']['views']} photos in "
                 f"{report['texture']['seconds']} s")
        else:
            v, n = place(m.vertices, m.normals)
            glb.write_glb(run_dir / "model.glb", v, m.faces, m.colors, n)

    jobs = {"frames": do_frames, "masks": do_masks, "poses": do_poses, "dense": do_dense,
            "mesh": do_mesh}
    for name in stages_for(mode):
        try:
            stage(name, jobs[name])
        finally:
            state_path.write_text(json.dumps(state, indent=1))
    (run_dir / "report.json").write_text(json.dumps(report, indent=1))
    return report
