# CAST Live — plan

## Context
CAST (photogrammetry: COLMAP → depth → Poisson mesh → texrecon) works end to end, but takes ~110 min per object, leaves holes on plain surfaces, shows slight corner doubling at the turn-over join, and gives the filmer no feedback until long after capture. Newer learned tools (MASt3R, VGGT, splat surface methods) fix most of this, but their weights are non-commercial. So we build a sibling project that may use them freely, keeping CAST's proven layout and services, and rebuilding the modelling backend around:
- **live SLAM while filming** — the phone (and studio) see the object fill in, and where coverage is missing;
- **gaussian splats** as the final model, plus a mesh made from them.

Decisions: live view on phone + studio; output splat + textured mesh; own code MIT, non-commercial terms come only from the models (listed in LICENSES.md). Working name **CAST Live**, new repo `LuckyStrix/CAST-Live`, package `castlive/`, user sole contributor (no co-author lines). Same host (luckydesktop, RTX 5060 Ti 16 GB), same Tailscale setup (Funnel = capture only, Serve = studio).

## Tool choices (research, Nov 2026 — verify each licence before adding)
| Job | Choice | Licence | Why |
|---|---|---|---|
| Live SLAM | **MASt3R-SLAM** | code: own licence; MASt3R weights CC BY-NC-SA (modified) | Most mature; 15 FPS on a 4090 (expect ~6–10 on the 5060 Ti at 512 px); dense pointmaps, loop closure, no calibration needed |
| SLAM alternative (benchmarked in phase 1) | **VGGT-SLAM 2.0** (MIT-SPARK) | code BSD-2; VGGT weights non-commercial | 8.4 FPS on a 3090, submaps, newer; real-time code may be unreleased |
| Turn-over join + final poses | **CAST's COLMAP candidates + turn-over check** (VGGT as a fast first guess per pass) | BSD / NC | Phase 1: MASt3R and VGGT both fall for the mirror solution on the box |
| Fast pose fallback | **MapAnything** (Apache variant) or **Depth Anything 3** (Large/Giant CC BY-NC; small Apache) | mixed | Feed-forward poses + depth in seconds |
| Splat training | **gsplat** (nerfstudio) — MCMC densification, 2DGS mode, masks | Apache-2.0 | Maintained, fast, has 2DGS; install from source (main is ahead of PyPI) |
| Splat → mesh | **PGSR** (best accuracy, dense views — ours are dense); **MILo** as alternative (compact meshes) | Inria-style non-commercial | Sharp surfaces; then texture with CAST's texrecon |
| Web splat viewer | **Spark** (sparkjs, three.js) with **SPZ** files (Niantic, MIT) | MIT (check Spark 2.0 licence) | Sits alongside the existing three.js mesh viewer |
| Masks | **BiRefNet** (reuse CAST) | MIT | Unchanged |

Rejected for now: per-frame live splat training on the phone feed (too heavy alongside SLAM on 16 GB); WebRTC (WebSocket JPEG frames are simpler; switch only if latency demands it).

## Phase 1 results (6 Nov 2026, box + mouse frames from CAST, RTX 5060 Ti)
| | MASt3R-SLAM | VGGT-SLAM 2.0 |
|---|---|---|
| Speed | 4–6 fps on every frame | ~2 keyframes/s (flow picks ~1 in 3–5 frames) |
| Extra GPU memory | ~4 GB | ~5.3 GB (+5 GB model on disk) |
| Box side 1 / 2 vs COLMAP (median position % of orbit radius, rotation) | 3.7% 1.7° / 3.3% 1.4° | 2.1% 1.4° / 1.5% 0.9° |
| Mouse | 4.4% (max 12%) 2.3° | 5.0% (max 25%) 1.1° |
| Build on Blackwell | 5 source patches (old `.type()`, removed `linalg_norm`, sm_120, legacy checkpoint load) | none; but SALAD loop closure is **GPL-3**, weights from Google Drive |
| Turn-over join (box, both sides at once) | **fails** (MASt3R global alignment: sides 10–20° apart, should be ~176°) | **fails** (VGGT feed-forward: 8–10°) |

Decisions from this: **live SLAM = MASt3R-SLAM** (faster on every frame, less memory, no GPL); **VGGT** as a fast per-pass pose start for the final job (24 frames in 8 s); **the turn-over join stays with CAST's COLMAP candidates + turn-over check** — both learned models fall for the mirror solution on the box. Spike images: `castlive-spike:local`, `castlive-vggt:local`; data in `/data/exp/slam/`.

## Architecture
```
phone ──Funnel──► capture (public)  ── /data/live/<session>/ (files) ──► live (GPU, no network)
  │  WebSocket: small JPEG frames up,          ▲ coverage + points          │ MASt3R-SLAM
  │  coverage/points down                      └────────────────────────────┘
  │  full-res video recorded on the phone, uploaded after (CAST chunked upload)
  ▼
worker (checks upload) ──► reconstructor (GPU): frames → masks → poses (SLAM init + MASt3R join + BA)
                                               → splat (gsplat) → mesh (PGSR) → texture (texrecon) → SPZ + GLB
studio (tailnet): projects, live watch page, splat viewer (Spark) + mesh viewer (raking light)
```
- **Live link without breaking isolation:** capture keeps no route to the GPU containers. Capture writes incoming frames into `/data/live/<session>/in/`; the `live` service (network_mode none) consumes them and writes `state.json` (keyframe poses, coverage) + `points.bin` (decimated coloured points) to `out/`; capture streams changes back over the same WebSocket. File hand-off adds ~50–100 ms — fine.
- **Coverage map:** a sphere of viewing directions around the object centre (e.g. 64 bins in azimuth × elevation), filled by keyframe directions weighted by point density seen; the phone shows a ring/hemisphere "donut" + the growing point cloud, and the existing guide prompts ("go lower", "flip it now") are driven by real coverage instead of the compass heading.
- **Two passes (object mode):** SLAM session per pass (the flip breaks tracking by design); passes joined in the final reconstruction by CAST's COLMAP candidates, checked with its turn-over test (`up_angle`, opposite sides). Live view after the flip shows pass 2 alone, then the joined preview once a join is found.
- **GPU sharing (one 16 GB card):** `live` and `reconstructor` share a lock file in `/data`; a live session has priority — the reconstructor checks the lock between stages and waits; a live session is refused with a clear message only if a stage is mid-way and the card is full.
- **Live preview splat (later phase):** after the user taps Stop, a 2-minute quick gsplat on SLAM keyframes gives a first splat in studio before the full job finishes.

## Project layout (mirrors CAST)
```
castlive/common/   config.py db.py web.py            ← copied from CAST (cast/common/*)
castlive/capture/  app.py (+ live WebSocket)          ← CAST capture/app.py + new /live endpoint
castlive/studio/   app.py (+ live watch, splat files) ← CAST studio/app.py
castlive/worker/   loop.py                            ← CAST worker/loop.py (adds live service entry)
castlive/live/     slam.py coverage.py session.py     ← NEW: MASt3R-SLAM wrapper, coverage bins, file hand-off
castlive/recon/    CAST's pipeline (copied in phase 2); splat.py surface.py spz.py added and
                   the Poisson mesh retired in phase 4
web/capture/       index.html capture.js guide.js store.js live.js (NEW: socket + coverage overlay)
web/studio/        index/log/view + splat.js (NEW, Spark) + live.html (NEW)
docker/            compose.yml (+ live service), Dockerfile, Dockerfile.gpu (PyTorch CUDA 12.8 for Blackwell sm_120,
                   gsplat, MASt3R-SLAM, PGSR, texrecon stage from CAST), serve.json, update.ps1, .env.example
tests/             same style as CAST (fake phone, fake GPU tools)
LICENSES.md PLAN.md README.md LICENSE (MIT)
```
Reused as-is from CAST: invite/project model, chunked resumable upload, CSP headers (`cast/common/web.py`, incl. `blob:` for textures), worker upload checks, frame selection (`cast/pipeline/frames.py`), BiRefNet masks (`masks.py`), texrecon build stage and `texture.py`, `glb.py` (textured parts), `mesh.placement`/`normals_for`, studio log/report pages, `update.ps1` deploy.

## Phases (code review after each, as in CAST)
1. **Spike (no app code) — done, results above:** build the GPU image on the 5060 Ti (sm_120 + CUDA 12.8 — main risk: MASt3R-SLAM/PGSR custom CUDA kernels must compile for Blackwell). Run MASt3R-SLAM and VGGT-SLAM 2.0 on the box and mouse frames from CAST: FPS at 512 px, VRAM, pose agreement with CAST's COLMAP poses, does the flip join with MASt3R matching. Pick the SLAM engine. Report numbers.
2. **Repo skeleton:** new repo, copy common/capture/studio/worker/docker/web from CAST, rename package, CI, LICENSES.md with every model's terms, (the `live` service is added in phase 3 with its code). Existing CAST tests pass under the new names.
3. **Live loop — done (6 Nov 2026):** box frames at phone speed through the deployed site: 93/95
   and 63/65 frames placed, ~4 frames/s, coverage 100% on both sides, "Turn it over" shown by
   coverage. Live positions are taken from keyframe-corrected poses (tracked poses drift).
   Original scope: phone WebSocket (invite-checked, rate-limited, max frame size, session timeout) → file hand-off → `live` SLAM → coverage + points back to phone and to a studio live page. Phone records full-res video at the same time and uploads it after (existing upload path).
4. **Final reconstruction — done (7 Nov 2026):** poses stay CAST's COLMAP + turn-over check
   (learned joins failed in phase 1). COLMAP's dense stereo is replaced by a **2DGS splat**
   (gsplat, 7000 steps, masked) with PGSR-style multi-view terms in our own code (patch NCC
   through each pixel's plane into neighbouring photos, forward-backward depth check). Its
   median depth from every camera is fused into a GPU TSDF (off-mask pixels carve), marching
   cubes only between seen cells, largest piece, 500k triangles → texrecon → GLB. Every model
   gets a quality report: outline holes/extras vs the masks from every camera, distance of
   COLMAP's best points to the surface, held-out photo PSNR.
   Box (same frames, RTX 5060 Ti): outline missed 32.9% → 0.8%, dense-MVS points to surface
   median 0.14% of size, stage ~8 min instead of 55. Also checked on a second box scan, CAST's
   box and an in-the-ground mouse (closed, where CAST's mesh had holes). Tried and dropped:
   MoGe-2 normal prior (terracing), expected depth (embosses printed ink). Open: faint false
   relief where only colour changes (ink, printed text) on plain faces; SPZ export (phase 5).
5. **Studio viewers:** Spark splat viewer with the mesh viewer as a toggle (raking light stays on the mesh); downloads SPZ, PLY, GLB.
6. **Polish:** quick preview splat after Stop; Android + iPhone tests; docs for untrained users; scale card later.

## Security (public capture site)
- Live WebSocket only with a valid invite, one session per invite, frames ≤ 200 KB JPEG, ≤ 15 fps, session ≤ 10 min, decoded with Pillow inside capture (size checked before decode); capture still has no network path to studio or GPU containers.
- Points sent back are low-res and only to the same session; studio live page tailnet-only.

## Verification
- Unit tests (pytest, ruff) like CAST: coverage binning, session hand-off with a fake SLAM writing files, WebSocket limits with the fake phone, SPZ writer round-trip, pose join check.
- Phase 1 numbers on the box/mouse frames (FPS, VRAM, pose difference vs COLMAP).
- End to end on the desktop: film with the iPhone through Funnel, watch coverage fill on the phone and in studio, flip, finish; final job produces SPZ + GLB; compare with CAST on the same object (time, holes on plain faces, corner doubling, text sharpness).
- Android capture test before calling phase 3 done.
