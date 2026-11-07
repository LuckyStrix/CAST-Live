# Licences

CAST Live's own code is MIT (see `LICENSE`). **The project as a whole is for non-commercial
research use only**, because it runs models whose licences forbid commercial use (marked **NC**
below). Models made with it may carry those terms too. Code that should stay usable commercially
belongs in [CAST](https://github.com/LuckyStrix/CAST), which is permissive only.

Add every new dependency or model here **before** using it.

| Component | Used for | Licence | Status |
|---|---|---|---|
| FastAPI, Starlette | web apps | MIT / BSD-3 | in use |
| Uvicorn | web server | BSD-3 | in use |
| segno | invite QR codes | BSD-3 | in use |
| FFmpeg (Debian build) | probing and decoding video | LGPL-2.1+ / GPL parts | in use, run as a separate program only |
| tailscale/tailscale image | Funnel and Serve | BSD-3 | in use |
| Python 3.12 slim image | base image | PSF | in use |
| COLMAP 4.2.1 (incl. global mapper) | final camera poses, turn-over join | BSD-3 | in use |
| NVIDIA CUDA runtime + cuDNN | GPU | NVIDIA licences (not ours to redistribute) | in use, pulled at build time |
| BiRefNet via rembg; ONNX Runtime | object masks | MIT | in use |
| websockets | the live view's WebSocket | BSD-3 | in use |
| Open3D, SciPy, OpenCV, NumPy, Pillow | meshing, images | MIT / BSD / Apache-2.0 / HPND | in use |
| three.js r170 (vendored) | studio viewer | MIT | in use |
| ALIKED / LightGlue (COLMAP's ONNX models) | features, matching | BSD-3 / Apache-2.0 | in use |
| mvs-texturing (texrecon) with mve, rayint, mapmap_cpu; Eigen (MPL-2-only) | texture from the photos | BSD-3 / MPL-2.0 | in use |
| PyTorch 2.7 (CUDA 12.8 image) | running the learned models | BSD-3 | in use (live image) |
| **MASt3R-SLAM** (windows branch, 6717231) | live tracking and coverage | **CC BY-NC-SA 4.0** — NC | in use; patched at build time, its code is called, not copied |
| **MASt3R / DUSt3R checkpoints** (Naver) | MASt3R-SLAM's model | **CC BY-NC-SA 4.0**; trained on data with non-commercial terms (CHECKPOINTS_NOTICE) — NC | in use, hashes pinned |
| lietorch (e7df865) | Lie groups for MASt3R-SLAM | BSD-3 | in use |
| **VGGT-1B** (Meta) | fast first camera poses per pass | **VGGT licence, non-commercial** — NC | candidate (phase 4) |
| gsplat | gaussian splat training (MCMC, 2DGS) | Apache-2.0 | planned (phase 4) |
| **PGSR** / **MILo** | mesh from the splats | **Inria-style, non-commercial** — NC | candidate (phase 4), check each |
| Spark (sparkjs) | splat viewer in the studio | MIT (check 2.0) | planned (phase 5) |
| SPZ (Niantic) | compressed splat files | MIT | planned (phase 4) |

**Not used:** VGGT-SLAM's SALAD loop-closure model (GPL-3), pymeshlab (GPL).
