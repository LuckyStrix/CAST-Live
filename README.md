# CAST Live

A phone video of an artifact in, a **gaussian splat** and a textured mesh out, with the phone
showing **live** which parts of the object have been covered while you film.

CAST Live is the non-commercial research sibling of [CAST](https://github.com/LuckyStrix/CAST)
(Capture, Align, Surface, Texture). It keeps CAST's capture and studio sites, worker and Tailscale
setup, and rebuilds the modelling around newer learned tools (MASt3R-SLAM, VGGT, gaussian
splatting) whose licences allow research use only. See [LICENSES.md](LICENSES.md) and the plan in
[PLAN.md](PLAN.md).

**Status:** phase 3 — **live coverage works**: while you film, the phone sends small frames
(about 4 a second) to the GPU, MASt3R-SLAM tracks them, and the phone's ring lights up each
direction filmed so far, with a small turning preview of the points. "Turn it over" appears when
the round is complete. The studio can watch the same session (`watch live` on the capture). The
final model is still CAST's photogrammetry (`castlive/recon/`); gaussian splats come next.

```
phone ──(invite link, Tailscale Funnel)──► capture site ──► worker ──► reconstructor ──► studio (your tailnet)
```

While filming, the capture site and the `live` service share only files (`/data/live/`): the
public site still has no network path to the GPU containers. A reconstruction waits between its
stages while someone films live, so the GPU is theirs.

## Run it

Needs a machine with Docker, an NVIDIA GPU and a Tailscale account.

1. In the [Tailscale admin console](https://login.tailscale.com/admin): turn on **HTTPS certificates**
   (DNS page), allow **Funnel** in the access policy (`nodeAttrs` with `"attr": ["funnel"]`), and
   create an **auth key** (Settings → Keys).
2. Start it:
   ```sh
   cd docker
   cp .env.example .env    # paste the auth key into .env
   docker compose up -d --build
   ```
3. On any device on your tailnet open **`https://castlive.<your-tailnet>.ts.net:8443`** (studio).
   Create a project, then an invite link, and scan its QR code with a phone (Safari on iPhone,
   Chrome on Android; nothing to install).

It can run next to CAST on the same machine: separate machine name, containers and data volume.
To update: `git pull`, then `docker compose up -d --build` (Windows: `powershell -File docker\update.ps1`).

## Develop

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest && .venv/bin/ruff check castlive tests
CASTLIVE_TRUST_LOCAL=1 CASTLIVE_PUBLIC_URL=http://127.0.0.1:8001 .venv/bin/castlive-studio   # http://127.0.0.1:8002
.venv/bin/castlive-capture                        # http://127.0.0.1:8001
.venv/bin/castlive-worker                         # needs ffmpeg on PATH
python tests/fake_phone.py http://127.0.0.1:8001/c/<token> some.mp4 --drop 0.3
```

`CASTLIVE_TRUST_LOCAL=1` lets studio run without Tailscale. Never set it in the container.

## License

Code: MIT (`LICENSE`). The models it runs are **non-commercial** — the project as a whole is for
research use only. Details: [LICENSES.md](LICENSES.md).
