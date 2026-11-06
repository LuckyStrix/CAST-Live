"""Settings, read once from the environment. Everything has a safe default."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEB = Path(os.environ.get("CASTLIVE_WEB", str(REPO / "web")))


@dataclass(frozen=True)
class Config:
    data: Path
    chunk_bytes: int  # largest chunk a phone may send
    max_capture_bytes: int  # largest whole capture
    max_capture_seconds: int
    trust_local: bool  # studio: accept requests with no Tailscale identity (dev only)

    @property
    def db(self) -> Path:
        return self.data / "castlive.db"

    @property
    def inbox(self) -> Path:
        return self.data / "inbox"

    @property
    def runs(self) -> Path:
        return self.data / "runs"


def load() -> Config:
    env = os.environ.get
    return Config(
        data=Path(env("CASTLIVE_DATA", str(REPO / "data"))),
        chunk_bytes=int(env("CASTLIVE_CHUNK_BYTES", str(8 * 1024 * 1024))),
        max_capture_bytes=int(env("CASTLIVE_MAX_CAPTURE_BYTES", str(4 * 1024**3))),
        max_capture_seconds=int(env("CASTLIVE_MAX_CAPTURE_SECONDS", "300")),
        trust_local=env("CASTLIVE_TRUST_LOCAL", "0") == "1",
    )
