import pytest
from fastapi.testclient import TestClient

from castlive.capture.app import create_app as capture_app
from castlive.common import config, db
from castlive.studio.app import create_app as studio_app


@pytest.fixture
def cfg(tmp_path):
    return config.Config(data=tmp_path, chunk_bytes=1024, max_capture_bytes=10 * 1024,
                         max_capture_seconds=300, trust_local=False)


@pytest.fixture
def capture(cfg):
    return TestClient(capture_app(cfg))


@pytest.fixture
def studio(cfg):
    return TestClient(studio_app(cfg), headers={"Tailscale-User-Login": "me@example.com"})


@pytest.fixture
def token(cfg):
    db.init(cfg.db)
    with db.connect(cfg.db) as con:
        pid = db.create_project(con, "Test", "object")
        return db.create_invite(con, pid, "phone")
