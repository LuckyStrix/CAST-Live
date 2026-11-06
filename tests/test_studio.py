from fastapi.testclient import TestClient

from castlive.studio.app import create_app


def test_needs_tailnet_identity(cfg):
    bare = TestClient(create_app(cfg))
    assert bare.get("/").status_code == 403
    assert bare.get("/api/projects").status_code == 403


def test_project_invite_flow(studio, capture):
    pid = studio.post("/api/projects", json={"name": "Nippur", "mode": "object"}).json()["id"]
    inv = studio.post(f"/api/projects/{pid}/invites", json={"label": "team A", "days": 7}).json()
    assert inv["url"] == f"https://testserver/c/{inv['token']}"
    assert capture.get(f"/api/{inv['token']}/info").json()["project"] == "Nippur"
    assert studio.get(f"/api/invites/{inv['token']}/qr.svg").text.startswith("<svg")
    d = studio.get(f"/api/projects/{pid}").json()
    assert d["invites"][0]["active"] and d["invites"][0]["label"] == "team A"
    studio.post(f"/api/invites/{inv['token']}/revoke")
    assert not studio.get(f"/api/projects/{pid}").json()["invites"][0]["active"]
    assert capture.get(f"/api/{inv['token']}/info").status_code == 404


def test_bad_project(studio):
    assert studio.post("/api/projects", json={"name": ""}).status_code == 400
    assert studio.post("/api/projects", json={"name": "x", "mode": "nope"}).status_code == 400
    assert studio.get("/api/projects/999").status_code == 404


def test_status(studio):
    s = studio.get("/api/status").json()
    assert s["user"] == "me@example.com" and s["worker_ok"] is False
