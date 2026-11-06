"""SQLite: the one shared store between capture, studio and worker."""

import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

MODES = ("object", "in-situ")
VIDEO_EXT = {"video/mp4": "mp4", "video/webm": "webm"}  # accepted upload types

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    mode TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS invites (
    token TEXT PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    label TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    expires REAL,
    max_captures INTEGER,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS captures (
    id TEXT PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    invite TEXT NOT NULL REFERENCES invites(token),
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    mime TEXT NOT NULL,
    bytes INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER,
    meta TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL DEFAULT '',
    progress TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL,
    updated REAL NOT NULL
);
"""

# Capture status: uploading -> received (all chunks joined) -> checked | rejected (worker)
#   -> processing (stage = current pipeline stage) -> done | failed
# An upload can also end as cancelled (by the phone) or abandoned (no activity for weeks).
CLOSED = ("rejected", "failed", "abandoned", "cancelled")  # these don't use up an invite's capture limit


def is_capture_id(s: str) -> bool:
    """A capture id is a UUID in its one canonical spelling (it becomes a folder name)."""
    try:
        return str(uuid.UUID(s)) == s
    except (ValueError, TypeError, AttributeError):
        return False


def video_path(inbox: Path, cid: str, mime: str) -> Path:
    return inbox / cid / f"video.{VIDEO_EXT[mime]}"


def invite_active(row) -> bool:
    return not row["revoked"] and (row["expires"] is None or row["expires"] > time.time())


@contextmanager
def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
    finally:
        con.close()


def init(path: Path) -> None:
    with connect(path) as con:
        con.executescript(SCHEMA)
        cols = {r["name"] for r in con.execute("PRAGMA table_info(captures)")}
        for col, ddl in (("stage", "TEXT NOT NULL DEFAULT ''"),  # added in Phase 2
                         ("progress", "TEXT NOT NULL DEFAULT '{}'")):
            if col not in cols:
                con.execute(f"ALTER TABLE captures ADD COLUMN {col} {ddl}")


def create_project(con, name: str, mode: str) -> int:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cur = con.execute(
        "INSERT INTO projects (name, mode, created) VALUES (?, ?, ?)", (name, mode, time.time())
    )
    return cur.lastrowid


def create_invite(con, project_id: int, label: str = "", days: float | None = 30,
                  max_captures: int | None = None) -> str:
    token = secrets.token_urlsafe(16)  # 128 bits
    expires = time.time() + days * 86400 if days else None
    con.execute(
        "INSERT INTO invites (token, project_id, label, created, expires, max_captures)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (token, project_id, label, time.time(), expires, max_captures),
    )
    return token


def valid_invite(con, token: str):
    """The invite row if the token may be used right now, else None."""
    row = con.execute("SELECT * FROM invites WHERE token = ?", (token,)).fetchone()
    return row if row is not None and invite_active(row) else None


def captures_used(con, token: str) -> int:
    return con.execute(
        f"SELECT COUNT(*) FROM captures WHERE invite = ? AND status NOT IN {CLOSED}", (token,)
    ).fetchone()[0]


def set_capture(con, capture_id: str, **fields) -> None:
    for key in ("meta", "progress"):
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json.dumps(fields[key])
    fields["updated"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    con.execute(f"UPDATE captures SET {cols} WHERE id = ?", (*fields.values(), capture_id))
