"""Browser sessions: a random cookie value; only its sha256 is stored (``auth/sessions.sqlite``).

Kept beside ``auth.json`` rather than in the main DB: the main DB path is cwd-relative, and ``fts auth sessions
clear`` must reach the same sessions as the web app from any directory.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from contextlib import closing

from finetune_studio.access.paths import auth_dir, sessions_file

COOKIE = "fts_session"
IDLE_TTL_S = 7 * 24 * 3600          # signed out after a week without use
ABSOLUTE_TTL_S = 30 * 24 * 3600     # and after 30 days whatever happens
_TOUCH_EVERY_S = 300

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    sha256 TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    user_agent TEXT NOT NULL DEFAULT '',
    client TEXT NOT NULL DEFAULT ''
)"""


def _db() -> sqlite3.Connection:
    auth_dir(create=True)
    f = sessions_file()
    con = sqlite3.connect(f, timeout=10)
    con.execute(_SCHEMA)
    f.chmod(0o600)
    return con


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create(*, user_agent: str = "", client: str = "") -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with closing(_db()) as con, con:
        con.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)", (_sha(token), now, now, user_agent[:200], client[:64]))
    return token


def validate(token: str | None) -> bool:
    if not token:
        return False
    now = time.time()
    with closing(_db()) as con, con:
        row = con.execute("SELECT created_at, last_seen_at FROM sessions WHERE sha256 = ?", (_sha(token),)).fetchone()
        if row is None:
            return False
        created, seen = row
        if now - created > ABSOLUTE_TTL_S or now - seen > IDLE_TTL_S:
            con.execute("DELETE FROM sessions WHERE sha256 = ?", (_sha(token),))
            return False
        if now - seen > _TOUCH_EVERY_S:
            con.execute("UPDATE sessions SET last_seen_at = ? WHERE sha256 = ?", (now, _sha(token)))
    return True


def revoke(token: str) -> None:
    with closing(_db()) as con, con:
        con.execute("DELETE FROM sessions WHERE sha256 = ?", (_sha(token),))


def revoke_all(*, except_token: str | None = None) -> int:
    with closing(_db()) as con, con:
        if except_token:
            cur = con.execute("DELETE FROM sessions WHERE sha256 != ?", (_sha(except_token),))
        else:
            cur = con.execute("DELETE FROM sessions")
        return cur.rowcount


def count() -> int:
    if not sessions_file().exists():
        return 0
    cutoff = time.time() - IDLE_TTL_S
    with closing(_db()) as con:
        return int(con.execute("SELECT COUNT(*) FROM sessions WHERE last_seen_at > ?", (cutoff,)).fetchone()[0])
