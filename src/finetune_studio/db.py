"""SQLite-backed project/run/RAG storage.

WHAT THIS FILE DOES
==================
Single source of truth for the persistent layer that holds Projects,
their RAGs, their Training Runs and their Benchmark Runs. Everything
else (CLI, webui, sub-agents) talks to the DB through these helpers.

KEY CONCEPTS
============
- stdlib sqlite3 — no SQLAlchemy/ORM. Minimal dependency surface.
- One DB file at settings.db_path. Initialised on first import.
- Each helper opens a short-lived connection. The studio is a single-
  user local app, so we don't need connection pooling.
- All IDs are short random strings (8 hex chars). Human-readable in URLs.

SCHEMA
======
projects        — Project (a thing you train a model for)
project_rags    — Vector stores scoped to a Project (legal / social / …)
training_runs   — One training attempt with settings + outputs + metadata
benchmark_runs  — Benchmark suite results tied to a training run
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

from finetune_studio.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    base_model      TEXT NOT NULL DEFAULT '',
    system_prompt   TEXT NOT NULL DEFAULT '',
    production_run  TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS project_rags (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL,
    name         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    tags         TEXT NOT NULL DEFAULT '',
    store_path   TEXT NOT NULL,
    doc_count    INTEGER NOT NULL DEFAULT 0,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rags_project ON project_rags(project_id);

CREATE TABLE IF NOT EXISTS training_runs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    name          TEXT NOT NULL,
    base_model    TEXT NOT NULL,
    data_path     TEXT NOT NULL DEFAULT '',
    rag_ids_json  TEXT NOT NULL DEFAULT '[]',
    settings_json TEXT NOT NULL DEFAULT '{}',
    system_prompt TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'created',
    started_at    REAL,
    finished_at   REAL,
    output_path   TEXT NOT NULL DEFAULT '',
    metrics_json  TEXT NOT NULL DEFAULT '{}',
    notes         TEXT NOT NULL DEFAULT '',
    parent_run_id TEXT,
    created_at    REAL NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON training_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON training_runs(status);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    suite_name  TEXT NOT NULL,
    scores_json TEXT NOT NULL DEFAULT '{}',
    time_ms     INTEGER NOT NULL DEFAULT 0,
    ran_at      REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_bench_run ON benchmark_runs(run_id);

CREATE TABLE IF NOT EXISTS data_review (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    row_index   INTEGER NOT NULL,
    decision    TEXT NOT NULL,           -- approved | rejected | edited
    edited_json TEXT,
    reviewed_at REAL NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_review_project ON data_review(project_id, dataset);
"""


def _new_id() -> str:
    """8-char hex id."""
    return secrets.token_hex(4)


def _connect() -> sqlite3.Connection:
    """Open a connection with foreign keys enabled and row factory set."""
    os.makedirs(os.path.dirname(settings.db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    """Short-lived cursor context. Commits on exit, rolls back on error."""
    with _connect() as conn:
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def init_db() -> None:
    """Create tables if they don't exist. Safe to call repeatedly."""
    with cursor() as c:
        c.executescript(_SCHEMA)


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    # Decode JSON columns.
    for col in ("rag_ids_json", "settings_json", "metrics_json", "scores_json"):
        if col in d and isinstance(d[col], str) and d[col]:
            try:
                d[col.removesuffix("_json")] = json.loads(d[col])
            except Exception:  # noqa: BLE001
                d[col.removesuffix("_json")] = None
            d.pop(col, None)
    return d


# ── Projects ─────────────────────────────────────────────────────────────

def create_project(name: str, description: str = "", base_model: str = "",
                   system_prompt: str = "") -> dict:
    pid = _new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO projects (id, name, description, base_model, system_prompt, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, name, description, base_model, system_prompt, now, now),
        )
    return get_project(pid)  # type: ignore[return-value]


def get_project(pid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    return row_to_dict(r)


def list_projects() -> list[dict]:
    with cursor() as c:
        rows = c.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
    return [row_to_dict(r) for r in rows]


def update_project(pid: str, **fields: Any) -> dict | None:
    allowed = {"name", "description", "base_model", "system_prompt", "production_run"}
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return get_project(pid)
    sets.append("updated_at = ?")
    vals.append(time.time())
    vals.append(pid)
    with cursor() as c:
        c.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", vals)
    return get_project(pid)


def delete_project(pid: str) -> bool:
    with cursor() as c:
        c.execute("DELETE FROM projects WHERE id = ?", (pid,))
    return True


# ── Project RAGs ─────────────────────────────────────────────────────────

def create_rag(project_id: str, name: str, description: str = "",
               tags: str = "", store_path: str = "") -> dict:
    rid = _new_id()
    now = time.time()
    if not store_path:
        store_path = os.path.join("data", "rags", project_id, rid)
    with cursor() as c:
        c.execute(
            "INSERT INTO project_rags (id, project_id, name, description, tags, store_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, project_id, name, description, tags, store_path, now),
        )
    return get_rag(rid)  # type: ignore[return-value]


def get_rag(rid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM project_rags WHERE id = ?", (rid,)).fetchone()
    return row_to_dict(r)


def list_rags(project_id: str) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM project_rags WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def update_rag(rid: str, **fields: Any) -> dict | None:
    allowed = {"name", "description", "tags", "doc_count", "chunk_count"}
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return get_rag(rid)
    vals.append(rid)
    with cursor() as c:
        c.execute(f"UPDATE project_rags SET {', '.join(sets)} WHERE id = ?", vals)
    return get_rag(rid)


def delete_rag(rid: str) -> bool:
    with cursor() as c:
        c.execute("DELETE FROM project_rags WHERE id = ?", (rid,))
    return True


# ── Training Runs ────────────────────────────────────────────────────────

def create_run(project_id: str, name: str, base_model: str = "",
               data_path: str = "", rag_ids: list | None = None,
               settings_obj: dict | None = None, system_prompt: str = "",
               parent_run_id: str | None = None, notes: str = "") -> dict:
    rid = _new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO training_runs (id, project_id, name, base_model, data_path, "
            "rag_ids_json, settings_json, system_prompt, parent_run_id, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, project_id, name, base_model, data_path,
             json.dumps(rag_ids or []), json.dumps(settings_obj or {}),
             system_prompt, parent_run_id, notes, now),
        )
    return get_run(rid)  # type: ignore[return-value]


def get_run(rid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM training_runs WHERE id = ?", (rid,)).fetchone()
    return row_to_dict(r)


def list_runs(project_id: str | None = None) -> list[dict]:
    with cursor() as c:
        if project_id:
            rows = c.execute(
                "SELECT * FROM training_runs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM training_runs ORDER BY created_at DESC").fetchall()
    return [row_to_dict(r) for r in rows]


def update_run(rid: str, **fields: Any) -> dict | None:
    allowed = {
        "name", "base_model", "data_path", "system_prompt",
        "status", "started_at", "finished_at", "output_path",
        "metrics_json", "notes", "parent_run_id",
    }
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            if k in ("metrics_json",) and isinstance(v, (dict, list)):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return get_run(rid)
    vals.append(rid)
    with cursor() as c:
        c.execute(f"UPDATE training_runs SET {', '.join(sets)} WHERE id = ?", vals)
    return get_run(rid)


def delete_run(rid: str) -> bool:
    with cursor() as c:
        c.execute("DELETE FROM training_runs WHERE id = ?", (rid,))
    return True


# ── Benchmark Runs ───────────────────────────────────────────────────────

def create_benchmark(run_id: str, suite_name: str, scores: dict,
                     time_ms: int = 0) -> dict:
    bid = _new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO benchmark_runs (id, run_id, suite_name, scores_json, time_ms, ran_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (bid, run_id, suite_name, json.dumps(scores), time_ms, now),
        )
    return get_benchmark(bid)  # type: ignore[return-value]


def get_benchmark(bid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM benchmark_runs WHERE id = ?", (bid,)).fetchone()
    return row_to_dict(r)


def list_benchmarks(run_id: str | None = None) -> list[dict]:
    with cursor() as c:
        if run_id:
            rows = c.execute(
                "SELECT * FROM benchmark_runs WHERE run_id = ? ORDER BY ran_at DESC",
                (run_id,),
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM benchmark_runs ORDER BY ran_at DESC").fetchall()
    return [row_to_dict(r) for r in rows]


# ── Data review (approve / reject / edited rows) ────────────────────────

def record_review(project_id: str, dataset: str, row_index: int,
                  decision: str, edited_json: str | None = None) -> dict:
    """Record a row-level review decision. Overwrites prior decision for (project, dataset, row)."""
    rid = _new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "DELETE FROM data_review WHERE project_id = ? AND dataset = ? AND row_index = ?",
            (project_id, dataset, row_index),
        )
        c.execute(
            "INSERT INTO data_review (id, project_id, dataset, row_index, decision, edited_json, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, project_id, dataset, row_index, decision, edited_json, now),
        )
    return {"id": rid, "row_index": row_index, "decision": decision}


def list_review(project_id: str, dataset: str) -> list[dict]:
    with cursor() as c:
        rows = c.execute(
            "SELECT * FROM data_review WHERE project_id = ? AND dataset = ?",
            (project_id, dataset),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


# Initialise on import so callers don't have to remember.
init_db()
