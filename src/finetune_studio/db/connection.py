"""SQLite connection plumbing + schema. Single low-level concern."""
from __future__ import annotations

import json
import os
import sqlite3
import secrets
from contextlib import contextmanager
from typing import Iterator

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


def new_id() -> str:
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
