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
    id                TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL,
    name              TEXT NOT NULL,
    description       TEXT NOT NULL DEFAULT '',
    tags              TEXT NOT NULL DEFAULT '',
    store_path        TEXT NOT NULL,
    doc_count         INTEGER NOT NULL DEFAULT 0,
    chunk_count       INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL DEFAULT 'ready',
    last_build_at     REAL,
    last_build_status TEXT,
    error             TEXT NOT NULL DEFAULT '',
    created_at        REAL NOT NULL,
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
    error         TEXT NOT NULL DEFAULT '',
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

-- Per-project jsonl training files (uploaded or exported from data-prep).
-- The training tab reads from this table instead of taking a raw path.
CREATE TABLE IF NOT EXISTS project_datasets (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    name          TEXT NOT NULL,
    data_path     TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'upload',   -- upload | data-prep-export
    qa_count      INTEGER NOT NULL DEFAULT 0,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    last_used_at  REAL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_datasets_project ON project_datasets(project_id);

-- data_prep_runs: per-run persistence for the data-prep pipeline.
-- The in-memory _RUNS dict in routes/data_prep.py stays for in-progress
-- polling, but the DB is the durable record.
CREATE TABLE IF NOT EXISTS data_prep_runs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    source_id     TEXT NOT NULL DEFAULT '',
    filename      TEXT NOT NULL DEFAULT '',
    byte_count    INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'queued',
    started_at    REAL,
    finished_at   REAL,
    duration_ms   INTEGER,
    qa_total      INTEGER NOT NULL DEFAULT 0,
    qa_approved   INTEGER NOT NULL DEFAULT 0,
    output_path   TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    settings_json TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_data_prep_project ON data_prep_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_data_prep_status  ON data_prep_runs(status);

-- rag_corpora: one row per RAG build attempt. Tracks timing + status +
-- final doc/chunk counts. project_rags holds the latest summary; this
-- table is the history.
CREATE TABLE IF NOT EXISTS rag_corpora (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    rag_id        TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued',
    started_at    REAL,
    finished_at   REAL,
    duration_ms   INTEGER,
    doc_count     INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    error         TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (rag_id) REFERENCES project_rags(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rag_corpora_project ON rag_corpora(project_id);
CREATE INDEX IF NOT EXISTS idx_rag_corpora_rag     ON rag_corpora(rag_id);

-- hf_downloads: durable record of HF model download jobs. Replaces the
-- in-memory _DOWNLOADS dict in routes/hf_models.py so downloads survive
-- a service restart.
CREATE TABLE IF NOT EXISTS hf_downloads (
    id            TEXT PRIMARY KEY,
    repo_id       TEXT NOT NULL,
    filename      TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'queued',
    started_at    REAL,
    finished_at   REAL,
    duration_ms   INTEGER,
    bytes_total   INTEGER,
    bytes_done    INTEGER,
    path          TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hf_downloads_status ON hf_downloads(status);
"""


def new_id() -> str:
    """8-char hex id."""
    return secrets.token_hex(4)


def _connect() -> sqlite3.Connection:
    """Open a connection with foreign keys enabled and row factory set.

    `defer_foreign_keys = ON` lets us INSERT a child row before its parent
    is fully visible — useful when a route creates a parent (e.g. project)
    and a child (e.g. run) in the same request.
    """
    os.makedirs(os.path.dirname(settings.db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA defer_foreign_keys = ON")
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


def _safe_alter(cur: sqlite3.Cursor, sql: str) -> None:
    """Run an ALTER TABLE; ignore `duplicate column name` errors.

    We use this for additive migrations on tables that pre-date the current
    schema. If the column is already there, the ALTER is a no-op.
    """
    try:
        cur.execute(sql)
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e):
            raise


def init_db() -> None:
    """Create tables if they don't exist, then run additive migrations.

    Safe to call repeatedly. New columns added in newer code are detected
    via _safe_alter — repeated calls are no-ops.
    """
    with cursor() as c:
        c.executescript(_SCHEMA)
        # Migrations: widen tables that already exist on upgraded installs.
        _safe_alter(c, "ALTER TABLE project_rags ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'")
        _safe_alter(c, "ALTER TABLE project_rags ADD COLUMN last_build_at REAL")
        _safe_alter(c, "ALTER TABLE project_rags ADD COLUMN last_build_status TEXT")
        _safe_alter(c, "ALTER TABLE project_rags ADD COLUMN error TEXT NOT NULL DEFAULT ''")
        _safe_alter(c, "ALTER TABLE training_runs ADD COLUMN error TEXT NOT NULL DEFAULT ''")


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
