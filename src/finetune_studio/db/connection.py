"""SQLite connection plumbing + schema. Single low-level concern."""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from finetune_studio.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    base_model      TEXT NOT NULL DEFAULT '',
    system_prompt   TEXT NOT NULL DEFAULT '',
    tags            TEXT NOT NULL DEFAULT '',
    notes           TEXT NOT NULL DEFAULT '',
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
    system_prompt_mode TEXT NOT NULL DEFAULT 'bake',
    status        TEXT NOT NULL DEFAULT 'created',
    started_at    REAL,
    finished_at   REAL,
    output_path   TEXT NOT NULL DEFAULT '',
    metrics_json  TEXT NOT NULL DEFAULT '{}',
    final_loss    REAL,
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
    model_path  TEXT NOT NULL DEFAULT '',
    scores_json TEXT NOT NULL DEFAULT '{}',
    time_ms     INTEGER NOT NULL DEFAULT 0,
    ran_at      REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_bench_run ON benchmark_runs(run_id);

CREATE TABLE IF NOT EXISTS benchmark_cases (
    id              TEXT PRIMARY KEY,
    benchmark_id    TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    case_name       TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'general',
    question        TEXT NOT NULL,
    correct_answer  TEXT NOT NULL DEFAULT '',
    model_answer    TEXT NOT NULL DEFAULT '',
    transcript      TEXT NOT NULL DEFAULT '',   -- full chat JSON for judge context
    judge           TEXT NOT NULL DEFAULT 'none',  -- none | ai | human
    judge_model     TEXT NOT NULL DEFAULT '',      -- model used for AI judge
    verdict         TEXT NOT NULL DEFAULT '',      -- pass | fail | partial
    judge_reasoning TEXT NOT NULL DEFAULT '',      -- judge explanation
    scored_at       REAL,
    scoring_method  TEXT NOT NULL DEFAULT '',
    validity        TEXT NOT NULL DEFAULT '',
    error           TEXT NOT NULL DEFAULT '',
    judge_input     TEXT NOT NULL DEFAULT '',
    source_id       TEXT NOT NULL DEFAULT '',
    chunk_idx       INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (benchmark_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_bc_benchmark ON benchmark_cases(benchmark_id);
CREATE INDEX IF NOT EXISTS idx_bc_run ON benchmark_cases(run_id);

CREATE TABLE IF NOT EXISTS auto_suites (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    project_id      TEXT NOT NULL,
    suite_name      TEXT NOT NULL,
    suite_path      TEXT NOT NULL,
    case_count      INTEGER NOT NULL DEFAULT 0,
    categories_json TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_auto_suites_run ON auto_suites(run_id);

CREATE TABLE IF NOT EXISTS abliteration_runs (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    project_id      TEXT NOT NULL,
    model_path      TEXT NOT NULL,
    output_path     TEXT NOT NULL,
    strength        REAL NOT NULL DEFAULT 1.0,
    magnitude       REAL NOT NULL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_abliteration_run ON abliteration_runs(run_id);

CREATE TABLE IF NOT EXISTS quant_exports (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    project_id      TEXT NOT NULL,
    model_path      TEXT NOT NULL,
    output_path     TEXT NOT NULL,
    method          TEXT NOT NULL,  -- gptq, imatrix (legacy awq rows may remain)
    bits            INTEGER NOT NULL DEFAULT 4,
    group_size      INTEGER NOT NULL DEFAULT 128,
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_quant_exports_run ON quant_exports(run_id);

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

-- model_exports: GGUF / future-format exports of a trained run.
-- One row per export attempt. Tracks timing + status + output path
-- + file size. Replaces any prior ad-hoc logging.
CREATE TABLE IF NOT EXISTS model_exports (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    format        TEXT NOT NULL DEFAULT 'gguf',
    quant         TEXT NOT NULL DEFAULT 'Q4_K_M',
    status        TEXT NOT NULL DEFAULT 'queued',
    started_at    REAL,
    finished_at   REAL,
    duration_ms   INTEGER,
    output_path   TEXT NOT NULL DEFAULT '',
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    size_human    TEXT NOT NULL DEFAULT '',
    intermediate_path TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_exports_project ON model_exports(project_id);
CREATE INDEX IF NOT EXISTS idx_exports_run     ON model_exports(run_id);
CREATE INDEX IF NOT EXISTS idx_exports_status  ON model_exports(status);

-- system_updates: every self-healing update attempt. Logs the full
-- script output so the UI can show what happened during the install.
-- Not tied to a project — it's a system-wide concern.
CREATE TABLE IF NOT EXISTS system_updates (
    id            TEXT PRIMARY KEY,
    mode          TEXT NOT NULL DEFAULT 'update',  -- update | check | repair
    status        TEXT NOT NULL DEFAULT 'queued',  -- queued | running | done | error | cancelled
    started_at    REAL,
    finished_at   REAL,
    duration_ms   INTEGER,
    log_text      TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    options_json  TEXT NOT NULL DEFAULT '{}',       -- no_pull, no_llama, no_restart
    triggered_by  TEXT NOT NULL DEFAULT 'user',     -- user | system
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_updates_status ON system_updates(status);

-- ── File library (Stage 1) ────────────────────────────────────────────────
-- User-named folders, organised independently of file storage. Raw files
-- are immutable + MIME-segregated at upload time. Converted versions are
-- versioned per file so we can detect drift across dataset builds.

CREATE TABLE IF NOT EXISTS file_folders (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'user',  -- 'user' | 'auto' (e.g. 'pdfs' under raw)
    parent_id   TEXT,
    created_at  REAL NOT NULL,
    UNIQUE(project_id, parent_id, name),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_file_folders_project ON file_folders(project_id);

CREATE TABLE IF NOT EXISTS model_favorites (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_path      TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL DEFAULT '',
    added_at        REAL NOT NULL,
    note            TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_model_favorites_added ON model_favorites(added_at);

CREATE TABLE IF NOT EXISTS project_files (
    id              TEXT PRIMARY KEY,      -- first 16 hex of sha256(raw bytes)
    project_id      TEXT NOT NULL,
    original_name   TEXT NOT NULL,         -- user-visible filename, unique per project
    mime_type       TEXT NOT NULL DEFAULT 'application/octet-stream',
    tags            TEXT NOT NULL DEFAULT '',
    notes           TEXT NOT NULL DEFAULT '',
    current_version INTEGER NOT NULL DEFAULT 1,
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    uploaded_at     REAL NOT NULL,
    uploaded_by     TEXT NOT NULL DEFAULT 'user',
    last_trained_at REAL,                  -- last version included in a dataset
    deleted_at      REAL,                  -- soft-delete timestamp (NULL = live)
    trash_kind      TEXT,                  -- 'raw' | 'converted' | NULL
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    UNIQUE(project_id, original_name)
);
CREATE INDEX IF NOT EXISTS idx_project_files_project ON project_files(project_id);
CREATE INDEX IF NOT EXISTS idx_project_files_deleted ON project_files(project_id, deleted_at);

CREATE TABLE IF NOT EXISTS file_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     TEXT NOT NULL,
    version     INTEGER NOT NULL,
    raw_path    TEXT NOT NULL,             -- absolute path on disk under files/raw/...
    raw_hash    TEXT NOT NULL,             -- sha256 of raw bytes (== file id when stable)
    raw_size    INTEGER NOT NULL,
    uploaded_at REAL NOT NULL,
    uploaded_by TEXT NOT NULL DEFAULT 'user',
    UNIQUE(file_id, version),
    FOREIGN KEY (file_id) REFERENCES project_files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_file_versions_file ON file_versions(file_id);

CREATE TABLE IF NOT EXISTS file_conversions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id         TEXT NOT NULL,
    version         INTEGER NOT NULL,
    format          TEXT NOT NULL,         -- 'md' | 'txt' | 'structured_json'
    converted_path  TEXT NOT NULL,         -- absolute path under files/converted/...
    converted_hash  TEXT NOT NULL,
    converted_size  INTEGER NOT NULL,
    converter       TEXT NOT NULL,         -- 'docling' | 'pypdf' | 'tesseract' | ...
    converted_at    REAL NOT NULL,
    status          TEXT NOT NULL,         -- 'ok' | 'error'
    error_message   TEXT,
    UNIQUE(file_id, version, format),
    FOREIGN KEY (file_id) REFERENCES project_files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_file_conversions_file ON file_conversions(file_id);

CREATE TABLE IF NOT EXISTS folder_membership (
    folder_id TEXT NOT NULL,
    file_id   TEXT NOT NULL,
    PRIMARY KEY (folder_id, file_id),
    FOREIGN KEY (folder_id) REFERENCES file_folders(id) ON DELETE CASCADE,
    FOREIGN KEY (file_id)   REFERENCES project_files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_folder_membership_file ON folder_membership(file_id);
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
        _safe_alter(
            c,
            "ALTER TABLE training_runs ADD COLUMN system_prompt_mode TEXT NOT NULL DEFAULT 'bake'",
        )
        _safe_alter(c, "ALTER TABLE training_runs ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'")
        _safe_alter(c, "ALTER TABLE training_runs ADD COLUMN final_loss REAL")
        _safe_alter(c, "ALTER TABLE training_runs ADD COLUMN notes TEXT NOT NULL DEFAULT ''")
        _safe_alter(c, "ALTER TABLE training_runs ADD COLUMN parent_run_id TEXT")
        _safe_alter(
            c,
            "ALTER TABLE project_files ADD COLUMN tags TEXT NOT NULL DEFAULT ''",
        )
        _safe_alter(
            c,
            "ALTER TABLE project_files ADD COLUMN notes TEXT NOT NULL DEFAULT ''",
        )
        _safe_alter(c, "ALTER TABLE benchmark_cases ADD COLUMN scoring_method TEXT NOT NULL DEFAULT ''")
        _safe_alter(c, "ALTER TABLE benchmark_cases ADD COLUMN validity TEXT NOT NULL DEFAULT ''")
        _safe_alter(c, "ALTER TABLE benchmark_cases ADD COLUMN error TEXT NOT NULL DEFAULT ''")
        _safe_alter(c, "ALTER TABLE benchmark_cases ADD COLUMN judge_input TEXT NOT NULL DEFAULT ''")
        _safe_alter(c, "ALTER TABLE benchmark_cases ADD COLUMN source_id TEXT NOT NULL DEFAULT ''")
        _safe_alter(c, "ALTER TABLE benchmark_cases ADD COLUMN chunk_idx INTEGER NOT NULL DEFAULT 0")
        # Create benchmark_cases if it doesn't exist (new in v2)
        c.executescript("""
            CREATE TABLE IF NOT EXISTS benchmark_cases (
                id              TEXT PRIMARY KEY,
                benchmark_id    TEXT NOT NULL,
                run_id          TEXT NOT NULL,
                case_name       TEXT NOT NULL,
                category        TEXT NOT NULL DEFAULT 'general',
                question        TEXT NOT NULL,
                correct_answer  TEXT NOT NULL DEFAULT '',
                model_answer    TEXT NOT NULL DEFAULT '',
                transcript      TEXT NOT NULL DEFAULT '',
                judge           TEXT NOT NULL DEFAULT 'none',
                judge_model     TEXT NOT NULL DEFAULT '',
                verdict         TEXT NOT NULL DEFAULT '',
                judge_reasoning TEXT NOT NULL DEFAULT '',
                scored_at       REAL,
                FOREIGN KEY (benchmark_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE,
                FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_bc_benchmark ON benchmark_cases(benchmark_id);
            CREATE INDEX IF NOT EXISTS idx_bc_run ON benchmark_cases(run_id);
        """)


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    # Decode JSON columns (TEXT → dict under the unsuffixed key).
    for col in (
        "rag_ids_json",
        "settings_json",
        "metrics_json",
        "scores_json",
        "options_json",
    ):
        if col in d and isinstance(d[col], str) and d[col]:
            try:
                d[col.removesuffix("_json")] = json.loads(d[col])
            except json.JSONDecodeError:
                d[col.removesuffix("_json")] = None
            d.pop(col, None)
    return d
