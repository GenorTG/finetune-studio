"""SQLite connection plumbing + schema. Single low-level concern."""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager

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

-- brain_indexes: the project's Smart Brain index (one per project; the chat attaches to it).
CREATE TABLE IF NOT EXISTS brain_indexes (
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
CREATE INDEX IF NOT EXISTS idx_brain_indexes_project ON brain_indexes(project_id);

CREATE TABLE IF NOT EXISTS training_runs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    name          TEXT NOT NULL,
    base_model    TEXT NOT NULL,
    data_path     TEXT NOT NULL DEFAULT '',
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

-- Every opinion on a case's correctness, never overwritten: an AI judge, a human, or (legacy rows only) the old
-- scripted matcher. benchmark_cases.verdict is the EFFECTIVE verdict derived from these rows (human > AI > scripted).
CREATE TABLE IF NOT EXISTS case_judgements (
    id            TEXT PRIMARY KEY,
    case_id       TEXT NOT NULL,
    benchmark_id  TEXT NOT NULL,
    kind          TEXT NOT NULL,                 -- ai | human | exact (official public benchmark) | scripted (legacy rows)
    provider_id   TEXT NOT NULL DEFAULT '',
    judge_model   TEXT NOT NULL DEFAULT '',
    verdict       TEXT NOT NULL DEFAULT '',      -- pass | partial | fail | '' (a failed judge call, or a retracted human verdict)
    reasoning     TEXT NOT NULL DEFAULT '',
    confidence    REAL,
    error         TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT '',     -- judge prompt revision that produced an AI verdict
    created_at    REAL NOT NULL,
    FOREIGN KEY (case_id) REFERENCES benchmark_cases(id) ON DELETE CASCADE,
    FOREIGN KEY (benchmark_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_cj_case ON case_judgements(case_id);
CREATE INDEX IF NOT EXISTS idx_cj_benchmark ON case_judgements(benchmark_id);

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
    method          TEXT NOT NULL,  -- imatrix (legacy gptq/awq rows may remain)
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

-- project_versions: pinned snapshots (datasets/files/brain index/runs/base model).
CREATE TABLE IF NOT EXISTS project_versions (
    id                TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL,
    version_number    INTEGER NOT NULL,
    label             TEXT NOT NULL DEFAULT '',
    notes             TEXT NOT NULL DEFAULT '',
    manifest_json     TEXT NOT NULL DEFAULT '{}',  -- pinned datasets/files/brain index/runs/base model
    parent_version_id TEXT,                          -- lineage: version this was built from
    created_at        REAL NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    UNIQUE(project_id, version_number)
);
CREATE INDEX IF NOT EXISTS idx_project_versions_project
    ON project_versions(project_id, version_number DESC);

-- mining: per-file mining jobs, the file's data-point inventory and every model round (db/mining.py).
CREATE TABLE IF NOT EXISTS mining_jobs (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    filename     TEXT NOT NULL DEFAULT '',
    kind         TEXT NOT NULL DEFAULT 'mine',      -- mine | fill
    status       TEXT NOT NULL DEFAULT 'queued',    -- queued | running | done | gaps | failed | cancelled | interrupted
    stage        TEXT NOT NULL DEFAULT 'queued',    -- queued | inventory | mining | checking | fixing | done
    message      TEXT NOT NULL DEFAULT '',
    stats_json   TEXT NOT NULL DEFAULT '{}',
    helper_json  TEXT NOT NULL DEFAULT '{}',
    error        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mining_jobs_project ON mining_jobs(project_id, source_id);
CREATE INDEX IF NOT EXISTS idx_mining_jobs_status  ON mining_jobs(status);

CREATE TABLE IF NOT EXISTS mining_items (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    job_id          TEXT NOT NULL DEFAULT '',
    chunk_idx       INTEGER NOT NULL DEFAULT 0,
    origin          TEXT NOT NULL,                  -- rule | agent | agent_file
    kind            TEXT NOT NULL DEFAULT 'prose',  -- prose | list | table | fact
    text            TEXT NOT NULL,
    header          TEXT NOT NULL DEFAULT '',
    values_json     TEXT NOT NULL DEFAULT '[]',
    excluded_reason TEXT NOT NULL DEFAULT '',
    excluded_at     REAL,
    proposed_reason TEXT NOT NULL DEFAULT '',       -- the helper judged it not a trainable fact; a reviewer confirms
    created_at      REAL NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mining_items_file ON mining_items(project_id, source_id);

CREATE TABLE IF NOT EXISTS mining_rounds (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    job_id        TEXT NOT NULL,
    chunk_idx     INTEGER NOT NULL DEFAULT 0,       -- 0 = the whole file
    kind          TEXT NOT NULL,                    -- inventory | inventory_file | mine | fix
    model         TEXT NOT NULL DEFAULT '',
    messages_json TEXT NOT NULL DEFAULT '[]',
    item_ids_json TEXT NOT NULL DEFAULT '[]',
    reply         TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    started_at    REAL NOT NULL,
    seconds       REAL NOT NULL DEFAULT 0,
    parsed        INTEGER NOT NULL DEFAULT 0,
    accepted      INTEGER NOT NULL DEFAULT 0,
    items_added   INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mining_rounds_file ON mining_rounds(project_id, source_id);
CREATE INDEX IF NOT EXISTS idx_mining_rounds_job  ON mining_rounds(job_id);

-- brain_builds: one row per index build attempt. Tracks timing + status +
-- final doc/chunk counts. brain_indexes holds the latest summary; this
-- table is the history.
CREATE TABLE IF NOT EXISTS brain_builds (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    index_id      TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued',
    started_at    REAL,
    finished_at   REAL,
    duration_ms   INTEGER,
    doc_count     INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    error         TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (index_id) REFERENCES brain_indexes(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_brain_builds_project ON brain_builds(project_id);
CREATE INDEX IF NOT EXISTS idx_brain_builds_index   ON brain_builds(index_id);

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
    phase         TEXT NOT NULL DEFAULT '',   -- queued | merging | converting | quantizing | abliterating | finalizing
    phase_detail  TEXT NOT NULL DEFAULT '',   -- e.g. "Q4_K_M (2/4)"
    heartbeat_at  REAL,                        -- last sign of life from the worker
    quants_json   TEXT NOT NULL DEFAULT '[]', -- every quant the job was asked for
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

-- Append-only request log for operations that do not have their own job
-- table (uploads, chat/query, model loads, and small mutations). Durable job
-- tables remain the source of progress; this table preserves the operation.
CREATE TABLE IF NOT EXISTS activity_events (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL,
    operation   TEXT NOT NULL,
    method      TEXT NOT NULL,
    path        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'done',
    http_status INTEGER NOT NULL DEFAULT 200,
    message     TEXT NOT NULL DEFAULT '',
    started_at  REAL NOT NULL,
    finished_at REAL NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activity_events_created ON activity_events(created_at);
CREATE INDEX IF NOT EXISTS idx_activity_events_project ON activity_events(project_id, created_at);

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
    format          TEXT NOT NULL,         -- 'parsed' = the file's one parse (knowledge.sources)
    converted_path  TEXT NOT NULL,         -- absolute path of files/<sha12>/parsed.txt
    converted_hash  TEXT NOT NULL,         -- sha256 of the parsed text ('' until parsed)
    converted_size  INTEGER NOT NULL,
    converter       TEXT NOT NULL,         -- parser name, 'manual-edit', '<parser>+vision'
    converted_at    REAL NOT NULL,
    status          TEXT NOT NULL,         -- 'queued' | 'parsing' | 'ready' | 'error'
    error_message   TEXT,
    char_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,   -- passages written to files/<sha12>/chunks/
    edited_at       REAL,                  -- last hand edit or Vision reading of the parsed text
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


BUSY_TIMEOUT_S = 30.0


def _connect() -> sqlite3.Connection:
    """Open a connection with foreign keys enabled and row factory set.

    `defer_foreign_keys = ON` lets us INSERT a child row before its parent
    is fully visible — useful when a route creates a parent (e.g. project)
    and a child (e.g. run) in the same request.

    WAL + a 30 s busy timeout: per-file mining jobs write from up to 16 threads at once, and the default rollback journal
    with sqlite's 5 s wait failed jobs with "database is locked" (readers no longer block writers either).
    """
    os.makedirs(os.path.dirname(settings.db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    if settings.db_path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA defer_foreign_keys = ON")
    return conn


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    """Short-lived cursor context. Commits on exit, rolls back on error, always closes."""
    # sqlite3's own ``with conn`` only commits/rolls back; it never closes.
    with closing(_connect()) as conn:
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


def _rename_rag_tables(c: sqlite3.Cursor) -> None:
    """One-time rename of the RAG era (2026-10-14): project_rags -> brain_indexes, rag_corpora -> brain_builds
    (rag_id -> index_id), the unused training_runs.rag_ids_json dropped, suite kind 'rag' -> 'brain', activity kinds
    rag_* -> brain_*. Runs before ``_SCHEMA`` so an upgraded DB keeps its rows; a new table that a half-finished run
    created empty next to the old one is dropped first. Idempotent."""
    old_index, old_builds = "project_" + "rags", "rag_" + "corpora"     # spelled apart: repo-wide renames skip them
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for old, new in ((old_index, "brain_indexes"), (old_builds, "brain_builds")):
        if old in tables and new in tables and not c.execute(f"SELECT 1 FROM {new} LIMIT 1").fetchone():
            c.execute(f"DROP TABLE {new}")
            tables.discard(new)
    if old_index in tables and "brain_indexes" not in tables:
        c.execute(f"ALTER TABLE {old_index} RENAME TO brain_indexes")
        c.execute("DROP INDEX IF EXISTS idx_rags_project")
    if old_builds in tables and "brain_builds" not in tables:
        c.execute(f"ALTER TABLE {old_builds} RENAME TO brain_builds")
        c.execute("ALTER TABLE brain_builds RENAME COLUMN rag_id TO index_id")
        c.execute("DROP INDEX IF EXISTS idx_rag_corpora_project")
        c.execute("DROP INDEX IF EXISTS idx_rag_corpora_rag")
    if "training_runs" in tables:
        cols = {r[1] for r in c.execute("PRAGMA table_info(training_runs)")}
        if "rag_ids_json" in cols:
            c.execute("ALTER TABLE training_runs DROP COLUMN rag_ids_json")
    if "benchmark_runs" in tables and "kind" in {r[1] for r in c.execute("PRAGMA table_info(benchmark_runs)")}:
        c.execute("UPDATE benchmark_runs SET kind = 'brain' WHERE kind = 'rag'")
    if "benchmark_runs" in tables and "config_json" in {r[1] for r in c.execute("PRAGMA table_info(benchmark_runs)")}:
        for rid, raw in c.execute("SELECT id, config_json FROM benchmark_runs WHERE kind = 'brain' "
                                  "AND config_json LIKE '%\"rag\"%'").fetchall():
            cfg = json.loads(raw or "{}")
            if "rag" in cfg:
                cfg["brain"] = cfg.pop("rag")
                c.execute("UPDATE benchmark_runs SET config_json = ? WHERE id = ?", (json.dumps(cfg), rid))
    if "activity_events" in tables:
        c.execute("UPDATE activity_events SET kind = 'brain_build' WHERE kind = 'rag_build'")
        c.execute("UPDATE activity_events SET kind = 'brain_query' WHERE kind = 'rag_" "query'")
    if "project_versions" in tables:
        for vid, raw in c.execute("SELECT id, manifest_json FROM project_versions "
                                  "WHERE manifest_json LIKE '%\"rag_corpora\"%'").fetchall():
            m = json.loads(raw or "{}")
            m["brain_builds"] = [{"build_id": b.get("build_id", ""), "index_id": b.get("rag_id", ""),
                                  "doc_count": b.get("doc_count", 0), "chunk_count": b.get("chunk_count", 0)}
                                 for b in m.pop("rag_corpora", None) or []]
            c.execute("UPDATE project_versions SET manifest_json = ? WHERE id = ?", (json.dumps(m), vid))


def init_db() -> None:
    """Create tables if they don't exist, then run additive migrations.

    Safe to call repeatedly. New columns added in newer code are detected
    via _safe_alter — repeated calls are no-ops.
    """
    with cursor() as c:
        _rename_rag_tables(c)
        c.executescript(_SCHEMA)
        # Migrations: widen tables that already exist on upgraded installs.
        for col in (
            "phase TEXT NOT NULL DEFAULT ''",
            "phase_detail TEXT NOT NULL DEFAULT ''",
            "heartbeat_at REAL",
            "quants_json TEXT NOT NULL DEFAULT '[]'",
        ):
            _safe_alter(c, f"ALTER TABLE model_exports ADD COLUMN {col}")
        c.execute("DROP TABLE IF EXISTS data_prep_runs")  # replaced by mining_jobs (per-file mining, 2026-10-13)
        _safe_alter(c, "ALTER TABLE mining_items ADD COLUMN proposed_reason TEXT NOT NULL DEFAULT ''")
        for col in ("char_count INTEGER NOT NULL DEFAULT 0", "chunk_count INTEGER NOT NULL DEFAULT 0", "edited_at REAL"):
            _safe_alter(c, f"ALTER TABLE file_conversions ADD COLUMN {col}")
        _safe_alter(c, "ALTER TABLE brain_indexes ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'")
        _safe_alter(c, "ALTER TABLE brain_indexes ADD COLUMN last_build_at REAL")
        _safe_alter(c, "ALTER TABLE brain_indexes ADD COLUMN last_build_status TEXT")
        _safe_alter(c, "ALTER TABLE brain_indexes ADD COLUMN error TEXT NOT NULL DEFAULT ''")
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
        _safe_alter(c, "ALTER TABLE benchmark_runs ADD COLUMN model_path TEXT NOT NULL DEFAULT ''")
        # A test run is a job now: raw transcripts are saved as it goes, judging is a second, separate job.
        for col in (
            "status TEXT NOT NULL DEFAULT 'done'",          # queued | running | done | failed | cancelled
            "kind TEXT NOT NULL DEFAULT 'suite'",           # suite | brain | heldout | training_leakage | public
            "scoring TEXT NOT NULL DEFAULT 'judge'",        # judge (AI/human) | exact (official public benchmark)
            "config_json TEXT NOT NULL DEFAULT '{}'",       # suite path, top_k, corpus, sampling, dataset ...
            "error TEXT NOT NULL DEFAULT ''",
            "progress_done INTEGER NOT NULL DEFAULT 0",
            "progress_total INTEGER NOT NULL DEFAULT 0",
            "heartbeat_at REAL",
            "judge_status TEXT NOT NULL DEFAULT ''",        # '' (never) | running | done | failed | cancelled
            "judge_provider_id TEXT NOT NULL DEFAULT ''",
            "judge_model TEXT NOT NULL DEFAULT ''",
            "judge_done INTEGER NOT NULL DEFAULT 0",
            "judge_total INTEGER NOT NULL DEFAULT 0",
            "judge_error TEXT NOT NULL DEFAULT ''",
        ):
            _safe_alter(c, f"ALTER TABLE benchmark_runs ADD COLUMN {col}")
        _safe_alter(c, "ALTER TABLE benchmark_cases ADD COLUMN keywords TEXT NOT NULL DEFAULT '[]'")
        _safe_alter(c, "ALTER TABLE benchmark_cases ADD COLUMN expect_abstain INTEGER NOT NULL DEFAULT 0")
        _safe_alter(c, "ALTER TABLE benchmark_cases ADD COLUMN time_ms REAL NOT NULL DEFAULT 0")
        _safe_alter(c, "ALTER TABLE case_judgements ADD COLUMN prompt_version TEXT NOT NULL DEFAULT ''")
        c.execute(  # runs saved before progress was tracked: they are complete, so progress = their cases
            "UPDATE benchmark_runs SET "
            "progress_total = (SELECT COUNT(*) FROM benchmark_cases WHERE benchmark_id = benchmark_runs.id), "
            "progress_done = (SELECT COUNT(*) FROM benchmark_cases WHERE benchmark_id = benchmark_runs.id) "
            "WHERE progress_total = 0 AND status = 'done'"
        )
        c.execute(  # official public benchmarks (HF MMLU/GSM8K ...) were always exact-match scored
            "UPDATE benchmark_runs SET scoring = 'exact', kind = 'public' "
            "WHERE scoring = 'judge' AND kind = 'suite' AND scores_json LIKE '%\"is_real_benchmark\": true%'"
        )
        # Verdicts written before case_judgements existed become judgement rows, once, so the history is complete.
        c.execute(
            "INSERT INTO case_judgements (id, case_id, benchmark_id, kind, provider_id, judge_model, verdict, "
            "reasoning, confidence, error, created_at) "
            "SELECT lower(hex(randomblob(6))), bc.id, bc.benchmark_id, "
            "  CASE WHEN br.scoring = 'exact' THEN 'exact' WHEN bc.judge = 'human' THEN 'human' "
            "       WHEN bc.judge IN ('ai', 'local') THEN 'ai' ELSE 'scripted' END, "
            "  '', bc.judge_model, bc.verdict, bc.judge_reasoning, NULL, '', COALESCE(bc.scored_at, 0) "
            "FROM benchmark_cases bc JOIN benchmark_runs br ON br.id = bc.benchmark_id "
            "WHERE bc.verdict != '' AND NOT EXISTS (SELECT 1 FROM case_judgements cj WHERE cj.case_id = bc.id)"
        )
        # Auto-generated "Run · <dataset file>" names collide for every run on
        # the same dataset; tag them with the run id so selectors can tell
        # runs (and their exports) apart. Custom names are left untouched.
        c.execute(
            "UPDATE training_runs SET name = 'Run ' || id || ' · ' || substr(name, 7) "
            "WHERE name LIKE 'Run · %' AND instr(name, id) = 0"
        )


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    # Decode JSON columns (TEXT → dict under the unsuffixed key).
    for col in (
        "settings_json",
        "metrics_json",
        "scores_json",
        "options_json",
        "stats_json",
        "helper_json",
        "values_json",
        "messages_json",
        "item_ids_json",
    ):
        if col in d and isinstance(d[col], str) and d[col]:
            try:
                d[col.removesuffix("_json")] = json.loads(d[col])
            except json.JSONDecodeError:
                d[col.removesuffix("_json")] = None
            d.pop(col, None)
    return d
