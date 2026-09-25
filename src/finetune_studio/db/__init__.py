"""SQLite-backed project/run/RAG storage.

WHAT THIS MODULE DOES
=====================
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

LAYOUT
======
db/
  __init__.py    — this file (re-exports + init on import)
  connection.py  — low-level conn / cursor / row_to_dict / schema
  projects.py    — CRUD for `projects`
  rags.py        — CRUD for `project_rags`
  runs.py        — CRUD for `training_runs`
  benchmarks.py  — CRUD for `benchmark_runs`
  reviews.py     — row-level approve/reject/edit decisions
"""

from finetune_studio.db.activity_events import (
    list_recent as list_activity_events_recent,
)
from finetune_studio.db.activity_events import record as record_activity_event
from finetune_studio.db.benchmarks import (
    create_benchmark,
    create_case,
    get_benchmark,
    list_benchmarks,
    list_cases,
    update_benchmark_scores,
    update_case,
)
from finetune_studio.db.benchmarks import (
    list_recent as list_benchmarks_recent,
)
from finetune_studio.db.connection import cursor, init_db, row_to_dict
from finetune_studio.db.data_prep_runs import (
    create_run as create_data_prep_run,
)
from finetune_studio.db.data_prep_runs import (
    get_run as get_data_prep_run,
)
from finetune_studio.db.data_prep_runs import (
    list_for_project as list_data_prep_for_project,
)
from finetune_studio.db.data_prep_runs import (
    list_recent as list_data_prep_recent,
)
from finetune_studio.db.data_prep_runs import (
    mark_done as mark_data_prep_done,
)
from finetune_studio.db.data_prep_runs import (
    mark_failed as mark_data_prep_failed,
)
from finetune_studio.db.data_prep_runs import (
    mark_running as mark_data_prep_running,
)
from finetune_studio.db.data_prep_runs import (
    reconcile_stale as reconcile_stale_data_prep,
)
from finetune_studio.db.data_prep_runs import (
    update_run as update_data_prep_run,
)
from finetune_studio.db.datasets import (
    count_qa_pairs,
    create_dataset,
    datasets_dir,
    delete_dataset,
    get_dataset,
    list_datasets,
    update_dataset,
)
from finetune_studio.db.hf_downloads import (
    create_job as create_hf_download,
)
from finetune_studio.db.hf_downloads import (
    get_job as get_hf_download,
)
from finetune_studio.db.hf_downloads import (
    list_in_progress as list_hf_downloads_in_progress,
)
from finetune_studio.db.hf_downloads import (
    list_recent as list_hf_downloads_recent,
)
from finetune_studio.db.hf_downloads import (
    mark_cancelled as mark_hf_download_cancelled,
)
from finetune_studio.db.hf_downloads import (
    mark_done as mark_hf_download_done,
)
from finetune_studio.db.hf_downloads import (
    mark_failed as mark_hf_download_failed,
)
from finetune_studio.db.hf_downloads import (
    mark_running as mark_hf_download_running,
)
from finetune_studio.db.hf_downloads import (
    update_job as update_hf_download,
)
from finetune_studio.db.model_exports import (
    create_export,
    get_export,
    update_export,
)
from finetune_studio.db.model_exports import (
    list_for_project as list_exports_for_project,
)
from finetune_studio.db.model_exports import (
    list_for_run as list_exports_for_run,
)
from finetune_studio.db.model_exports import (
    list_recent as list_exports_recent,
)
from finetune_studio.db.model_exports import (
    mark_done as mark_export_done,
)
from finetune_studio.db.model_exports import (
    mark_failed as mark_export_failed,
)
from finetune_studio.db.model_exports import (
    mark_running as mark_export_running,
)
from finetune_studio.db.model_exports import (
    reconcile_stale as reconcile_stale_exports,
)
from finetune_studio.db.project_versions import (
    create_version,
    delete_version,
    get_version,
    get_version_by_number,
    latest_version,
    list_versions,
    version_lineage,
)
from finetune_studio.db.projects import (
    add_model_favorite,
    create_project,
    delete_project,
    get_project,
    is_model_favorited,
    list_model_favorites,
    list_projects,
    remove_model_favorite,
    update_project,
)
from finetune_studio.db.rag_corpora import (
    create_build as create_rag_build,
)
from finetune_studio.db.rag_corpora import (
    get_build as get_rag_build,
)
from finetune_studio.db.rag_corpora import (
    latest_for_rag as latest_rag_build,
)
from finetune_studio.db.rag_corpora import (
    list_for_rag as list_rag_builds,
)
from finetune_studio.db.rag_corpora import (
    list_recent as list_rag_builds_recent,
)
from finetune_studio.db.rag_corpora import (
    mark_done as mark_rag_build_done,
)
from finetune_studio.db.rag_corpora import (
    mark_failed as mark_rag_build_failed,
)
from finetune_studio.db.rag_corpora import (
    mark_running as mark_rag_build_running,
)
from finetune_studio.db.rag_corpora import (
    reconcile_stale as reconcile_stale_rag_builds,
)
from finetune_studio.db.rag_corpora import (
    update_build as update_rag_build,
)
from finetune_studio.db.rags import (
    create_rag,
    delete_rag,
    ensure_portable_rag,
    get_rag,
    list_rags,
    update_rag,
)
from finetune_studio.db.reviews import list_review, record_review
from finetune_studio.db.runs import (
    create_run,
    delete_run,
    get_run,
    list_runs,
    reconcile_stale_runs,
    update_run,
)
from finetune_studio.db.system_updates import (
    append_log as append_update_log,
)
from finetune_studio.db.system_updates import (
    create_update,
    get_update,
    update_update,
)
from finetune_studio.db.system_updates import (
    latest_in_progress as latest_update_in_progress,
)
from finetune_studio.db.system_updates import (
    list_recent as list_updates_recent,
)
from finetune_studio.db.system_updates import (
    mark_cancelled as mark_update_cancelled,
)
from finetune_studio.db.system_updates import (
    mark_done as mark_update_done,
)
from finetune_studio.db.system_updates import (
    mark_failed as mark_update_failed,
)
from finetune_studio.db.system_updates import (
    mark_running as mark_update_running,
)
from finetune_studio.db.system_updates import (
    reconcile_stale as reconcile_stale_updates,
)

# Public surface of the DB layer. Every name below is re-exported for
# `db.<name>` attribute access (56 call sites depend on it) and is NOT dead
# code — without this list ruff F401 flags the whole facade as unused.
# Keep in sync when adding a re-export below.
__all__ = ["add_model_favorite", "append_update_log", "count_qa_pairs",
    "create_benchmark", "create_case", "create_data_prep_run", "create_dataset",
    "create_export", "create_hf_download", "create_project", "create_rag",
    "create_rag_build", "create_run", "create_update", "create_version", "cursor",
    "datasets_dir", "delete_dataset", "delete_project", "delete_rag", "delete_run",
    "delete_version", "ensure_portable_rag", "get_benchmark", "get_data_prep_run",
    "get_dataset", "get_export", "get_hf_download", "get_project", "get_rag",
    "get_rag_build", "get_run", "get_update", "get_version", "get_version_by_number",
    "init_db", "is_model_favorited", "latest_rag_build", "latest_update_in_progress",
    "latest_version", "list_activity_events_recent", "list_benchmarks",
    "list_benchmarks_recent", "list_cases", "list_data_prep_for_project",
    "list_data_prep_recent", "list_datasets", "list_exports_for_project",
    "list_exports_for_run", "list_exports_recent", "list_hf_downloads_in_progress",
    "list_hf_downloads_recent", "list_model_favorites", "list_projects",
    "list_rag_builds", "list_rag_builds_recent", "list_rags", "list_review",
    "list_runs", "list_updates_recent", "list_versions", "mark_data_prep_done",
    "mark_data_prep_failed", "mark_data_prep_running", "mark_export_done",
    "mark_export_failed", "mark_export_running", "mark_hf_download_cancelled",
    "mark_hf_download_done", "mark_hf_download_failed", "mark_hf_download_running",
    "mark_rag_build_done", "mark_rag_build_failed", "mark_rag_build_running",
    "mark_update_cancelled", "mark_update_done", "mark_update_failed",
    "mark_update_running", "reconcile_stale_data_prep", "reconcile_stale_exports",
    "reconcile_stale_rag_builds", "reconcile_stale_runs", "reconcile_stale_updates",
    "record_activity_event", "record_review", "remove_model_favorite", "row_to_dict",
    "update_benchmark_scores", "update_case", "update_data_prep_run", "update_dataset",
    "update_export", "update_hf_download", "update_project", "update_rag",
    "update_rag_build", "update_run", "update_update", "version_lineage"]

# Initialise on import so callers don't have to remember.
init_db()
