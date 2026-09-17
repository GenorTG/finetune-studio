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

from finetune_studio.db.connection import cursor, init_db, row_to_dict
from finetune_studio.db.projects import (
    create_project, delete_project, get_project, list_projects, update_project,
    add_model_favorite, list_model_favorites, remove_model_favorite,
    is_model_favorited,
)
from finetune_studio.db.rags import (
    create_rag, delete_rag, ensure_portable_rag, get_rag, list_rags, update_rag,
)
from finetune_studio.db.runs import (
    create_run, delete_run, get_run, list_runs, update_run,
    reconcile_stale_runs,
)
from finetune_studio.db.datasets import (
    create_dataset, delete_dataset, datasets_dir, get_dataset, list_datasets,
    update_dataset, count_qa_pairs,
)
from finetune_studio.db.benchmarks import (
    create_benchmark, get_benchmark, list_benchmarks,
    list_recent as list_benchmarks_recent,
    create_case, list_cases, update_case, update_benchmark_scores,
)
from finetune_studio.db.reviews import list_review, record_review
from finetune_studio.db.data_prep_runs import (
    create_run as create_data_prep_run,
    get_run as get_data_prep_run,
    list_for_project as list_data_prep_for_project,
    list_recent as list_data_prep_recent,
    mark_done as mark_data_prep_done,
    mark_failed as mark_data_prep_failed,
    mark_running as mark_data_prep_running,
    update_run as update_data_prep_run,
)
from finetune_studio.db.rag_corpora import (
    create_build as create_rag_build,
    get_build as get_rag_build,
    latest_for_rag as latest_rag_build,
    list_for_rag as list_rag_builds,
    list_recent as list_rag_builds_recent,
    mark_done as mark_rag_build_done,
    mark_failed as mark_rag_build_failed,
    mark_running as mark_rag_build_running,
    update_build as update_rag_build,
)
from finetune_studio.db.hf_downloads import (
    create_job as create_hf_download,
    get_job as get_hf_download,
    list_in_progress as list_hf_downloads_in_progress,
    list_recent as list_hf_downloads_recent,
    mark_cancelled as mark_hf_download_cancelled,
    mark_done as mark_hf_download_done,
    mark_failed as mark_hf_download_failed,
    mark_running as mark_hf_download_running,
    update_job as update_hf_download,
)
from finetune_studio.db.model_exports import (
    create_export,
    get_export,
    list_for_project as list_exports_for_project,
    list_for_run as list_exports_for_run,
    list_recent as list_exports_recent,
    mark_done as mark_export_done,
    mark_failed as mark_export_failed,
    mark_running as mark_export_running,
    update_export,
)
from finetune_studio.db.system_updates import (
    append_log as append_update_log,
    create_update,
    get_update,
    latest_in_progress as latest_update_in_progress,
    list_recent as list_updates_recent,
    mark_cancelled as mark_update_cancelled,
    reconcile_stale as reconcile_stale_updates,
    mark_done as mark_update_done,
    mark_failed as mark_update_failed,
    mark_running as mark_update_running,
    update_update,
)
from finetune_studio.db.activity_events import list_recent as list_activity_events_recent, record as record_activity_event

# Initialise on import so callers don't have to remember.
init_db()
