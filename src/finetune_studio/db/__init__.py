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
)
from finetune_studio.db.rags import (
    create_rag, delete_rag, get_rag, list_rags, update_rag,
)
from finetune_studio.db.runs import (
    create_run, delete_run, get_run, list_runs, update_run,
)
from finetune_studio.db.datasets import (
    create_dataset, delete_dataset, datasets_dir, get_dataset, list_datasets,
    update_dataset, count_qa_pairs,
)
from finetune_studio.db.benchmarks import (
    create_benchmark, get_benchmark, list_benchmarks,
)
from finetune_studio.db.reviews import list_review, record_review

# Initialise on import so callers don't have to remember.
init_db()
