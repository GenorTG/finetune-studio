"""Resolve project-scoped models for the Testing tab.

Keeps inference auto-load and the model selector aligned with on-disk
exports under each run's ``output_path`` (merged/, gguf/, …), not the
global HF discovery list.
"""

from __future__ import annotations

import os
from typing import Any

from finetune_studio.training.run_export import merged_dir_ready

# Status values written by the training engine / older callers.
_DONE_STATUSES = frozenset({"done", "completed"})

# Formats the Testing inference engine can load (HF dir or GGUF).
_INFERENCE_FORMATS = frozenset({"safetensors", "gguf"})

BASE_MODEL_CHOICE = "__base__"   # the Testing / Compare pages' "untrained base model" entry


def local_model_missing(path: str) -> bool:
    """True when ``path`` names a local file/dir that is not on disk.

    Hub ids (``org/name``) and empty values are not local paths, so they are never "missing".
    """
    p = (path or "").strip()
    return p.startswith(("/", "~", "./", "../")) and not os.path.exists(os.path.expanduser(p))


def is_run_done(run: dict[str, Any]) -> bool:
    """True when a training run is finished successfully."""
    return (run.get("status") or "").lower() in _DONE_STATUSES


def resolve_latest_merged_model(pid: str, *, list_runs_fn: Any = None) -> str | None:
    """Return path to the newest completed run's ready ``merged/`` dir.

    Skips runs with empty ``output_path`` or without weight files under
    ``merged/``. ``list_runs_fn`` defaults to ``db.list_runs`` (injectable
    for tests).
    """
    if list_runs_fn is None:
        from finetune_studio import db

        list_runs_fn = db.list_runs

    runs = list_runs_fn(pid)
    for run in runs:  # newest-first
        if not is_run_done(run):
            continue
        output_path = (run.get("output_path") or "").strip()
        if not output_path:
            continue
        if not merged_dir_ready(output_path):
            continue
        return os.path.join(output_path, "merged")
    return None


def models_for_testing_page(project_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter + sort project export rows for the Testing model selector.

    Prefer safetensors (merged/abliterated), then gguf. Newest mtime first.
    """
    usable = [
        m
        for m in project_models
        if (m.get("format") or "").lower() in _INFERENCE_FORMATS
        and (m.get("path") or "").strip()
    ]
    usable.sort(key=lambda m: float(m.get("mtime") or 0), reverse=True)
    return usable


def default_model_path_for_testing(models: list[dict[str, Any]]) -> str | None:
    """Path of the preferred default selection (latest merged safetensors)."""
    for m in models:
        path = (m.get("path") or "").rstrip("/\\")
        if (m.get("format") or "").lower() == "safetensors" and path.endswith(
            "merged"
        ):
            return m.get("path")
    return models[0]["path"] if models else None


def target_model(pid: str, override: str) -> str:
    """The model a test run will use: the explicit choice, the project's untrained base, else the latest merged export.

    Never "whatever happens to be loaded": the helper or another page's model may be resident.
    A local path that is no longer on disk is refused here, before a run is created.
    """
    if override == BASE_MODEL_CHOICE:
        from finetune_studio import db
        from finetune_studio.db.runs import backfill_project_base_model

        base = backfill_project_base_model(pid) or str((db.get_project(pid) or {}).get("base_model") or "")
        if not base.strip():
            raise ValueError("this project has no base model recorded yet; train a run or set one under project Settings")
        chosen = base.strip()
    elif override:
        chosen = override
    else:
        chosen = resolve_latest_merged_model(pid) or ""
        if not chosen:
            raise ValueError("no model to test: pick one in the Model list, or run training + merge/export first")
    if local_model_missing(chosen):
        raise ValueError(f"model not found on disk: {chosen} (re-create it, or pick another model)")
    return chosen
