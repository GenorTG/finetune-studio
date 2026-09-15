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
