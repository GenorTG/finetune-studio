"""Helpers for the project file-browser page (``/projects/{pid}/data``).

Pure functions — no FastAPI routes. Call from ``project_data_page``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.webui.project_dashboard import truncate_description


def format_bytes(n: float | None) -> str:
    """Humanise a byte count (B / KB / MB / GB)."""
    try:
        size = float(n or 0)
    except (TypeError, ValueError):
        size = 0.0
    if size < 1024:
        return f"{int(size)} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def file_browser_stats(files: list[dict]) -> dict[str, Any]:
    """Counts + storage from ``list_files(include_deleted=True)`` rows."""
    active = 0
    trash = 0
    storage = 0
    for row in files:
        try:
            storage += int(row.get("size_bytes") or 0)
        except (TypeError, ValueError):
            pass
        if row.get("deleted_at"):
            trash += 1
        else:
            active += 1
    return {
        "total": active + trash,
        "library": active,
        "trash": trash,
        "storage_bytes": storage,
        "storage_label": format_bytes(storage),
    }


def runs_started_max(runs: list[dict]) -> float | None:
    """Latest ``started_at`` across project runs (for drift badges)."""
    vals: list[float] = []
    for run in runs:
        started = run.get("started_at")
        if started is None:
            continue
        try:
            vals.append(float(started))
        except (TypeError, ValueError):
            continue
    return max(vals) if vals else None


def build_file_browser_ctx(
    project: dict,
    *,
    files: list[dict] | None = None,
) -> dict[str, Any]:
    """Assemble template extras for ``project_data.html``."""
    file_rows = files if files is not None else []
    desc = truncate_description(project.get("description"))
    stats = file_browser_stats(file_rows)
    return {
        "desc_short": desc["short"],
        "desc_long": desc["long"],
        "desc_full": desc["full"],
        "fb_stats": stats,
        "runs_started_max": runs_started_max(project.get("runs") or []),
        "format_bytes": format_bytes,
    }
