"""Project export page helpers — expand-row context for trained exports.

Dir contents + model load reuse existing APIs in ``project_models`` and
``models``; this module only builds template context for the export page.
"""
from __future__ import annotations

from finetune_studio.training.run_export import (
    adapter_dir_ready,
    merged_dir_ready,
)


def runs_by_id(runs: list[dict]) -> dict[str, dict]:
    """Index training runs by id for expand-row source-run lookups."""
    return {r["id"]: r for r in runs if r.get("id")}


def export_path_row_id(path: str) -> str:
    """Stable DOM id / URL hash slug for an export path (``#m-…``)."""
    slug = (
        str(path or "")
        .replace("/", "-")
        .replace(" ", "_")
        .replace(".", "_")
    )
    return f"m-{slug}"


def annotate_runs_for_export(runs: list[dict]) -> list[dict]:
    """Copy runs with ``has_merged`` / ``has_adapter`` / ``exportable`` flags.

    A completed run is exportable when it has a merged model **or** a raw
    adapter (export will merge onto a compatible base at request time).
    """
    annotated: list[dict] = []
    for run in runs:
        row = dict(run)
        out = (row.get("output_path") or "").strip()
        has_merged = bool(out) and merged_dir_ready(out)
        has_adapter = bool(out) and adapter_dir_ready(out)
        status = (row.get("status") or "").lower()
        status_ok = status in ("done", "completed", "failed")
        row["has_merged"] = has_merged
        row["has_adapter"] = has_adapter
        row["exportable"] = status_ok and (has_merged or has_adapter)
        annotated.append(row)
    return annotated