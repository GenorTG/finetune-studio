"""Project export page helpers — expand-row context for trained exports.

Dir contents + model load reuse existing APIs in ``project_models`` and
``models``; this module only builds template context for the export page.
"""
from __future__ import annotations


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
