"""Project dataset build: coverage gate → export → persisted + registered dataset.

One owner for the steps behind ``GET /projects/{pid}/data-prep/export`` and
``fts dataset build``, so the route and the CLI can never drift:

1. ``coverage_gate`` — the deterministic coverage-fill pass; a chunk that still
   has no usable Q&A blocks the export (``ExportBlocked``) unless ``force``.
2. ``data.prep.export.build_qa_export`` — dedupe, optional grounded rows, serialise.
3. ``persist_export`` — write the JSONL into the project's datasets dir and
   register it (create or update) so the Training tab can select it.

Errors are typed and honest: no step swallows a failure. The route decides how
to map them to HTTP; the CLI maps them to an exit status.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune_studio.data.prep.export import ExportResult, build_qa_export
from finetune_studio.data.prep.grounding import GroundingOptions

log = logging.getLogger("fts.dataset_build")

EXPORT_FORMATS = ("sharegpt", "alpaca", "openai")
STATUS_FILTERS = ("approved", "pending", "rejected", "all")
_NAMES_SHOWN = 5


class ExportBlocked(Exception):
    """The coverage gate found chunks with no usable Q&A (and ``force`` was off)."""

    def __init__(self, uncovered_chunks: list[dict[str, Any]]) -> None:
        self.uncovered_chunks = uncovered_chunks
        self.files = sorted({str(u.get("filename") or u.get("source") or "?")
                             for u in uncovered_chunks})
        shown = ", ".join(self.files[:_NAMES_SHOWN]) + (" and more" if len(self.files) > _NAMES_SHOWN else "")
        super().__init__(f"dataset export blocked: no usable Q&A could be made from {shown}. "
                         "Delete that file, or export anyway without it.")


class CoverageCheckFailed(Exception):
    """The coverage-fill pass itself crashed — never export without verifying coverage."""


def coverage_gate(pid: str, *, force: bool = False) -> dict[str, Any]:
    """Run the fill pass; return its summary. Raises ``ExportBlocked`` / ``CoverageCheckFailed``.

    Fills the chunks the stochastic mining pass never converted as approved
    extractive pairs, so a dataset never ships with silent coverage holes.
    With ``force`` the remaining uncovered chunks are logged, not blocking.
    """
    try:
        from finetune_studio.data.prep.coverage_fill import fill_all_project_gaps
        summary = fill_all_project_gaps(pid)
    except Exception as exc:
        log.exception("coverage fill failed — export blocked")
        raise CoverageCheckFailed(f"dataset export blocked: coverage verification failed: {exc}") from exc
    uncovered = list((summary or {}).get("uncovered_chunks") or [])
    if uncovered and not force:
        raise ExportBlocked(uncovered)
    if uncovered:
        log.warning("export: force=true — exporting project %s with %d uncovered chunk(s)",
                    pid, len(uncovered))
    return summary or {}


@dataclass(frozen=True)
class PersistedDataset:
    path: Path
    rows: int
    dataset: dict[str, Any] = field(default_factory=dict)


def register_dataset_file(pid: str, filename: str, body: str, display: str, *,
                          source: str) -> PersistedDataset:
    """Write ``body`` to ``<datasets_dir>/<filename>`` and create/update its registry row.

    The one place a built dataset becomes a ``project_datasets`` row, shared by the
    SFT export and the preference builder. Re-writing the same file updates the row.
    """
    from finetune_studio.db.datasets import (
        count_qa_pairs,
        create_dataset,
        datasets_dir,
        get_dataset_by_path,
        update_dataset,
    )
    target = datasets_dir(pid) / filename
    target.write_text(body, encoding="utf-8")
    rows = count_qa_pairs(str(target))
    display = display.replace("{rows}", str(rows))
    existing = get_dataset_by_path(pid, str(target))
    if existing:
        update_dataset(existing["id"], name=display, qa_count=rows, size_bytes=target.stat().st_size)
        ds = {**existing, "name": display, "qa_count": rows}
    else:
        ds = create_dataset(project_id=pid, name=display, data_path=str(target),
                            source=source, qa_count=rows, size_bytes=target.stat().st_size)
    return PersistedDataset(path=target, rows=rows, dataset=ds)


def persist_export(pid: str, fmt: str, only: str, body: str, grounded_rows: int = 0,
                   *, name: str | None = None) -> PersistedDataset:
    """Write ``body`` into the project's datasets dir and register/update the dataset row.

    Default file ``<pid>-<fmt>-<only>.jsonl`` and display name
    ``<project> · <fmt> · <N> rows`` (re-exporting updates the same registry row);
    ``name`` replaces both with a sanitised custom label.
    """
    from finetune_studio import db
    if name is not None:
        stem = re.sub(r"[^\w.\-]", "_", name).strip("._")
        if not stem:
            raise ValueError(f"dataset name {name!r} has no usable characters")
        fname = f"{pid}-{stem}.jsonl"
        display = name
    else:
        fname = f"{pid}-{fmt}-{only}.jsonl"
        # Readable registry name (Genor 2026-09-20): project · format · rows — never a bare pid hash.
        proj = db.get_project(pid) or {}
        display = f"{proj.get('name') or pid} · {fmt} · {{rows}} rows"
    if grounded_rows:
        display += f" ({grounded_rows} with retrieved context)"
    return register_dataset_file(pid, fname, body, display, source="data-prep-export")


@dataclass(frozen=True)
class BuiltDataset:
    """Everything the CLI reports about one build."""

    persisted: PersistedDataset
    export: ExportResult
    coverage: dict[str, Any]

    @property
    def grounded_rows(self) -> int:
        return self.export.grounding.grounded if self.export.grounding else 0


def build_project_dataset(pid: str, fmt: str = "sharegpt", only: str = "approved", *,
                          grounding: GroundingOptions | None = None, force: bool = False,
                          name: str | None = None) -> BuiltDataset:
    """Gate → export → persist. Raises ``ExportBlocked``, ``CoverageCheckFailed`` or ``ValueError``."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unknown format: {fmt} (expected one of {', '.join(EXPORT_FORMATS)})")
    if only not in STATUS_FILTERS:
        raise ValueError(f"unknown only filter: {only} (expected one of {', '.join(STATUS_FILTERS)})")
    coverage = coverage_gate(pid, force=force)
    result = build_qa_export(pid, fmt, only, grounding=grounding)
    if result.rows == 0:
        raise ValueError(f"no {only} Q&A pairs to export for this project")
    grounded = result.grounding.grounded if result.grounding else 0
    persisted = persist_export(pid, fmt, only, result.body, grounded, name=name)
    return BuiltDataset(persisted=persisted, export=result, coverage=coverage)
