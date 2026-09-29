"""Derived per-file data-prep state.

Readiness is computed from durable source and pair artifacts. A mutable flag
would go stale whenever a file is reparsed, a pair is rejected, or the source
revision changes.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.data import project_filesystem as pfs


def summarize_source(pid: str, source: dict[str, Any]) -> dict[str, Any]:
    """Return parse, review, coverage, and training-readiness for one source."""
    source_id = str(source.get("id") or "")
    pairs = pfs.list_qa_pairs(pid, source_id=source_id)
    approved = [p for p in pairs if p.get("status") == "approved"]
    pending = [p for p in pairs if p.get("status") == "pending"]
    rejected = [p for p in pairs if p.get("status") == "rejected"]
    chunks_total = max(0, int(source.get("chunk_count") or 0))
    approved_chunks = {
        int(p.get("chunk_idx") or 0)
        for p in approved
        if 1 <= int(p.get("chunk_idx") or 0) <= chunks_total
    }
    covered_chunks = len(approved_chunks)
    coverage_pct = (
        round(100.0 * covered_chunks / chunks_total, 1) if chunks_total else 0.0
    )
    parse_status = str(source.get("status") or "registered")
    training_ready = (
        parse_status not in {"queued", "parsing", "error", "registered"}
        and chunks_total > 0
        and covered_chunks == chunks_total
        and len(approved) > 0
    )
    if parse_status in {"queued", "parsing", "error", "registered"}:
        stage = parse_status
    elif training_ready:
        stage = "training_ready"
    elif pairs:
        stage = "needs_review"
    else:
        stage = "parsed"
    return {
        "stage": stage,
        "training_ready": training_ready,
        "pairs_total": len(pairs),
        "pairs_approved": len(approved),
        "pairs_pending": len(pending),
        "pairs_rejected": len(rejected),
        "chunks_total": chunks_total,
        "chunks_approved": covered_chunks,
        "coverage_pct": coverage_pct,
    }
