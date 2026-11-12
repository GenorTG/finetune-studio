"""Result diversity: one document may not fill the top results.

Tarrowfield (2026-11-12): one e-mail thread filled ranks 1-3 for in-001, and the deciding passage of a multi-part
question sat at rank 9-10 behind passages of the same few documents. ``diversify`` walks the final ranking (after rank
fusion and the reranker) and keeps at most ``max_per_document`` passages of each document until ``top_k`` are chosen;
the passages it skipped are not cut but appended in their order when other documents run out (a one-document corpus
still gets ``top_k`` passages). Scores are kept as they were; only the order and the selection change.

``standalone_server.py`` ships verbatim with stdlib + numpy, so it carries its own copy of this function;
``tests/test_search_diversity.py`` pins the two together.
"""
from __future__ import annotations

from typing import Any

DEFAULT_MAX_PER_DOCUMENT = 3
CANDIDATES_PER_RESULT = 4      # without a reranker the fused ranking is read this many times deeper than top_k


def diversify(hits: list[dict[str, Any]], top_k: int, max_per_document: int) -> list[dict[str, Any]]:
    """The first ``top_k`` of ``hits`` with at most ``max_per_document`` per document first, then the overflow in
    order (0 or less = no cap: ``hits[:top_k]``)."""
    if max_per_document <= 0:
        return hits[:top_k]
    kept: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    per_doc: dict[str, int] = {}
    for h in hits:
        if len(kept) >= top_k:
            break
        key = str(h.get("document_id") or h.get("filename") or h.get("source") or "")
        if per_doc.get(key, 0) < max_per_document:
            per_doc[key] = per_doc.get(key, 0) + 1
            kept.append(h)
        else:
            overflow.append(h)
    return kept + overflow[: max(0, top_k - len(kept))]
