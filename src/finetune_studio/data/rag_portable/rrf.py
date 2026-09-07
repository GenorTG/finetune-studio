"""Reciprocal Rank Fusion.

Each input list is a ranking (rank 1 first). The output score for a doc-id is
sum of 1/(k + rank) across all rankings where the doc appears.
"""
from __future__ import annotations

from finetune_studio.data.rag_portable.constants import RRF_K


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Each ranked_lists[i] is a list of doc-ids from rank 1 down. Returns {doc_id: rrf_score}."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
