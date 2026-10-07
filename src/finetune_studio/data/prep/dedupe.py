"""Near-duplicate pairs: the same fact asked twice in slightly different words.

Exhaustive mining writes one pair per fact and, run over a table, often rewords the same field question ("What is the stage of the
X deal" / "... the X opportunity"). Those pairs carry no new fact but cost review time and over-weight the fact in training.
Two pairs are near-duplicates when their answers state the same distinctive tokens (numbers, codes, names) AND their questions
share most of their content words. A pair that adds a token, or asks about something else, is never a duplicate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from finetune_studio.data.prep.exhaustive import canon, distinctive_tokens
from finetune_studio.data.prep.qa_validate import content_tokens

QUESTION_JACCARD = 0.6


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _answer_key(answer: str) -> frozenset[str]:
    return distinctive_tokens(answer) or frozenset(canon(t) for t in content_tokens(answer))


def find_near_duplicates(pairs: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """``[(duplicate_id, kept_id), ...]``: the first pair of each cluster is kept, in input order, per source and chunk."""
    kept: list[tuple[str, frozenset[str], set[str], tuple[Any, Any]]] = []
    out: list[tuple[str, str]] = []
    for p in pairs:
        pid = str(p.get("id"))
        scope = (p.get("source_id"), p.get("chunk_idx"))
        key = _answer_key(str(p.get("answer", "")))
        qtoks = content_tokens(str(p.get("question", "")))
        twin = next((k_id for k_id, k_key, k_q, k_scope in kept
                     if k_scope == scope and k_key == key and _jaccard(qtoks, k_q) >= QUESTION_JACCARD), None)
        if twin is not None:
            out.append((pid, twin))
        else:
            kept.append((pid, key, qtoks, scope))
    return out
