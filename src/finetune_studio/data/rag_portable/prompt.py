"""The one definition of the RAG chat prompt layout (stdlib only).

``/api/projects/<pid>/rag/chat`` sends the model::

    system:    <instructions>\\n\\nCONTEXT:\\n<[rank] (source: file, score s)\\ntext blocks>
    ...conversation turns unchanged...

Training rows that teach a small model to use that context (data-prep
"context-grounded rows") must carry the *same* layout, so both sides build it
here and nothing else. Pinned by ``tests/test_rag_chat_prompt.py`` and
``tests/test_grounded_rows.py``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.data.rag_portable.constants import RRF_K

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer using ONLY the context below. "
    "If the answer isn't in the context, say so. Quote the source filename in [brackets] when relevant."
)

CONTEXT_MARKER = "CONTEXT:\n"
_BLOCK_SEP = "\n\n---\n\n"


def format_context_blocks(results: list[dict[str, Any]], max_chars: int = 4000) -> str:
    """Render retrieval hits as numbered, source-cited blocks (stops at ``max_chars``)."""
    blocks: list[str] = []
    total = 0
    for r in results:
        score_str = r.get("ce_score") if "ce_score" in r else r.get("rrf_score", 0)
        label = r.get("filename") or r.get("source") or "source"
        block = f"[{r['rank']}] (source: {label}, score {score_str:.3f})\n{r['text']}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return _BLOCK_SEP.join(blocks)


def build_system_prompt(context: str, system_prompt: str = "") -> str:
    """Instructions + ``CONTEXT:`` block exactly as ``rag_chat`` sends them."""
    return f"{system_prompt or DEFAULT_SYSTEM_PROMPT}\n\n{CONTEXT_MARKER}{context}"


def build_messages(history: list[dict[str, Any]], context: str, system_prompt: str = "") -> list[dict[str, str]]:
    """System turn with the context, then the conversation turns unchanged."""
    return [{"role": "system", "content": build_system_prompt(context, system_prompt)}] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]


def synthetic_hit(rank: int, filename: str, text: str) -> dict[str, Any]:
    """A hit shaped like a live retrieval result, for training rows.

    No retrieval ran, so the score is the RRF value a chunk would get when
    dense and BM25 agree on ``rank`` (``2 / (RRF_K + rank)``) — a layout-faithful
    placeholder, not a measurement.
    """
    return {"rank": rank, "filename": filename, "text": text, "rrf_score": 2.0 / (RRF_K + rank)}
