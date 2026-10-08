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

# Retrieved context is sized from the loaded model's window, not a fixed few thousand characters (the old
# 4000/5000-char caps fed a 32k-token model ~1.3k tokens of context: ~2 of the 5 retrieved chunks).
CHARS_PER_TOKEN = 3            # conservative: ids and numbers tokenise at ~3.6 chars/token on the Korvane corpus
PROMPT_OVERHEAD_TOKENS = 1024  # system prompt, source labels, the question
DEFAULT_CONTEXT_CHARS = 24000  # when the window is unknown (API models): 5 chunks of up to ~4.5k chars

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


def context_char_budget(n_ctx: int | None, *, max_new_tokens: int = 512, history_chars: int = 0) -> int:
    """Characters of retrieved context that fit a model window of ``n_ctx`` tokens next to the answer and the chat history.

    ``n_ctx`` unknown (API provider, nothing loaded) -> ``DEFAULT_CONTEXT_CHARS``.
    """
    if not isinstance(n_ctx, int) or n_ctx <= 0:   # engines report None / a non-number when the window is unknown
        return DEFAULT_CONTEXT_CHARS
    tokens = n_ctx - max_new_tokens - PROMPT_OVERHEAD_TOKENS - history_chars // CHARS_PER_TOKEN
    return max(4000, tokens * CHARS_PER_TOKEN)


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
