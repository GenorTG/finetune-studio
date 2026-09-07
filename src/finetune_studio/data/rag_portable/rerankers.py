"""Rerank-model adapter (cross-encoder from sentence-transformers).

Special name prefixes:
  - "reranker_local:<path>"  -> load from local CrossEncoder-saved dir
  - any other name           -> fetch from HuggingFace
"""
from __future__ import annotations

from finetune_studio.data.rag_portable.constants import DEFAULT_RERANKER, RERANKER_LOCAL_PREFIX


def get_reranker(name: str = DEFAULT_RERANKER, device: str = "cpu"):
    """Return (rerank, name) where rerank(query, docs) -> list[float]."""
    from sentence_transformers import CrossEncoder

    local_path = None
    if name.startswith(RERANKER_LOCAL_PREFIX):
        local_path = name[len(RERANKER_LOCAL_PREFIX):]
    load_target = local_path if local_path else name

    model = CrossEncoder(load_target, device=device, max_length=512)

    def rerank(query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        pairs = [[query, d] for d in docs]
        scores = model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
    return rerank, name
