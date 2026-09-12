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

    # Strip known library prefixes that were incorrectly saved to manifests
    clean_name = name
    for library_prefix in ("sentence-transformers/shared:", "cross-encoder/shared:"):
        if clean_name.startswith(library_prefix):
            clean_name = clean_name[len(library_prefix):]
            break
    # Now resolve shared:reranker: references
    from finetune_studio.data.rag_portable.shared_refs import resolve_model_ref
    resolved = resolve_model_ref(clean_name, "reranker")

    local_path = None
    if resolved.startswith(RERANKER_LOCAL_PREFIX):
        local_path = resolved[len(RERANKER_LOCAL_PREFIX):]
    load_target = local_path if local_path else resolved

    model = CrossEncoder(load_target, device=device, max_length=512)

    def rerank(query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        pairs = [[query, d] for d in docs]
        scores = model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
    return rerank, name
