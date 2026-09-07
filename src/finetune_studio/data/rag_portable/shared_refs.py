"""Resolve `shared:embedder:<short_id>` and `shared:reranker:<short_id>` refs
to actual local paths in the shared model store.

Why this exists: PortableRAG.load() reads the manifest, which may store
embedder/reranker names as `shared:embedder:<short_id>` (a reference, not a
loadable HF repo id). The load path used to pass this directly to
sentence-transformers, which then prefixed it with "sentence-transformers/"
treating it as a HuggingFace repo id -- failing with a malformed-id error.

This module translates `shared:<kind>:<short_id>` to the actual local path
under `~/.finetune-studio/shared/{kind}s/<short_id>/` so get_embedder() sees
a `embedder_local:<path>` prefix it already understands.
"""
from __future__ import annotations

from pathlib import Path

from finetune_studio.data import shared_models as _sm
from finetune_studio.data.rag_portable.constants import (
    EMBEDDER_LOCAL_PREFIX, RERANKER_LOCAL_PREFIX,
)


def resolve_model_ref(name: str, kind: str) -> str:
    """Resolve a model name to one get_embedder/get_reranker understands.

    - "shared:embedder:<short_id>" -> "embedder_local:<abs_path>"
    - "shared:reranker:<short_id>" -> "reranker_local:<abs_path>"
    - "embedder_local:<path>" / "reranker_local:<path>" -> unchanged
    - anything else -> unchanged (treated as an HF repo id by the caller)
    """
    if not name:
        return name
    prefix = f"shared:{kind}:"
    if name.startswith(prefix):
        short_id = name[len(prefix):]
        try:
            local_path: Path = _sm.resolve(short_id, kind)
        except Exception:
            # If resolve fails (e.g. the model wasn't actually registered in
            # the shared store), leave the name unchanged -- the downstream
            # load will produce a clearer error than we can fabricate here.
            return name
        magic = EMBEDDER_LOCAL_PREFIX if kind == "embedder" else RERANKER_LOCAL_PREFIX
        return f"{magic}{str(local_path.resolve())}"
    # Already a usable form.
    return name
