"""Embedding-model adapter (sentence-transformers).

Special name prefixes:
  - "embedder_local:<path>"  -> load from local SentenceTransformer-saved dir
  - any other name          -> fetch from HuggingFace
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from finetune_studio.data.rag_portable.constants import (
    DEFAULT_EMBEDDER,
    EMBEDDER_LOCAL_PREFIX,
)
from finetune_studio.data.rag_portable.schema import EmbeddingModelInfo


def _load_sentence_transformer(
    load_target: str, *, device: str, cache_dir: Path
) -> object:
    """Load a SentenceTransformer from HF id or local path.

    Local dirs are prepared first so incomplete shared-store copies (missing
    ``1_Pooling/config.json``) load under current sentence-transformers without
    switching to a different embedder.
    """
    from sentence_transformers import SentenceTransformer

    from finetune_studio.data.sentence_transformer_local import (
        prepare_local_sentence_transformer_dir,
    )

    path = Path(load_target)
    if path.is_dir():
        prepare_local_sentence_transformer_dir(path)
        return SentenceTransformer(str(path), device=device)
    return SentenceTransformer(
        load_target, device=device, cache_folder=str(cache_dir)
    )


def get_embedder(
    name: str = DEFAULT_EMBEDDER, device: str = "cpu"
) -> tuple[object, EmbeddingModelInfo]:
    """Return (encode, info) where encode(text|list[str]) -> ndarray(float32)."""
    # Use canonical cache under user's home (NOT /tmp) — see
    # finetune_studio.data.shared_models.hf_cache_dir for rationale.
    from finetune_studio.data.shared_models import hf_cache_dir
    cache_dir = hf_cache_dir()
    os.environ.setdefault("HF_HOME", str(cache_dir))

    local_path = None
    if name.startswith(EMBEDDER_LOCAL_PREFIX):
        local_path = name[len(EMBEDDER_LOCAL_PREFIX):]
    load_target = local_path if local_path else name

    model = _load_sentence_transformer(
        load_target, device=device, cache_dir=cache_dir
    )
    dim = model.get_embedding_dimension()

    def encode(texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            v = model.encode([texts], normalize_embeddings=True)
            return np.asarray(v, dtype=np.float32)[0]
        v = model.encode(
            list(texts),
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
        return np.asarray(v, dtype=np.float32)

    info = EmbeddingModelInfo(
        name=name, dim=dim, normalize=True, distance="cosine",
        cached_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return encode, info
