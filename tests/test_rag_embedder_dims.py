"""Embedder provenance: never silently fall back across dimensions."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from finetune_studio.data.rag_portable.schema import EmbeddingModelInfo
from finetune_studio.data.rag_portable.store import PortableRAG


def _write_corpus(root: Path, *, dim: int = 384, embedder: str = "mini-fake") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sources").mkdir(exist_ok=True)
    (root / "sources" / "doc1.txt").write_text("alpha beta gamma", encoding="utf-8")
    chunks = pd.DataFrame(
        [
            {
                "id": "doc1_0",
                "document_id": "doc1",
                "chunk_index": 0,
                "source": "doc1.txt",
                "filename": "doc1.txt",
                "text": "alpha beta gamma",
            }
        ]
    )
    chunks.to_parquet(root / "chunks.parquet", index=False)
    vectors = np.zeros((1, dim), dtype=np.float32)
    np.save(root / "vectors.npy", vectors)
    (root / "vectors.idx.json").write_text(json.dumps({"doc1_0": 0}), encoding="utf-8")
    (root / "bm25.json").write_text(
        json.dumps({
            "terms": {},
            "doc_lens": [1],
            "df": {},
            "avgdl": 1.0,
            "doc_count": 1,
            "k1": 1.5,
            "b": 0.75,
        }),
        encoding="utf-8",
    )
    manifest = {
        "name": "dim-test",
        "version": "2",
        "created_at": 1.0,
        "updated_at": 1.0,
        "embedding_model": {
            "name": embedder,
            "dim": dim,
            "normalize": True,
            "distance": "cosine",
        },
        "chunk_settings": {"size": 400, "overlap": 80, "splitter": "word"},
        "rag_settings": {
            "embedder": embedder,
            "reranker": "",
            "rerank_enabled": False,
            "hybrid_enabled": False,
            "rrf_k": 60,
            "rerank_top_n": 50,
        },
        "documents": 1,
        "chunks": 1,
        "extra": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_load_rejects_dimension_mismatch(tmp_path: Path) -> None:
    """Loaded embedder dim must match vectors.npy — no silent wrong-dim fallback."""
    root = _write_corpus(tmp_path / "c", dim=384, embedder="fake/mini")

    def _wrong_dim_embedder(name: str = "", device: str = "cpu"):
        info = EmbeddingModelInfo(name=name or "fake/large", dim=1024)

        def encode(texts):
            if isinstance(texts, str):
                return np.zeros(1024, dtype=np.float32)
            return np.zeros((len(list(texts)), 1024), dtype=np.float32)

        return encode, info

    with (
        patch(
            "finetune_studio.data.rag_portable.store.get_embedder",
            side_effect=_wrong_dim_embedder,
        ),
        pytest.raises(ValueError, match="dimension mismatch"),
    ):
        PortableRAG(root).load()


def test_load_raises_actionable_error_when_embedder_fails(tmp_path: Path) -> None:
    """Missing/broken embedder must not fall back to DEFAULT_EMBEDDER."""
    root = _write_corpus(tmp_path / "c2", dim=384, embedder="missing/model")

    with (
        patch(
            "finetune_studio.data.rag_portable.store.get_embedder",
            side_effect=RuntimeError("HF offline"),
        ),
        pytest.raises(RuntimeError, match="Failed to load corpus embedder"),
    ):
        PortableRAG(root).load()


def test_search_rejects_query_dim_mismatch(tmp_path: Path) -> None:
    """Even if load is forced, search must refuse matmul across dims."""
    root = _write_corpus(tmp_path / "c3", dim=384, embedder="ok/model")

    def _ok_load_embedder(name: str = "", device: str = "cpu"):
        info = EmbeddingModelInfo(name=name or "ok/model", dim=384)

        def encode(texts):
            # Deliberately wrong query size to simulate polluted encode.
            if isinstance(texts, str):
                return np.zeros(1024, dtype=np.float32)
            return np.zeros((len(list(texts)), 1024), dtype=np.float32)

        return encode, info

    with patch(
        "finetune_studio.data.rag_portable.store.get_embedder",
        side_effect=_ok_load_embedder,
    ):
        # load checks embed_info.dim vs vectors — info says 384 so load succeeds
        q = PortableRAG(root).load()
    with pytest.raises(ValueError, match="Query embedding dimension"):
        q.search("hello", top_k=1)
