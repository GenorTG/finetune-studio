"""Regression tests for the A4 RAG-lane audit (2026-10-01).

Pins three real bugs found while auditing finetune_studio/rag/ (legacy
Chroma-backed RAG) and finetune_studio/data/rag_portable/ (the live
PortableRAG system):

1. PortableRAG.remove_source() deleted the raw sources/*.txt file but left
   the matching chunks fully indexed in chunks.parquet/vectors.npy/bm25.json
   and listed in manifest documents_meta — a "removed" source stayed fully
   searchable and still appeared in list_sources().
2. finetune_studio.rag.store.VectorStore._get_embedder() silently ignored
   the embedding_model argument and always loaded all-MiniLM-L6-v2.
3. PortableRAGQuery.search() recomputed self.bm25.score(query) once per
   candidate result instead of reusing the single bm25_scores array already
   computed for ranking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finetune_studio.data.rag_portable.schema import EmbeddingModelInfo
from finetune_studio.data.rag_portable.store import PortableRAG


def _fake_get_embedder(name: str = "fake-deterministic", device: str = "cpu"):
    dim = 32

    def encode(texts):
        items = [texts] if isinstance(texts, str) else list(texts)
        out = np.zeros((len(items), dim), dtype=np.float32)
        for i, text in enumerate(items):
            out[i, hash(text) % dim] = 1.0
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out[0] if isinstance(texts, str) else out

    info = EmbeddingModelInfo(
        name=name or "fake-deterministic",
        dim=dim,
        normalize=True,
        distance="cosine",
        cached_at="1970-01-01T00:00:00Z",
    )
    return encode, info


@pytest.fixture()
def patched_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "finetune_studio.data.rag_portable.store.get_embedder",
        _fake_get_embedder,
    )


def test_remove_source_purges_chunks_vectors_bm25_and_manifest(
    tmp_path: Path, patched_embedder: None,
) -> None:
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "alpha.txt").write_text(
        "Alpha document about zebras and giraffes.", encoding="utf-8",
    )
    (src_dir / "beta.txt").write_text(
        "Beta document about rockets and planets.", encoding="utf-8",
    )

    corpus_dir = tmp_path / "corpus"
    rag = PortableRAG(corpus_dir)
    result = rag.build_from_directory(src_dir, name="audit-test")
    assert result["documents"] == 2
    assert result["chunks"] >= 2

    sources_before = rag.list_sources()
    assert len(sources_before) == 2
    alpha_id = next(
        s["id"] for s in sources_before if "alpha" in s["filename"].lower()
    )

    removed = rag.remove_source(alpha_id)
    assert removed is True

    # The source is gone from the manifest-backed listing (previously it
    # stayed listed because remove_source never touched documents_meta).
    sources_after = rag.list_sources()
    assert all(s["id"] != alpha_id for s in sources_after)
    assert len(sources_after) == 1

    # The raw text file is gone.
    assert not (corpus_dir / "sources" / f"{alpha_id}.txt").exists()

    # Chunks/vectors/bm25 no longer reference the removed document — this is
    # the core of the bug: previously these stayed fully indexed and the
    # "removed" source was still returned by search().
    import pandas as pd

    chunks_df = pd.read_parquet(corpus_dir / "chunks.parquet")
    assert alpha_id not in set(chunks_df["document_id"])
    vectors = np.load(corpus_dir / "vectors.npy")
    assert vectors.shape[0] == len(chunks_df)

    loaded = rag.load()
    hits = loaded.search("zebras giraffes", top_k=5)
    assert all(h["document_id"] != alpha_id for h in hits)

    manifest = loaded.manifest
    assert manifest.documents == 1
    assert manifest.chunks == len(chunks_df)


def test_remove_source_unknown_id_is_noop(
    tmp_path: Path, patched_embedder: None,
) -> None:
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "alpha.txt").write_text("Alpha document.", encoding="utf-8")

    corpus_dir = tmp_path / "corpus"
    rag = PortableRAG(corpus_dir)
    rag.build_from_directory(src_dir, name="audit-test")

    removed = rag.remove_source("does-not-exist")
    assert removed is False
    assert len(rag.list_sources()) == 1


def test_vector_store_get_embedder_respects_embedding_model(monkeypatch):
    """finetune_studio.rag.store.VectorStore previously always loaded
    all-MiniLM-L6-v2 regardless of the embedding_model argument."""
    from finetune_studio.rag.store import VectorStore

    loaded_names = []

    class _FakeST:
        def __init__(self, name):
            loaded_names.append(name)
            self.name = name

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", _FakeST,
    )

    store = VectorStore("unused/path")
    emb_a = store._get_embedder("model-a")
    emb_b = store._get_embedder("model-b")
    emb_a_again = store._get_embedder("model-a")

    assert loaded_names == ["model-a", "model-b"]
    assert emb_a.name == "model-a"
    assert emb_b.name == "model-b"
    assert emb_a_again is emb_a  # cached per model name


def test_search_reuses_bm25_scores_instead_of_recomputing(
    tmp_path: Path, patched_embedder: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search() must call BM25Index.score() exactly once per query, not once
    per candidate result."""
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    for i in range(5):
        (src_dir / f"doc{i}.txt").write_text(
            f"Document number {i} about topic {i}.", encoding="utf-8",
        )

    corpus_dir = tmp_path / "corpus"
    rag = PortableRAG(corpus_dir)
    rag.build_from_directory(src_dir, name="audit-test")
    loaded = rag.load()

    from finetune_studio.data.rag_portable.bm25 import BM25Index

    call_count = {"n": 0}
    real_score = BM25Index.score

    def _counting_score(self, query):
        call_count["n"] += 1
        return real_score(self, query)

    monkeypatch.setattr(BM25Index, "score", _counting_score)

    loaded.search("topic", top_k=3, hybrid=True, rerank=False)
    assert call_count["n"] == 1
