"""Result diversity: one document may not fill the top results (``smart_brain.diversity``), in the studio search and
the standalone package server (which carries its own copy of ``diversify``)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finetune_studio.smart_brain import standalone_server as srv
from finetune_studio.smart_brain.bm25 import BM25Index
from finetune_studio.smart_brain.diversity import DEFAULT_MAX_PER_DOCUMENT, diversify
from finetune_studio.smart_brain.query import BrainQuery
from finetune_studio.smart_brain.schema import (
    EmbeddingModelInfo,
    Manifest,
    RetrievalSettings,
)


def hits(*docs: str) -> list[dict]:
    return [{"chunk_id": f"c{i}", "document_id": d, "score": 1.0 - i / 100} for i, d in enumerate(docs)]


def test_a_document_gets_at_most_the_cap_and_others_move_up() -> None:
    out = diversify(hits("a", "a", "a", "a", "a", "b", "c"), 5, 2)
    assert [h["chunk_id"] for h in out] == ["c0", "c1", "c5", "c6", "c2"]   # overflow fills the end, in order
    assert [h["score"] for h in out] == [1.0, 0.99, 0.95, 0.94, 0.98]        # scores are kept as they were


def test_one_document_corpus_still_gets_top_k_and_zero_is_no_cap() -> None:
    assert len(diversify(hits(*"aaaaaaa"), 5, 2)) == 5
    assert [h["chunk_id"] for h in diversify(hits("a", "a", "a", "b"), 3, 0)] == ["c0", "c1", "c2"]
    assert diversify([], 10, 3) == []


def test_the_package_server_copy_behaves_the_same() -> None:
    rnd = random.Random(7)
    assert srv.DEFAULT_MAX_PER_DOCUMENT == DEFAULT_MAX_PER_DOCUMENT
    for _ in range(300):
        docs = [rnd.choice("abcde") for _ in range(rnd.randint(0, 40))]
        k, cap = rnd.randint(1, 15), rnd.randint(0, 4)
        assert srv.diversify(hits(*docs), k, cap) == diversify(hits(*docs), k, cap)


def _query(tmp_path: Path, cap: int) -> BrainQuery:
    # 8 passages of one long e-mail thread best by meaning, then one passage each of 3 other documents
    docs = ["thread"] * 8 + ["lease", "board", "memo"]
    ids = [f"p{i}" for i in range(len(docs))]
    vectors = np.array([[1.0 - i / 50, (i / 50)] for i in range(len(docs))], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    manifest = Manifest(name="t", version="2", created_at=0.0, updated_at=0.0,
                        embedding_model=EmbeddingModelInfo(name="fake", dim=2),
                        rag_settings=RetrievalSettings(embedder="fake", reranker="", rerank_enabled=False,
                                                       hybrid_enabled=False, max_per_document=cap))
    chunks = pd.DataFrame({"id": ids, "text": [f"{d} passage {i}" for i, d in enumerate(docs)],
                           "filename": [f"{d}.pdf" for d in docs], "source": [f"{d}.pdf" for d in docs],
                           "document_id": docs, "chunk_index": list(range(len(docs)))})
    return BrainQuery(tmp_path, manifest, chunks, vectors, {c: i for i, c in enumerate(ids)},
                      BM25Index.build(chunks["text"].tolist()), lambda s: np.array([1.0, 0.0], dtype=np.float32))


def test_studio_search_applies_the_index_setting_and_an_override(tmp_path: Path) -> None:
    capped = _query(tmp_path, 3).search("anything", top_k=5)
    assert [h["document_id"] for h in capped] == ["thread", "thread", "thread", "lease", "board"]
    assert [h["rank"] for h in capped] == [1, 2, 3, 4, 5] and all("dense_score" in h for h in capped)
    plain = _query(tmp_path, 3).search("anything", top_k=5, max_per_document=0)
    assert {h["document_id"] for h in plain} == {"thread"}


def test_old_manifests_default_to_three() -> None:
    m = Manifest.from_json({"name": "x", "rag_settings": {"rerank_top_n": 50}})
    assert m.rag_settings.max_per_document == 3


@pytest.mark.parametrize(("raw", "want"), [("-1", -1), ("0", 0), ("2", 2)])
def test_server_config_takes_the_key_from_env(raw: str, want: int) -> None:
    cfg, origin = srv.resolve_config({}, {"RAG_MAX_PER_DOCUMENT": raw}, {})
    assert cfg["max_per_document"] == want and origin["max_per_document"] == "env"


def test_server_config_refuses_a_negative_cap() -> None:
    with pytest.raises(srv.ConfigError, match="max_per_document"):
        srv.resolve_config({}, {"RAG_MAX_PER_DOCUMENT": "-5"}, {})
