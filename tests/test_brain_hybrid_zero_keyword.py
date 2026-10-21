"""Hybrid search: a passage that shares no word with the question gets no keyword rank.

Regression (2026-10-21, website corpus): zero-score passages were ranked among themselves in index order, so with
no keyword evidence the earlier passage in the corpus got a keyword vote over the passage that was best by meaning
("Gdzie mieszkasz?" ranked the city passage #15)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from finetune_studio.smart_brain.bm25 import BM25Index
from finetune_studio.smart_brain.query import BrainQuery
from finetune_studio.smart_brain.schema import EmbeddingModelInfo, Manifest, RetrievalSettings


def test_no_keyword_evidence_leaves_the_order_to_meaning(tmp_path: Path) -> None:
    texts = [f"Passage number {i} about something else." for i in range(10)]
    ids = [f"c{i}" for i in range(10)]
    vectors = np.zeros((10, 4), dtype=np.float32)
    vectors[:, 1] = 1.0
    vectors[9] = [1, 0, 0, 0]          # last in the corpus, best by meaning
    vectors[0] = [0.9, 0.44, 0, 0]     # first in the corpus, second by meaning
    manifest = Manifest(name="t", version="2", created_at=0.0, updated_at=0.0,
                        embedding_model=EmbeddingModelInfo(name="fake", dim=4),
                        rag_settings=RetrievalSettings(embedder="fake", reranker="", rerank_enabled=False))
    chunks = pd.DataFrame({"id": ids, "text": texts, "filename": "x.md", "source": "x.md",
                           "document_id": ids, "chunk_index": 0})
    q = BrainQuery(tmp_path, manifest, chunks, vectors, {c: i for i, c in enumerate(ids)}, BM25Index.build(texts),
                   lambda s: np.array([1, 0, 0, 0], dtype=np.float32))
    hits = q.search("Gdzie mieszkasz?", top_k=3)
    assert [h["chunk_id"] for h in hits[:2]] == ["c9", "c0"]
