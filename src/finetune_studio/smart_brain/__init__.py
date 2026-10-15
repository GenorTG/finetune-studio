"""Portable RAG store — file-based vector + BM25 index + optional reranker.

Layout (one directory per corpus):

  corpus_dir/
    manifest.json            - corpus metadata, embedding model, settings
    chunks.parquet           - chunk text + metadata (pandas/pyarrow readable)
    vectors.npy              - (N, D) float32, L2-normalized
    vectors.idx.json         - chunk_id -> row index in vectors.npy
    bm25.json                - {terms, doc_lens, df, avgdl, doc_count}
    sources/<doc_id>.txt    - original parsed text per source
    tests/                   - regression tracking

Retrieval pipeline:

  query -> embed (e5-large) -> dense top-N
        \\-> tokenize      -> BM25 top-N
        \\-> RRF(fuse)           -> top-K candidates (~50)
        \\-> (optional) rerank with cross-encoder -> final top-k

The whole thing is portable: no service, no network, no vendor. Just a dir.

LAYOUT
------
smart_brain/
  __init__.py    — public API re-exports
  constants.py   — schema version, default model names, magic prefixes
  schema.py      — EmbeddingModelInfo, RetrievalSettings, ChunkSettings, Manifest dataclasses
  io.py          — write_json / read_json / pandas import helper
  tokenize.py    — tokenize() + _TOKEN_RE
  bm25.py        — BM25Index (build / score / to_dict / from_dict)
  embedders.py   — get_embedder()
  rerankers.py   — get_reranker()
  rrf.py         — rrf_fuse()
  store.py       — BrainIndex (build, load, rebuild, bundle, export, unshare)
  query.py       — BrainQuery (search, format_context)
  prompt.py      — the rag/chat prompt layout (shared with context-grounded training rows)
"""

from finetune_studio.smart_brain.bm25 import BM25Index
from finetune_studio.smart_brain.constants import (
    DEFAULT_EMBEDDER,
    DEFAULT_RERANKER,
    EMBEDDER_FALLBACK,
    EMBEDDER_LOCAL_PREFIX,
    RERANKER_LOCAL_PREFIX,
    RRF_K,
    SCHEMA_VERSION,
)
from finetune_studio.smart_brain.embedders import get_embedder
from finetune_studio.smart_brain.io import (
    read_json,
    try_import_pandas,
    write_json,
)
from finetune_studio.smart_brain.query import BrainQuery
from finetune_studio.smart_brain.rerankers import get_reranker
from finetune_studio.smart_brain.rrf import rrf_fuse
from finetune_studio.smart_brain.schema import (
    ChunkSettings,
    EmbeddingModelInfo,
    Manifest,
    RetrievalSettings,
)
from finetune_studio.smart_brain.store import BrainIndex
from finetune_studio.smart_brain.tokenize import tokenize

__all__ = [
    "DEFAULT_EMBEDDER",
    "DEFAULT_RERANKER",
    "EMBEDDER_FALLBACK",
    "EMBEDDER_LOCAL_PREFIX",
    "RERANKER_LOCAL_PREFIX",
    "RRF_K",
    "SCHEMA_VERSION",
    "BM25Index",
    "BrainIndex",
    "BrainQuery",
    "ChunkSettings",
    "EmbeddingModelInfo",
    "Manifest",
    "RetrievalSettings",
    "get_embedder",
    "get_reranker",
    "read_json",
    "rrf_fuse",
    "tokenize",
    "try_import_pandas",
    "write_json",
]
