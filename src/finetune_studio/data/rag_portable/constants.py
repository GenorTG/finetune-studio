"""Module-wide constants for the portable RAG store.

SCHEMA_VERSION is bumped when manifest.json or the on-disk file layout changes
in a way that's not backwards-compatible — readers can refuse old corpora.
"""
from __future__ import annotations


SCHEMA_VERSION = "2"

# Default models
DEFAULT_EMBEDDER = "intfloat/multilingual-e5-large"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDER_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard RRF constant

# Magic prefixes that mean "the model lives at the local path next to the corpus"
EMBEDDER_LOCAL_PREFIX = "embedder_local:"  # -> <corpus_dir>/embedder
RERANKER_LOCAL_PREFIX = "reranker_local:"  # -> <corpus_dir>/reranker
