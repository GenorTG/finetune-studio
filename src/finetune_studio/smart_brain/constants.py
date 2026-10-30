"""Module-wide constants for the portable RAG store.

SCHEMA_VERSION is bumped when manifest.json or the on-disk file layout changes
in a way that's not backwards-compatible — readers can refuse old corpora.
"""
from __future__ import annotations

SCHEMA_VERSION = "2"

# Default models
DEFAULT_EMBEDDER = "intfloat/multilingual-e5-large"
# Reranker for a new index on a machine with a GPU. Quillmoor S (tests/corpus/quillmoor/RESULTS.md §5), same hybrid
# candidates, one client on an RTX 3090: hit@5 89.2 % vs 78.6 % for MiniLM (reverse look-ups 40/40 vs 24/40), at
# 190 ms p50 in fp16. On a CPU it is ~2 GB of fp32 weights scoring 50 passages per search, so CPU-only machines get
# CPU_RERANKER (rerankers.default_reranker picks).
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
CPU_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDER_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard RRF constant

# Passages a search returns when the caller does not say. Measured on the Korvane quiz (tests/corpus/korvane/RESULTS.md §8):
# readers answer 84 -> 87 -> 89 of 102 at top-5/10/20, and 10 chunks still fit comfortably in a modern context window.
DEFAULT_TOP_K = 10

# Magic prefixes that mean "the model lives at the local path next to the corpus"
EMBEDDER_LOCAL_PREFIX = "embedder_local:"  # -> <corpus_dir>/embedder
RERANKER_LOCAL_PREFIX = "reranker_local:"  # -> <corpus_dir>/reranker
