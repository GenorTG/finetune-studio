"""Back-compat shim — the real code lives in finetune_studio.data.rag_portable/.

Older code that did `from finetune_studio.data.rag_portable import PortableRAG`
keeps working unchanged.
"""

from finetune_studio.data.rag_portable import (  # noqa: F401
    BM25Index,
    ChunkSettings,
    DEFAULT_EMBEDDER,
    DEFAULT_RERANKER,
    EMBEDDER_FALLBACK,
    EMBEDDER_LOCAL_PREFIX,
    EmbeddingModelInfo,
    Manifest,
    PortableRAG,
    PortableRAGQuery,
    RERANKER_LOCAL_PREFIX,
    RRF_K,
    RagSettings,
    SCHEMA_VERSION,
    get_embedder,
    get_reranker,
    rrf_fuse,
    tokenize,
)
