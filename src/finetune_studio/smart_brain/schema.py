"""Dataclasses describing the manifest schema.

Single responsibility: type definitions + (de)serialization for the corpus metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class IndexDocument:
    """One parsed document handed to the index: its id (a project source id, or a path hash for a plain directory),
    display filename, origin path (citations, labels) and full text. ``chunks`` set = index exactly these rows instead
    of chunking ``text`` (fact cards: one reviewed Q&A pair per row)."""
    document_id: str
    filename: str
    source: str
    text: str
    chunks: tuple[str, ...] = ()


@dataclass
class EmbeddingModelInfo:
    name: str = ""
    dim: int = 1024
    normalize: bool = True
    distance: str = "cosine"
    hf_revision: str = ""
    cached_at: str = ""


@dataclass
class RetrievalSettings:
    embedder: str = ""
    reranker: str = ""
    rerank_enabled: bool = True
    rerank_top_n: int = 50    # retrieve this many candidates before rerank
    hybrid_enabled: bool = True
    rrf_k: int = 60
    extra: dict = field(default_factory=dict)


@dataclass
class ChunkSettings:
    size: int = 400
    overlap: int = 80
    splitter: str = "word"   # "word" = indexes built before row-preserving chunking; "rows+tokens" / "rows+chars" = size counted
                             # in embedder tokens / by the chars-per-token estimate (data.smart_brain.chunking)


@dataclass
class Manifest:
    name: str
    version: str = "2"
    created_at: float = 0.0
    updated_at: float = 0.0
    embedding_model: EmbeddingModelInfo = field(default_factory=EmbeddingModelInfo)
    chunk_settings: ChunkSettings = field(default_factory=ChunkSettings)
    rag_settings: RetrievalSettings = field(default_factory=RetrievalSettings)
    documents: int = 0
    chunks: int = 0
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> Manifest:
        return cls(
            name=d["name"],
            version=d.get("version", "2"),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            embedding_model=EmbeddingModelInfo(**d.get("embedding_model", {})),
            chunk_settings=ChunkSettings(**d.get("chunk_settings", {})),
            rag_settings=RetrievalSettings(**d.get("rag_settings", {})),
            documents=d.get("documents", 0),
            chunks=d.get("chunks", 0),
            extra=d.get("extra", {}),
        )
