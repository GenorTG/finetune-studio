"""Dataclasses describing the manifest schema.

Single responsibility: type definitions + (de)serialization for the corpus metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EmbeddingModelInfo:
    name: str = ""
    dim: int = 1024
    normalize: bool = True
    distance: str = "cosine"
    hf_revision: str = ""
    cached_at: str = ""


@dataclass
class RagSettings:
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
    splitter: str = "word"


@dataclass
class Manifest:
    name: str
    version: str = "2"
    created_at: float = 0.0
    updated_at: float = 0.0
    embedding_model: EmbeddingModelInfo = field(default_factory=EmbeddingModelInfo)
    chunk_settings: ChunkSettings = field(default_factory=ChunkSettings)
    rag_settings: RagSettings = field(default_factory=RagSettings)
    documents: int = 0
    chunks: int = 0
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Manifest":
        return cls(
            name=d["name"],
            version=d.get("version", "2"),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            embedding_model=EmbeddingModelInfo(**d.get("embedding_model", {})),
            chunk_settings=ChunkSettings(**d.get("chunk_settings", {})),
            rag_settings=RagSettings(**d.get("rag_settings", {})),
            documents=d.get("documents", 0),
            chunks=d.get("chunks", 0),
            extra=d.get("extra", {}),
        )
