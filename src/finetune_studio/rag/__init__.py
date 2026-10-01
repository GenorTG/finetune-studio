"""RAG subpackage — retrieval-augmented generation."""

from finetune_studio.rag.ingest import (
    Chunk,
    Document,
    chunk_text,
    ingest_directory,
    ingest_file,
)
from finetune_studio.rag.manager import RAGManager
from finetune_studio.rag.query import RAGConfig, RAGQuery
from finetune_studio.rag.store import SearchResult, VectorStore

__all__ = [
    "Chunk",
    "Document",
    "RAGConfig",
    "RAGManager",
    "RAGQuery",
    "SearchResult",
    "VectorStore",
    "chunk_text",
    "ingest_directory",
    "ingest_file",
]
