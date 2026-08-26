"""RAG subpackage — retrieval-augmented generation."""

from finetune_studio.rag.store import VectorStore, SearchResult
from finetune_studio.rag.ingest import Document, Chunk, chunk_text, ingest_file, ingest_directory
from finetune_studio.rag.query import RAGConfig, RAGQuery
from finetune_studio.rag.manager import RAGManager

__all__ = [
    "VectorStore", "SearchResult",
    "Document", "Chunk", "chunk_text", "ingest_file", "ingest_directory",
    "RAGConfig", "RAGQuery",
    "RAGManager",
]
