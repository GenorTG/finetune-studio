"""High-level RAG operations.

WHAT THIS FILE DOES
==================
Provides friendly methods for the most common RAG operations:
  - ingest_file(path)
  - ingest_directory(path)
  - query(text)
  - list_documents()
  - remove_document(id)

KEY CONCEPTS
============
- Facade pattern: hides the complexity of store.py, ingest.py, and
  query.py behind a simple interface.
- Convenience over flexibility: the Tradeoff here is that these
  methods make common operations easy, but for complex workflows
  you might need to use the underlying modules directly.
"""

"""RAG manager — high-level operations for document management."""
from finetune_studio.rag.ingest import ingest_directory, ingest_file
from finetune_studio.rag.store import VectorStore


class RAGManager:
    """High-level RAG document management."""

    def __init__(self, store_path: str = "data/rag_store"):
        self.store = VectorStore(store_path)

    def ingest_file(self, file_path: str, chunk_size: int = 512, overlap: int = 50) -> dict:
        """Ingest a single file into the RAG store."""
        doc = ingest_file(file_path, chunk_size, overlap)
        added = self.store.add_chunks(doc.chunks)
        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "chunks_added": added,
            "total_chunks_in_store": self.store.count(),
        }

    def ingest_directory(self, directory: str, chunk_size: int = 512,
                         overlap: int = 50, extensions: list | None = None) -> dict:
        """Ingest all supported files in a directory."""
        documents = ingest_directory(directory, chunk_size, overlap, extensions)
        total_chunks = 0
        total_docs = 0

        for doc in documents:
            added = self.store.add_chunks(doc.chunks)
            total_chunks += added
            total_docs += 1

        return {
            "documents_ingested": total_docs,
            "chunks_added": total_chunks,
            "total_chunks_in_store": self.store.count(),
        }

    def remove_document(self, document_id: str) -> dict:
        """Remove a document from the RAG store."""
        removed = self.store.remove_document(document_id)
        return {
            "document_id": document_id,
            "chunks_removed": removed,
            "total_chunks_in_store": self.store.count(),
        }

    def list_documents(self) -> list:
        """List all documents in the RAG store."""
        return self.store.list_documents()

    def stats(self) -> dict:
        """Get RAG store statistics."""
        docs = self.list_documents()
        return {
            "total_chunks": self.store.count(),
            "total_documents": len(docs),
            "documents": docs,
        }

    def clear(self):
        """Clear all data from the RAG store."""
        self.store.clear()
