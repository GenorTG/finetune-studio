"""Vector store for document chunks.

WHAT THIS FILE DOES
==================
Stores text chunks as vector embeddings in ChromaDB. Supports:
  - Adding chunks (with metadata like source file, page number)
  - Searching by similarity to a query
  - Listing all documents
  - Removing documents

KEY CONCEPTS
============
- Vector embedding: a numerical representation of text (e.g., 384
  or 768 floats). Similar texts have similar embeddings.
- Cosine similarity: the standard way to measure similarity between
  vectors. 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite.
- ChromaDB: an open-source vector database. Stores embeddings on disk.
- Metadata filtering: when searching, you can filter by metadata
  (e.g., "only search PDFs").
"""

"""Vector store — ChromaDB-based persistent vector storage."""
from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: str = ""
    document_id: str = ""
    text: str = ""
    score: float = 0.0
    metadata: dict = None
    source: str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VectorStore:
    """Persistent vector store using ChromaDB."""

    def __init__(self, store_path: str = "data/rag_store"):
        self.store_path = store_path
        self._client = None
        self._collection = None
        self._embedder = None

    def _get_client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.store_path)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_embedder(self, embedding_model: str | None = None):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def add_chunks(self, chunks: list, batch_size: int = 100, embedding_model: str | None = None) -> int:
        """Add chunks to the vector store. Returns count added."""
        collection = self._get_collection()
        embedder = self._get_embedder(embedding_model)

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c.id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [
                {"document_id": c.document_id, "chunk_index": c.chunk_index, **(c.metadata or {})}
                for c in batch
            ]

            embeddings = embedder.encode(texts).tolist()

            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total += len(batch)

        return total

    def search(self, query: str, top_k: int = 5, filter_doc: str | None = None, embedding_model: str | None = None) -> list[SearchResult]:
        """Search for similar chunks."""
        collection = self._get_collection()
        embedder = self._get_embedder(embedding_model)

        query_embedding = embedder.encode([query]).tolist()

        where = {"document_id": filter_doc} if filter_doc else None

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    chunk_id=doc_id,
                    document_id=results["metadatas"][0][i].get("document_id", ""),
                    text=results["documents"][0][i],
                    score=1 - results["distances"][0][i],  # Convert distance to similarity
                    metadata=results["metadatas"][0][i],
                ))

        return search_results

    def remove_document(self, document_id: str) -> int:
        """Remove all chunks from a document."""
        collection = self._get_collection()

        # Find all chunks for this document
        results = collection.get(
            where={"document_id": document_id},
            include=[],
        )

        if results["ids"]:
            collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def list_documents(self) -> list[dict]:
        """List all unique documents in the store."""
        collection = self._get_collection()

        # Get all metadatas
        results = collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        # Group by document_id
        docs = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {"document_id": doc_id, "chunk_count": 0, "sources": set()}
            docs[doc_id]["chunk_count"] += 1
            if "source" in meta:
                docs[doc_id]["sources"].add(meta["source"])

        # Convert sets to lists for JSON
        for doc in docs.values():
            doc["sources"] = list(doc["sources"])

        return list(docs.values())

    def count(self) -> int:
        """Total chunks in store."""
        return self._get_collection().count()

    def clear(self):
        """Clear all data from the store."""
        client = self._get_client()
        client.delete_collection("documents")
        self._collection = None
