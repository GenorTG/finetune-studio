"""RAG query — retrieve context and format prompts.

WHAT THIS FILE DOES
==================
At query time:
  1. Embed the user's question
  2. Search the vector store for top-k similar chunks
  3. Format those chunks as context for the LLM
  4. Return the formatted prompt (or the augmented chat messages)

KEY CONCEPTS
============
- Top-k retrieval: take the k most similar chunks (default 5).
- Score threshold: ignore chunks with similarity below a threshold
  (e.g., 0.3) to avoid irrelevant context.
- Context template: how to format the chunks into a prompt. Example:
  "Use the following context to answer:
{context}

Question: {query}"
- Max context length: limit the total chunks so we don't exceed the
  model's context window.
"""

"""RAG query — retrieve context and inject into prompts."""
from dataclasses import dataclass


@dataclass
class RAGConfig:
    top_k: int = 5
    min_score: float = 0.3
    context_template: str = "Based on the following documents:\n\n{context}\n\nAnswer the question."
    max_context_length: int = 2000


class RAGQuery:
    """Query engine that combines retrieval with generation."""

    def __init__(self, store, config: RAGConfig = None):
        self.store = store
        self.config = config or RAGConfig()

    def retrieve(self, query: str, top_k: int | None = None) -> list:
        """Retrieve relevant chunks for a query."""
        k = top_k or self.config.top_k
        results = self.store.search(query, top_k=k)

        # Filter by minimum score
        return [r for r in results if r.score >= self.config.min_score]

    def build_context(self, query: str, top_k: int | None = None) -> str:
        """Build context string from retrieved chunks."""
        results = self.retrieve(query, top_k)

        if not results:
            return ""

        context_parts = []
        total_length = 0

        for r in results:
            chunk_text = f"[Source: {r.metadata.get('source', 'unknown')}]\n{r.text}"
            if total_length + len(chunk_text) > self.config.max_context_length:
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

    def augment_prompt(self, query: str, system_prompt: str = "", top_k: int | None = None) -> list:
        """Build augmented message list with RAG context."""
        context = self.build_context(query, top_k)

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context:
            rag_system = f"{self.config.context_template.format(context=context)}"
            if system_prompt:
                messages.append({"role": "system", "content": rag_system})
            else:
                messages.append({"role": "system", "content": rag_system})

        messages.append({"role": "user", "content": query})

        return messages

    def query(self, inference_engine, query: str, system_prompt: str = "",
              max_tokens: int = 512, temperature: float = 0.7, top_k: int | None = None) -> dict:
        """Full RAG query — retrieve, augment, generate."""
        results = self.retrieve(query, top_k)
        context = self.build_context(query, top_k)
        messages = self.augment_prompt(query, system_prompt, top_k)

        response = inference_engine.generate(
            messages, max_tokens=max_tokens, temperature=temperature
        )

        return {
            "response": response,
            "sources": [
                {
                    "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                    "score": round(r.score, 3),
                    "source": r.metadata.get("source", "unknown"),
                    "document_id": r.document_id,
                }
                for r in results
            ],
            "context_used": bool(context),
            "chunks_retrieved": len(results),
        }
