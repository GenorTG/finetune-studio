"""PortableRAGQuery — loaded corpus + search.

Returned by PortableRAG.load(). Owns the in-memory vectors + BM25 and the search pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from finetune_studio.data.rag_portable.bm25 import BM25Index
from finetune_studio.data.rag_portable.rerankers import get_reranker
from finetune_studio.data.rag_portable.rrf import rrf_fuse
from finetune_studio.data.rag_portable.schema import Manifest


class PortableRAGQuery:
    """Loaded corpus + search."""

    def __init__(self, corpus_dir: Path, manifest: Manifest, chunks,
                 vectors: np.ndarray, idx_map: dict, bm25: BM25Index, encode):
        self.dir = corpus_dir
        self.manifest = manifest
        self.chunks = chunks
        self.vectors = vectors
        self.idx_map = idx_map
        self.bm25 = bm25
        self.encode = encode
        self._chunk_ids = chunks["id"].tolist()
        self._reranker = None

    def _ensure_reranker(self):
        if (self._reranker is None
                and self.manifest.rag_settings.rerank_enabled
                and self.manifest.rag_settings.reranker):
            self._reranker, _ = get_reranker(
                name=self.manifest.rag_settings.reranker, device="cpu"
            )
        return self._reranker

    def search(self, query: str, top_k: int = 5,
               hybrid: bool | None = None,
               rerank: bool | None = None,
               rerank_top_n: int | None = None) -> list[dict]:
        """Top-k hits. hybrid/rerank default to manifest settings."""
        if not query.strip():
            return []
        s = self.manifest.rag_settings
        use_hybrid = s.hybrid_enabled if hybrid is None else hybrid
        use_rerank = s.rerank_enabled if rerank is None else rerank
        n_before_rerank = rerank_top_n or s.rerank_top_n

        q = self.encode(query)
        q_dim = int(q.shape[0]) if getattr(q, "ndim", 0) == 1 else int(q.shape[-1])
        corpus_dim = int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0
        if corpus_dim and q_dim != corpus_dim:
            raise ValueError(
                f"Query embedding dimension {q_dim} does not match corpus "
                f"vectors.npy dimension {corpus_dim} "
                f"(embedder={self.manifest.embedding_model.name!r}). "
                f"Rebuild the corpus or reload with the manifest embedder."
            )
        dense_scores = self.vectors @ q

        # Build rankings
        dense_rank = [self._chunk_ids[i] for i in np.argsort(-dense_scores)]
        if use_hybrid:
            bm25_scores = self.bm25.score(query)
            bm25_rank = [self._chunk_ids[i] for i in np.argsort(-bm25_scores)]
        else:
            bm25_rank = dense_rank

        # RRF
        rrf_scores = rrf_fuse([dense_rank, bm25_rank], k=s.rrf_k)
        candidates = sorted(rrf_scores.items(), key=lambda kv: -kv[1])
        candidates = candidates[: max(top_k, n_before_rerank if use_rerank else top_k)]

        # Initial ranking result (pre-rerank)
        results = []
        for rank, (cid, sc) in enumerate(candidates, start=1):
            i = self.idx_map[cid]
            results.append({
                "rank": rank, "chunk_id": cid, "score": float(sc),
                "rrf_score": float(sc),
                "dense_score": float(dense_scores[i]),
                "bm25_score": float(self.bm25.score(query)[i]) if use_hybrid else 0.0,
                "text": self.chunks.iloc[i]["text"],
                "source": self.chunks.iloc[i]["source"],
                "filename": self.chunks.iloc[i]["filename"],
                "document_id": self.chunks.iloc[i]["document_id"],
                "chunk_index": int(self.chunks.iloc[i]["chunk_index"]),
            })

        # Rerank top-N
        if use_rerank and len(results) > top_k:
            reranker = self._ensure_reranker()
            if reranker is not None:
                rerank_pool = results[:n_before_rerank]
                docs_to_rerank = [r["text"] for r in rerank_pool]
                ce_scores = reranker(query, docs_to_rerank)
                # Replace scores with CE scores; re-sort
                for i, r in enumerate(rerank_pool):
                    r["ce_score"] = ce_scores[i]
                rerank_pool.sort(key=lambda r: -r["ce_score"])
                # Pad with non-reranked tail
                results = rerank_pool + results[n_before_rerank:]

        # Re-number ranks and trim
        for rank, r in enumerate(results[:top_k], start=1):
            r["rank"] = rank
        return results[:top_k]

    def list_sources(self) -> list[dict]:
        """List all sources in the corpus."""
        sources_dir = self.dir / "sources"
        if not sources_dir.exists():
            return []
        # Build a map of document_id -> filename from chunks
        doc_to_filename = {}
        if self.chunks is not None and len(self.chunks) > 0:
            for _, row in self.chunks.iterrows():
                doc_id = row.get("document_id")
                if doc_id and doc_id not in doc_to_filename:
                    doc_to_filename[doc_id] = row.get("filename", "")
        sources = []
        for f in sorted(sources_dir.glob("*.txt")):
            doc_id = f.stem
            sources.append({
                "id": doc_id,
                "filename": doc_to_filename.get(doc_id, doc_id),
                "size": f.stat().st_size,
            })
        return sources

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        blocks = []
        total = 0
        for r in results:
            score_str = r.get("ce_score") if "ce_score" in r else r.get("rrf_score", 0)
            block = f"[{r['rank']}] (source: {r['source']}, score {score_str:.3f})\n{r['text']}"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n---\n\n".join(blocks)
