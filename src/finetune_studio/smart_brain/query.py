"""BrainQuery — loaded corpus + search.

Returned by BrainIndex.load(). Owns the in-memory vectors + BM25 and the search pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from finetune_studio.smart_brain.bm25 import BM25Index
from finetune_studio.smart_brain.constants import DEFAULT_TOP_K

# Process-wide cached reranker (model_cache); keep the ``get_reranker`` name — tests patch it.
from finetune_studio.smart_brain.model_cache import (
    cached_reranker as get_reranker,
)
from finetune_studio.smart_brain.prompt import format_context_blocks
from finetune_studio.smart_brain.rrf import rrf_fuse
from finetune_studio.smart_brain.schema import Manifest
from finetune_studio.smart_brain.source_labels import prettify_source_label


class BrainQuery:
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
                name=self.manifest.rag_settings.reranker, device="auto"
            )
        return self._reranker

    def search(self, query: str, top_k: int = DEFAULT_TOP_K,
               hybrid: bool | None = None,
               rerank: bool | None = None,
               rerank_top_n: int | None = None,
               max_per_document: int | None = None,
               source_prefix: str = "") -> list[dict]:
        """Top-k hits. hybrid/rerank/max_per_document default to manifest settings (``smart_brain.diversity``)."""
        from finetune_studio.smart_brain.diversity import (
            CANDIDATES_PER_RESULT,
            diversify,
        )

        if not query.strip():
            return []
        s = self.manifest.rag_settings
        use_hybrid = s.hybrid_enabled if hybrid is None else hybrid
        use_rerank = s.rerank_enabled if rerank is None else rerank
        n_before_rerank = rerank_top_n or s.rerank_top_n
        cap = s.max_per_document if max_per_document is None else int(max_per_document)

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
        allowed = np.arange(len(self._chunk_ids))
        if source_prefix:
            allowed = np.array([
                i for i, row in self.chunks.iterrows()
                if str(row.get("source") or "").startswith(source_prefix)
                or str(row.get("filename") or "").startswith(source_prefix)
            ], dtype=int)
            if allowed.size == 0:
                return []

        # Build rankings
        dense_rank = [self._chunk_ids[i] for i in allowed[np.argsort(-dense_scores[allowed])]]
        if use_hybrid:
            bm25_scores = self.bm25.score(query)
            # Only passages sharing a term get a keyword rank: an arbitrary rank for the zero-score rest pushed a
            # passage found by meaning alone below ones matching just "you" / "w" (standalone_server does the same).
            order = allowed[np.argsort(-bm25_scores[allowed])]
            bm25_rank = [self._chunk_ids[i] for i in order if bm25_scores[i] > 0]
        else:
            bm25_scores = None
            bm25_rank = dense_rank

        # RRF
        rrf_scores = rrf_fuse([dense_rank, bm25_rank], k=s.rrf_k)
        candidates = sorted(rrf_scores.items(), key=lambda kv: -kv[1])
        pool = top_k * CANDIDATES_PER_RESULT if cap > 0 else top_k     # room to pass over a document's extra passages
        candidates = candidates[: max(pool, n_before_rerank if use_rerank else top_k)]

        # Initial ranking result (pre-rerank)
        results = []
        for rank, (cid, sc) in enumerate(candidates, start=1):
            i = self.idx_map[cid]
            raw_filename = self.chunks.iloc[i]["filename"]
            raw_source = self.chunks.iloc[i]["source"]
            results.append({
                "rank": rank, "chunk_id": cid, "score": float(sc),
                "rrf_score": float(sc),
                "dense_score": float(dense_scores[i]),
                "bm25_score": float(bm25_scores[i]) if use_hybrid else 0.0,
                "text": self.chunks.iloc[i]["text"],
                "source": raw_source,
                "filename": prettify_source_label(
                    str(raw_filename) if raw_filename is not None else "",
                    raw_source,
                ),
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

        # One document may not fill the top results; then re-number ranks
        results = diversify(results, top_k, cap)
        for rank, r in enumerate(results, start=1):
            r["rank"] = rank
        return results

    def search_images(
        self,
        query: str = "",
        *,
        image_data: bytes | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Find indexed images by multilingual text or visual similarity to an uploaded image."""
        if not query.strip() and not image_data:
            raise ValueError("enter a text query or upload an image")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        assets_path = self.dir / "image_assets.json"
        vectors_path = self.dir / "image_vectors.npy"
        if not assets_path.is_file() or not vectors_path.is_file():
            return []

        import json

        assets = json.loads(assets_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path, allow_pickle=False).astype(np.float32, copy=False)
        if vectors.ndim != 2 or len(assets) != vectors.shape[0]:
            raise ValueError("the image index is incomplete; rebuild the Smart Brain index")
        metadata = (self.manifest.extra or {}).get("image_search") or {}
        from finetune_studio.smart_brain.model_cache import cached_image_embedder

        embedder = cached_image_embedder()
        if (int(metadata.get("dimension") or 0) != embedder.dim
                or vectors.shape[1] != embedder.dim
                or metadata.get("image_model") != embedder.image_model
                or metadata.get("text_model") != embedder.text_model):
            raise ValueError("the image embedding model does not match this index; rebuild the Smart Brain index")
        if image_data:
            from finetune_studio.smart_brain.image_embeddings import image_from_bytes

            query_vector = embedder.encode_images([image_from_bytes(image_data)])[0]
        else:
            query_vector = embedder.encode_texts(query)
        scores = vectors @ np.asarray(query_vector, dtype=np.float32)
        best = np.argsort(-scores)[:top_k]
        return [{**assets[int(i)], "score": float(scores[int(i)]), "rank": rank}
                for rank, i in enumerate(best, 1)]

    def list_sources(self) -> list[dict]:
        """List current corpus documents from manifest / chunks (not stale dirs)."""
        sources: list[dict] = []
        meta = []
        extra = getattr(self.manifest, "extra", None) or {}
        if isinstance(extra, dict):
            meta = extra.get("documents_meta") or []
        if meta:
            for d in meta:
                did = str(d.get("document_id") or d.get("id") or "")
                if not did:
                    continue
                fname = prettify_source_label(
                    str(d.get("filename") or ""),
                    d.get("source"),
                )
                src_path = self.dir / "sources" / f"{did}.txt"
                size = src_path.stat().st_size if src_path.is_file() else 0
                sources.append({"id": did, "filename": fname, "size": size})
            return sources

        if self.chunks is None or len(self.chunks) == 0:
            return []
        if "document_id" not in self.chunks.columns:
            return []
        seen: set[str] = set()
        for _, row in self.chunks.iterrows():
            did = str(row.get("document_id") or "")
            if not did or did in seen:
                continue
            seen.add(did)
            fname = prettify_source_label(
                str(row.get("filename") or ""),
                row.get("source"),
            )
            src_path = self.dir / "sources" / f"{did}.txt"
            size = src_path.stat().st_size if src_path.is_file() else 0
            sources.append({"id": did, "filename": fname, "size": size})
        return sources

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        return format_context_blocks(results, max_chars=max_chars)
