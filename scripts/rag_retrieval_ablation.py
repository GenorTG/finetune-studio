#!/usr/bin/env python3
"""Retrieval-only ablation: which retrieval method puts the expected values in the top-k?

Loads a project's PortableRAG corpus in-process (no chat model, no server round-trip) and, for every
quiz question that lists expected values, checks whether the union of the top-k chunks contains
all of them (recall@k), once per method:

    dense              cosine over the stored embedder vectors (what the app does today)
    bm25               Okapi BM25 only
    hybrid             dense + BM25 fused with RRF
    hybrid+rerank      the app default: hybrid top-50, cross-encoder reranked
    hybrid+rerankrrf   hybrid top-50 whose order is RRF-fused with the cross-encoder's (reranker as one vote)
    bm25+rerank        BM25 top-50, cross-encoder reranked
    dense/hybrid +e5   the same, with the e5 prefixes the model card requires ("query: ", "passage: ")

The ``+e5`` rows re-embed the chunks in memory (nothing on disk changes).

    .venv/bin/python scripts/rag_retrieval_ablation.py --pid e9f951f8 --suite <imported-suite.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from finetune_studio.data.fs.paths import rag_corpus_dir
from finetune_studio.data.rag_portable import PortableRAG
from finetune_studio.data.rag_portable.rrf import rrf_fuse
from finetune_studio.testing.rag_suite import _norm_text
from finetune_studio.testing.suite import load_test_suite

KS = (1, 3, 5, 10, 20, 50)
POOL = 50


class Ablation:
    """One loaded corpus plus the derived views the methods need."""

    def __init__(self, q) -> None:
        self.q = q
        self.ids: list[str] = q._chunk_ids
        self.texts: list[str] = [str(t) for t in q.chunks["text"].tolist()]
        self.norm_texts = [_norm_text(t) for t in self.texts]
        self.rerank = q._ensure_reranker()
        self._e5_vectors: np.ndarray | None = None

    def e5_vectors(self) -> np.ndarray:
        if self._e5_vectors is None:
            self._e5_vectors = self.q.encode(["passage: " + t for t in self.texts])
        return self._e5_vectors

    def order(self, query: str, method: str) -> list[int]:
        """Chunk row indexes best-first for ``method`` (see module docstring)."""
        q = self.q
        e5 = "+e5" in method
        base = method.replace("+e5", "")
        dense_rank: list[int] = []
        if base in ("dense", "hybrid", "hybrid+rerank", "hybrid+rerankrrf"):
            vecs, qv = (self.e5_vectors(), q.encode("query: " + query)) if e5 else (q.vectors, q.encode(query))
            dense_rank = [int(i) for i in np.argsort(-(vecs @ qv))]
        bm25_rank = [int(i) for i in np.argsort(-q.bm25.score(query))] if base.startswith(("bm25", "hybrid")) else []
        if base == "dense":
            return dense_rank
        if base == "bm25":
            return bm25_rank
        if base.startswith("hybrid"):
            fused = rrf_fuse([[str(i) for i in dense_rank], [str(i) for i in bm25_rank]], k=q.manifest.rag_settings.rrf_k)
            ranked = [int(i) for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
        else:
            ranked = bm25_rank
        if base.endswith(("+rerank", "+rerankrrf")) and self.rerank is not None:
            pool = ranked[:POOL]
            scores = self.rerank(query, [self.texts[i] for i in pool])
            by_score = [i for _, i in sorted(zip(scores, pool, strict=True), key=lambda p: -p[0])]
            if base.endswith("+rerankrrf"):
                fused = rrf_fuse([[str(i) for i in pool], [str(i) for i in by_score]], k=q.manifest.rag_settings.rrf_k)
                by_score = [int(i) for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
            ranked = by_score + ranked[POOL:]
        return ranked

    def covered(self, ranked: list[int], keywords: list[str], k: int) -> bool:
        seen = " ".join(self.norm_texts[i] for i in ranked[:k])
        return all(_norm_text(w) in seen for w in keywords)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--suite", type=Path, required=True, help="imported suite JSON (/api/benchmarks/projects/<pid>/suites/import)")
    ap.add_argument("--out", type=Path, default=None, help="write the per-question coverage as JSON")
    args = ap.parse_args()

    ab = Ablation(PortableRAG(rag_corpus_dir(args.pid)).load())
    cases = [c for c in load_test_suite(str(args.suite)) if c.keywords and not c.expect_abstain]
    methods = ("dense", "bm25", "hybrid", "hybrid+rerank", "hybrid+rerankrrf", "bm25+rerank", "dense+e5", "hybrid+e5", "hybrid+rerank+e5")
    cover: dict[str, dict[str, dict[int, bool]]] = {m: {} for m in methods}
    for c in cases:
        for m in methods:
            ranked = ab.order(c.question, m)
            cover[m][c.name] = {k: ab.covered(ranked, c.keywords, k) for k in KS}

    print(f"{len(cases)} answerable questions; recall@k = all expected values inside the union of the top-k chunks")
    print(f"{'method':<20}" + "".join(f"@{k:<5}" for k in KS))
    for m in methods:
        print(f"{m:<20}" + "".join(f"{sum(v[k] for v in cover[m].values()):<6}" for k in KS))
    if args.out:
        args.out.write_text(json.dumps(cover, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
