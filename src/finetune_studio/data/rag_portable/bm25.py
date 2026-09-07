"""Okapi BM25 index over a tokenized corpus.

Single responsibility: classic Okapi BM25 — build from docs, score a query,
persist to JSON via to_dict/from_dict.

Why in-process: corpora fit in RAM, BM25 scoring is O(query_terms * postings),
and serializing/deserializing is fast enough for our scale.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from finetune_studio.data.rag_portable.tokenize import tokenize


class BM25Index:
    """Okapi BM25 over a tokenized corpus. Persistent in bm25.json."""

    def __init__(self):
        self.terms: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_id, tf)]
        self.doc_lens: list[int] = []
        self.df: dict[str, int] = {}
        self.avgdl: float = 0.0
        self.doc_count: int = 0
        self.k1: float = 1.5
        self.b: float = 0.75

    @classmethod
    def build(cls, docs: list[str]) -> "BM25Index":
        idx = cls()
        idx.doc_count = len(docs)
        if idx.doc_count == 0:
            return idx
        idx.doc_lens = []
        # Per-doc tokenize + term counts
        tokenized = []
        for doc in docs:
            toks = tokenize(doc)
            tokenized.append(toks)
            idx.doc_lens.append(len(toks))
        idx.avgdl = sum(idx.doc_lens) / idx.doc_count
        # Build inverted index
        for i, toks in enumerate(tokenized):
            counts = Counter(toks)
            for term, tf in counts.items():
                idx.terms.setdefault(term, []).append((i, tf))
        idx.df = {term: len(postings) for term, postings in idx.terms.items()}
        return idx

    def score(self, query: str) -> np.ndarray:
        """Returns float32 array of length doc_count with BM25 score per doc."""
        scores = np.zeros(self.doc_count, dtype=np.float32)
        q_tokens = tokenize(query)
        if not q_tokens:
            return scores
        for term in set(q_tokens):
            postings = self.terms.get(term)
            if not postings:
                continue
            df = self.df[term]
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings:
                dl = self.doc_lens[doc_id]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * (num / den)
        return scores

    def to_dict(self) -> dict:
        # Convert term postings (list of tuples) to JSON-safe (list of [doc_id, tf])
        terms_safe = {t: [[d, tf] for (d, tf) in postings] for t, postings in self.terms.items()}
        return {
            "terms": terms_safe,
            "doc_lens": self.doc_lens,
            "df": self.df,
            "avgdl": self.avgdl,
            "doc_count": self.doc_count,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BM25Index":
        idx = cls()
        idx.terms = {t: [(d_id, tf) for d_id, tf in postings] for t, postings in d["terms"].items()}
        idx.doc_lens = d["doc_lens"]
        idx.df = d["df"]
        idx.avgdl = d["avgdl"]
        idx.doc_count = d["doc_count"]
        idx.k1 = d.get("k1", 1.5)
        idx.b = d.get("b", 0.75)
        return idx
