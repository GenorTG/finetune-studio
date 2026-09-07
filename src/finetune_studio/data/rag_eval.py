"""RAG evaluation harness — proper test suite for any RAG corpus.

Metrics implemented:
  - Recall@k           = (#questions where the right doc is in top-k) / total
  - MRR                = mean reciprocal rank of first correct doc
  - LLM-as-judge accuracy = #questions answered with all required facts
  - Round-trip portability (build → tar → fresh dir → load → query → identical
    top-1 hit)

Each test run writes to `corpus_dir/tests/<timestamp>/results.json` and updates
`corpus_dir/tests/history.json` for regression tracking.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class QAEntry:
    """One test question: query, list of must-contain fact strings, optional
    preferred source-doc substring (for retrieval tests)."""
    id: str
    query: str
    must_contain: list[str]
    expected_source_contains: str = ""   # expected source string (e.g. filename)
    difficulty: str = "easy"            # easy | medium | hard
    category: str = "general"
    notes: str = ""


@dataclass
class RetrievalResult:
    question_id: str
    query: str
    expected_source: str
    top_sources: list[str]
    top_scores: list[float]
    reciprocal_rank: float   # 1/rank of first match, or 0
    hit_at_k: bool


@dataclass
class LLMJudgeResult:
    question_id: str
    query: str
    answer: str
    must_contain: list[str]
    missing: list[str]
    passed: bool


@dataclass
class EvalReport:
    corpus: str = ""
    timestamp: float = 0.0
    embedding_model: str = ""
    total_questions: int = 0
    retrieval_results: list = field(default_factory=list)
    llm_judge_results: list = field(default_factory=list)
    recall_at_k: dict = field(default_factory=dict)   # {k: float}
    mrr: float = 0.0
    llm_pass_rate: float = 0.0
    portability_test: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "corpus": self.corpus,
            "timestamp": self.timestamp,
            "embedding_model": self.embedding_model,
            "total_questions": self.total_questions,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "llm_pass_rate": self.llm_pass_rate,
            "portability_test": self.portability_test,
            "timing": self.timing,
            "retrieval": [
                {"qid": r.question_id, "expected": r.expected_source,
                 "top": r.top_sources[:5], "rr": r.reciprocal_rank, "hit@": r.hit_at_k}
                for r in self.retrieval_results
            ],
            "llm": [
                {"qid": r.question_id, "passed": r.passed, "missing": r.missing}
                for r in self.llm_judge_results
            ],
        }


# ── Retrieval metrics ───────────────────────────────────────────────────

def evaluate_retrieval(rag_query, qa: list[QAEntry], ks: list[int] = (1, 3, 5)) -> tuple[list[RetrievalResult], dict]:
    """For each question, search the corpus, record whether the expected source
    appears in top-k and at what rank."""
    results = []
    recall = {k: 0 for k in ks}
    mrr_sum = 0.0

    for entry in qa:
        hits = rag_query.search(entry.query, top_k=max(ks))
        top_sources = [h["source"] for h in hits]
        rr = 0.0
        first_hit = False
        for rank, src in enumerate(top_sources, 1):
            if entry.expected_source_contains and entry.expected_source_contains in src:
                rr = 1.0 / rank
                first_hit = True
                break
        results.append(RetrievalResult(
            question_id=entry.id,
            query=entry.query,
            expected_source=entry.expected_source_contains,
            top_sources=top_sources,
            top_scores=[h["score"] for h in hits],
            reciprocal_rank=rr,
            hit_at_k=any(entry.expected_source_contains in s for s in top_sources[:max(ks)]),
        ))
        mrr_sum += rr
        for k in ks:
            if any(entry.expected_source_contains in s for s in top_sources[:k]):
                recall[k] += 1

    n = max(1, len(qa))
    recall_avg = {k: round(recall[k] / n, 3) for k in ks}
    mrr = round(mrr_sum / n, 3)

    return results, {"recall_at_k": recall_avg, "mrr": mrr}


# ── LLM-as-judge ───────────────────────────────────────────────────────

def llm_as_judge(mgr, rag_query, qa: list[QAEntry],
                 top_k: int = 5, max_tokens: int = 400,
                 temperature: float = 0.0) -> list[LLMJudgeResult]:
    """For each question, retrieve top-k + ask model to answer using ONLY context.

    Returns whether the model's answer includes all `must_contain` facts.
    """
    out = []
    for entry in qa:
        hits = rag_query.search(entry.query, top_k=top_k)
        ctx = rag_query.format_context(hits, max_chars=5000)
        prompt = (
            "You are answering questions about an internal Helios Industries corpus. "
            "Use ONLY the context below. If the answer is not in the context, reply exactly: "
            "\"I don't know from the provided documents.\" Otherwise answer concisely. "
            "Quote the source filename when relevant.\n\n"
            f"CONTEXT:\n{ctx}\n\nQUESTION: {entry.query}"
        )
        ans = mgr.chat(
            [{"role": "system", "content":
              "Answer using only the provided context. Quote source filename. "
              "If the answer isn't in the context, say you don't know."},
             {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, top_p=0.9,
        ).strip()
        missing = [f for f in entry.must_contain if f.lower() not in ans.lower()]
        out.append(LLMJudgeResult(
            question_id=entry.id, query=entry.query, answer=ans,
            must_contain=list(entry.must_contain), missing=missing,
            passed=not missing,
        ))
    return out


# ── Portability round-trip ─────────────────────────────────────────────

def portability_roundtrip(corpus: PortableRAG) -> dict:
    """Build corpus → tar it → untar to fresh dir → load → query → compare top-1.

    A portable RAG must round-trip without loss.
    """
    from finetune_studio.data.rag_portable import PortableRAG as PR

    if not corpus.exists():
        return {"ok": False, "error": "no corpus built"}

    # Get baseline top-1 for a known query
    q = "Who is the CEO of Helios Industries?"
    baseline_q = corpus.load()
    baseline_top1 = baseline_q.search(q, top_k=1)[0]["source"]

    # tar + untar into a fresh dir
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "corpus.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(corpus.dir), arcname="corpus")
        # extract to fresh path
        fresh_dir = Path(tmp) / "fresh_copy"
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(fresh_dir)
        # rename extracted "corpus" inner dir
        src_inner = fresh_dir / "corpus"
        if not src_inner.exists():
            src_inner = fresh_dir
        fresh = PR(src_inner)
        if not fresh.exists():
            return {"ok": False, "error": "no corpus after untar"}
        fresh_q = fresh.load()
        fresh_top1 = fresh_q.search(q, top_k=1)[0]["source"]

        ok = (baseline_top1 == fresh_top1)
        # Compare vectors (they must be bit-identical)
        base_vec = np.load(corpus.dir / "vectors.npy")
        fresh_vec = np.load(src_inner / "vectors.npy")
        vec_match = base_vec.shape == fresh_vec.shape and np.allclose(base_vec, fresh_vec, atol=1e-6)

        return {
            "ok": bool(ok and vec_match),
            "baseline_top1": baseline_top1,
            "fresh_top1": fresh_top1,
            "vectors_identical": bool(vec_match),
            "shape": list(base_vec.shape),
            "size_bytes_corpus": sum(p.stat().st_size for p in corpus.dir.rglob("*") if p.is_file()),
        }


# ── Results writer ─────────────────────────────────────────────────────

def write_results(corpus_dir: Path, report: EvalReport) -> Path:
    tests_dir = corpus_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = tests_dir / stamp
    run_dir.mkdir(exist_ok=True)
    out = run_dir / "results.json"
    out.write_text(json.dumps(report.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    # Append to history
    history_path = tests_dir / "history.json"
    history = {"runs": []}
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            pass
    history["runs"].append({"run": stamp, **report.to_json()})
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
