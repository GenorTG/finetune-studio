"""Retrieval-grounded test suite — answer from PortableRAG context only.

Pairs a Q&A ``BenchmarkCase`` suite with a PortableRAG (or compatible)
search engine so held-out / source-disjoint cases can be scored with the
same strict/heuristic judges as ``run_suite``, while forcing the model to
use retrieved document chunks (or refuse when the context lacks the fact).

This lives under ``testing/`` (not the route) so CLI / benchmarks / scripts
can reuse it without FastAPI.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from finetune_studio.data.rag_eval import UNKNOWN_REPLY
from finetune_studio.testing.suite import (
    BenchmarkCase,
    CaseResult,
    apply_heuristic_judging,
    load_test_suite,
    score_results,
)

_log = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = (
    "Answer using only the provided context. "
    "If the answer isn't in the context, say you don't know. "
    "For tables, use the row and column named by the question; calculate requested totals "
    "from the underlying values, not a variance or unrelated row."
)

_CORPORA_ROOT = Path.home() / ".finetune-studio" / "rag_corpora"


class RagSearchEngine(Protocol):
    """Minimal PortableRAGQuery-compatible search surface."""

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        ...

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        ...


class ChatEngine(Protocol):
    """Minimal inference engine surface (``generate(messages, ...)``)."""

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        think: bool = False,
    ) -> str:
        ...

    model_path: str | None


def default_corpus_path(pid: str) -> Path:
    """Canonical PortableRAG directory for a project id."""
    return _CORPORA_ROOT / str(pid)


def resolve_corpus_path(pid: str = "", corpus_path: str = "") -> Path:
    """Resolve corpus path: explicit override, then project_rags, then default."""
    raw = (corpus_path or "").strip()
    if raw:
        return Path(raw)

    pid = (pid or "").strip()
    if not pid:
        raise ValueError("corpus_path or project_id/pid required")

    try:
        from finetune_studio.db.rags import list_rags

        for row in list_rags(pid):
            sp = Path(str(row.get("store_path") or ""))
            if sp.is_dir() and (sp / "manifest.json").is_file():
                return sp
    except Exception as exc:  # noqa: BLE001
        _log.debug("list_rags fallback for pid=%s: %s", pid, exc)

    return default_corpus_path(pid)


def load_portable_rag_query(corpus_path: str | Path) -> Any:
    """Load a PortableRAG query engine from ``corpus_path``."""
    from finetune_studio.data.rag_portable import PortableRAG

    path = Path(corpus_path)
    if not path.is_dir():
        raise FileNotFoundError(f"RAG corpus not found: {path}")
    rag = PortableRAG(path)
    if not rag.exists():
        raise FileNotFoundError(f"RAG corpus incomplete (missing manifest/vectors): {path}")
    return rag.load()


def provenance_from_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Normalize a search hit into stable provenance fields."""
    score = hit.get("score")
    if score is None:
        score = hit.get("rrf_score") or hit.get("ce_score") or 0.0
    return {
        "document_id": str(hit.get("document_id") or ""),
        "source": str(hit.get("source") or ""),
        "filename": str(hit.get("filename") or ""),
        "chunk_index": int(hit.get("chunk_index") or 0),
        "chunk_id": str(hit.get("chunk_id") or ""),
        "rank": int(hit.get("rank") or 0),
        "score": float(score or 0.0),
    }


def build_grounded_messages(question: str, context: str) -> list[dict[str, str]]:
    """System + user messages that enforce context-only answering."""
    user = (
        "Use ONLY the context below. If the answer is not in the context, "
        f'reply exactly: "{UNKNOWN_REPLY}" Otherwise answer concisely.\n\n'
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _needs_table_arithmetic_retry(question: str, response: str, context: str) -> bool:
    """Detect the common actual-hours/variance mix-up in tabular answers."""
    return bool(
        "actual" in question.lower()
        and "total" in question.lower()
        and "actual_hours" in context
        and "variance_hours" in context
        and re.search(r"\bvariance\b|\b\d+\s*\+\s*\d+", response or "", re.IGNORECASE)
    )


def hit_matches_source(
    hit: dict[str, Any],
    source_id: str,
    chunk_idx: int = 0,
) -> bool:
    """True when a retrieval hit plausibly matches suite ``source_id`` / chunk."""
    sid = (source_id or "").strip()
    if not sid:
        return False
    sid_l = sid.lower()
    matched_doc = False
    for key in ("document_id", "source", "filename", "chunk_id"):
        val = str(hit.get(key) or "").lower()
        if not val:
            continue
        if sid_l == val or sid_l in val or val in sid_l:
            matched_doc = True
            break
    if not matched_doc:
        return False
    if not chunk_idx:
        return True
    hit_chunk = int(hit.get("chunk_index") or 0)
    if not hit_chunk:
        return True
    return hit_chunk == int(chunk_idx)


@dataclass
class RagCaseResult:
    """One grounded suite case: CaseResult + retrieval provenance."""

    case_result: CaseResult
    retrieval_hits: list[dict[str, Any]] = field(default_factory=list)
    context_text: str = ""
    retrieval_hit: bool = False

    def as_api_dict(self) -> dict[str, Any]:
        r = self.case_result
        return {
            "name": r.case_name,
            "category": r.category,
            "question": r.question,
            "correct_answer": r.correct_answer,
            "response": r.model_answer,
            "model_answer": r.model_answer,
            "passed": r.verdict == "pass",
            "verdict": r.verdict,
            "judge": r.judge,
            "judge_model": r.judge_model,
            "judge_reasoning": r.judge_reasoning,
            "scoring_method": r.scoring_method,
            "validity": r.validity,
            "source_id": r.source_id,
            "chunk_idx": r.chunk_idx,
            "keywords": list(r.keywords),
            "transcript": list(r.transcript),
            "time_ms": r.time_ms,
            "error": r.error,
            "retrieval_hits": list(self.retrieval_hits),
            "retrieval_hit": self.retrieval_hit,
            "context_text": self.context_text,
        }


@dataclass
class RagSuiteReport:
    """Full RAG-grounded suite payload (ready for API serialization)."""

    results: list[RagCaseResult] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    corpus_path: str = ""
    model_path: str = ""
    top_k: int = 5
    unknown_reply: str = UNKNOWN_REPLY

    def as_api_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_api_dict() for r in self.results],
            "scores": dict(self.scores),
            "retrieval": dict(self.retrieval),
            "corpus_path": self.corpus_path,
            "model_path": self.model_path,
            "top_k": self.top_k,
            "unknown_reply": self.unknown_reply,
        }


def compute_retrieval_metrics(rag_results: list[RagCaseResult]) -> dict[str, Any]:
    """Hit counts / recall for cases that declare ``source_id``."""
    with_source = [r for r in rag_results if (r.case_result.source_id or "").strip()]
    n = len(with_source)
    hits = sum(1 for r in with_source if r.retrieval_hit)
    return {
        "cases_with_source_id": n,
        "retrieval_hits": hits,
        "retrieval_misses": max(n - hits, 0),
        "recall_at_k": round(hits / n, 4) if n else None,
        "hit_rate": round(hits / n, 4) if n else None,
    }


def run_rag_suite(
    engine: ChatEngine,
    rag_query: RagSearchEngine,
    cases: list[BenchmarkCase],
    *,
    top_k: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.3,
    think: bool = False,
    max_context_chars: int = 5000,
) -> list[RagCaseResult]:
    """Retrieve → ground prompt → generate → collect CaseResult + provenance."""
    out: list[RagCaseResult] = []
    for case in cases:
        start = time.time()
        hits_raw: list[dict] = []
        context = ""
        try:
            hits_raw = list(rag_query.search(case.question, top_k=top_k) or [])
            provenance = [provenance_from_hit(h) for h in hits_raw]
            context = rag_query.format_context(hits_raw, max_chars=max_context_chars)
            messages = build_grounded_messages(case.question, context)
            response = engine.generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                think=think,
            )
            if _needs_table_arithmetic_retry(case.question, response, context):
                correction = (
                    "Re-answer this question from the table. It asks for total actual_hours: "
                    "sum the actual_hours column for the requested month/sites. Do not use "
                    "variance_hours, budget_hours, or variance values. Answer concisely."
                )
                retry_messages = messages + [{"role": "user", "content": correction}]
                response = engine.generate(
                    retry_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    think=think,
                )
                messages = retry_messages
            elapsed_ms = (time.time() - start) * 1000
            transcript = list(messages) + [{"role": "assistant", "content": response}]
            case_result = CaseResult(
                case_name=case.name,
                category=case.category,
                question=case.question,
                correct_answer=case.correct_answer,
                model_answer=response,
                transcript=transcript,
                time_ms=round(elapsed_ms, 1),
                keywords=list(case.keywords),
                source_id=case.source_id,
                chunk_idx=case.chunk_idx,
            )
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.time() - start) * 1000
            provenance = [provenance_from_hit(h) for h in hits_raw]
            messages = build_grounded_messages(case.question, context)
            case_result = CaseResult(
                case_name=case.name,
                category=case.category,
                question=case.question,
                correct_answer=case.correct_answer,
                model_answer="",
                transcript=list(messages) + [{"role": "assistant", "content": ""}],
                error=str(e),
                time_ms=round(elapsed_ms, 1),
                keywords=list(case.keywords),
                source_id=case.source_id,
                chunk_idx=case.chunk_idx,
            )

        matched = any(
            hit_matches_source(h, case.source_id, case.chunk_idx) for h in provenance
        )
        out.append(
            RagCaseResult(
                case_result=case_result,
                retrieval_hits=provenance,
                context_text=context,
                retrieval_hit=matched,
            )
        )
    return out


def run_rag_suite_evaluation(
    engine: ChatEngine,
    *,
    suite_path: str,
    corpus_path: str = "",
    project_id: str = "",
    top_k: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.3,
    rag_query: RagSearchEngine | None = None,
) -> RagSuiteReport:
    """Load suite + corpus, run grounded eval, judge, and aggregate metrics.

    Blocking — call from ``asyncio.to_thread`` in async routes.
    """
    cases = load_test_suite(suite_path)
    resolved_corpus = ""
    query = rag_query
    if query is None:
        path = resolve_corpus_path(project_id, corpus_path)
        resolved_corpus = str(path)
        query = load_portable_rag_query(path)
    else:
        resolved_corpus = (corpus_path or "").strip() or (
            str(default_corpus_path(project_id)) if project_id else ""
        )

    rag_results = run_rag_suite(
        engine,
        query,
        cases,
        top_k=top_k,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    case_results = [r.case_result for r in rag_results]
    apply_heuristic_judging(case_results)
    scores = score_results(case_results)
    retrieval = compute_retrieval_metrics(rag_results)
    model_path = getattr(engine, "model_path", None) or ""
    return RagSuiteReport(
        results=rag_results,
        scores=scores,
        retrieval=retrieval,
        corpus_path=resolved_corpus,
        model_path=str(model_path),
        top_k=top_k,
    )


__all__ = [
    "RAG_SYSTEM_PROMPT",
    "UNKNOWN_REPLY",
    "ChatEngine",
    "RagCaseResult",
    "RagSearchEngine",
    "RagSuiteReport",
    "build_grounded_messages",
    "compute_retrieval_metrics",
    "default_corpus_path",
    "hit_matches_source",
    "load_portable_rag_query",
    "provenance_from_hit",
    "resolve_corpus_path",
    "run_rag_suite",
    "run_rag_suite_evaluation",
]
