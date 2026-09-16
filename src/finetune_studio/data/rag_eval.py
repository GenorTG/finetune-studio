"""RAG evaluation harness — retrieval, grounding, unknown behaviour, portability.

Metrics (explicitly labelled — do not conflate):
  - Recall@k / MRR          = retrieval of expected source strings
  - answer_grounding        = lexical overlap of answer vs retrieved context
  - fact_coverage           = must_contain substring check on model answers
                              (legacy name in older reports: ``llm_pass_rate``;
                              that was NEVER an LLM-as-judge — keep the alias
                              but mark method in metadata)
  - no_context_unknown      = empty-context answers must refuse / say unknown
  - portability             = tar round-trip preserves top-1 + vectors

Primary entrypoint: ``run_rag_evaluation(...)``.
CLI shim: ``finetune_studio.data.rag`` ``eval`` subcommand → ``run_eval_on_corpus``.
"""

from __future__ import annotations

import json
import re
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from finetune_studio.data.rag_portable import PortableRAG

# Stable phrase the no-context protocol expects (also accepted loosely).
UNKNOWN_REPLY = "I don't know from the provided documents."
_UNKNOWN_MARKERS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "not in the",
    "no information",
    "cannot determine",
    "can't determine",
    UNKNOWN_REPLY.lower(),
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for",
    "is", "are", "was", "were", "be", "with", "as", "by", "at", "from",
    "that", "this", "it", "its",
})

EVAL_SCHEMA_VERSION = "rag_eval_v1"


class RagSearchEngine(Protocol):
    """Minimal surface used by retrieval / grounding helpers."""

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        ...

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        ...


AnswerFn = Callable[[str, str], str]  # (query, context) -> answer text


@dataclass
class QAEntry:
    """One test question: query, required fact strings, optional source hint."""

    id: str
    query: str
    must_contain: list[str]
    expected_source_contains: str = ""  # expected source string (e.g. filename)
    difficulty: str = "easy"  # easy | medium | hard
    category: str = "general"
    notes: str = ""


@dataclass
class RetrievalResult:
    question_id: str
    query: str
    expected_source: str
    top_sources: list[str]
    top_scores: list[float]
    reciprocal_rank: float  # 1/rank of first match, or 0
    hit_at_k: bool


@dataclass
class GroundingResult:
    """Deterministic answer↔context lexical grounding (not an LLM judge)."""

    question_id: str
    query: str
    answer: str
    context: str
    grounding_overlap: float
    fact_coverage: float  # fraction of must_contain found in answer
    missing_facts: list[str]
    grounded: bool
    facts_covered: bool


@dataclass
class NoContextResult:
    question_id: str
    query: str
    answer: str
    refused: bool
    method: str = "unknown_phrase_match"


@dataclass
class FactCoverageResult:
    """Legacy must_contain check formerly labelled ``llm_as_judge``.

    This is substring fact coverage on a model answer — NOT LLM-as-judge scoring.
    """

    question_id: str
    query: str
    answer: str
    must_contain: list[str]
    missing: list[str]
    passed: bool
    method: str = "must_contain_substring"


# Back-compat alias — older imports / reports used this name.
LLMJudgeResult = FactCoverageResult


@dataclass
class EvalMetadata:
    """Structured provenance for one evaluation run."""

    schema_version: str = EVAL_SCHEMA_VERSION
    corpus: str = ""
    corpus_path: str = ""
    embedding_model: str = ""
    qa_set_path: str = ""
    total_questions: int = 0
    ks: list[int] = field(default_factory=lambda: [1, 3, 5])
    top_k_answer: int = 5
    ran_retrieval: bool = True
    ran_grounding: bool = False
    ran_fact_coverage: bool = False
    ran_no_context: bool = False
    ran_portability: bool = False
    # Honest labels for report consumers
    fact_coverage_method: str = "must_contain_substring"
    grounding_method: str = "lexical_overlap"
    no_context_method: str = "unknown_phrase_match"
    # Legacy field name was misleading; keep explicit disclaimer.
    legacy_llm_pass_rate_alias: str = (
        "llm_pass_rate is an alias of fact_coverage_pass_rate "
        "(must_contain substring), not an LLM-as-judge score"
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    corpus: str = ""
    timestamp: float = 0.0
    embedding_model: str = ""
    total_questions: int = 0
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    grounding_results: list[GroundingResult] = field(default_factory=list)
    fact_coverage_results: list[FactCoverageResult] = field(default_factory=list)
    # Deprecated alias field — populated for back-compat JSON readers.
    llm_judge_results: list[FactCoverageResult] = field(default_factory=list)
    no_context_results: list[NoContextResult] = field(default_factory=list)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    grounding_pass_rate: float = 0.0
    fact_coverage_pass_rate: float = 0.0
    # Alias of fact_coverage_pass_rate — NOT an LLM judge score.
    llm_pass_rate: float = 0.0
    no_context_pass_rate: float = 0.0
    portability_test: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    metadata: EvalMetadata = field(default_factory=EvalMetadata)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": EVAL_SCHEMA_VERSION,
            "corpus": self.corpus,
            "timestamp": self.timestamp,
            "embedding_model": self.embedding_model,
            "total_questions": self.total_questions,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "mrr": self.mrr,
            "grounding_pass_rate": self.grounding_pass_rate,
            "fact_coverage_pass_rate": self.fact_coverage_pass_rate,
            "llm_pass_rate": self.llm_pass_rate,  # alias; see metadata disclaimer
            "no_context_pass_rate": self.no_context_pass_rate,
            "portability_test": self.portability_test,
            "timing": self.timing,
            "metadata": self.metadata.as_dict(),
            "retrieval": [
                {
                    "qid": r.question_id,
                    "query": r.query,
                    "expected": r.expected_source,
                    "top": r.top_sources[:5],
                    "scores": r.top_scores[:5],
                    "rr": r.reciprocal_rank,
                    "hit_at_k": r.hit_at_k,
                }
                for r in self.retrieval_results
            ],
            "grounding": [
                {
                    "qid": r.question_id,
                    "grounding_overlap": r.grounding_overlap,
                    "fact_coverage": r.fact_coverage,
                    "missing_facts": r.missing_facts,
                    "grounded": r.grounded,
                    "facts_covered": r.facts_covered,
                }
                for r in self.grounding_results
            ],
            "fact_coverage": [
                {
                    "qid": r.question_id,
                    "passed": r.passed,
                    "missing": r.missing,
                    "method": r.method,
                }
                for r in self.fact_coverage_results
            ],
            # Legacy key — same rows as fact_coverage.
            "llm": [
                {
                    "qid": r.question_id,
                    "passed": r.passed,
                    "missing": r.missing,
                    "method": r.method,
                    "note": "legacy key; use fact_coverage",
                }
                for r in self.fact_coverage_results
            ],
            "no_context": [
                {
                    "qid": r.question_id,
                    "refused": r.refused,
                    "method": r.method,
                }
                for r in self.no_context_results
            ],
            "cases": self._per_case_rows(),
        }

    def _per_case_rows(self) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for r in self.retrieval_results:
            by_id.setdefault(r.question_id, {"qid": r.question_id, "query": r.query})
            by_id[r.question_id]["retrieval"] = {
                "rr": r.reciprocal_rank,
                "hit_at_k": r.hit_at_k,
                "top": r.top_sources[:5],
            }
        for r in self.grounding_results:
            by_id.setdefault(r.question_id, {"qid": r.question_id, "query": r.query})
            by_id[r.question_id]["grounding"] = {
                "overlap": r.grounding_overlap,
                "grounded": r.grounded,
                "facts_covered": r.facts_covered,
            }
        for r in self.fact_coverage_results:
            by_id.setdefault(r.question_id, {"qid": r.question_id, "query": r.query})
            by_id[r.question_id]["fact_coverage"] = {
                "passed": r.passed,
                "missing": r.missing,
            }
        for r in self.no_context_results:
            by_id.setdefault(r.question_id, {"qid": r.question_id, "query": r.query})
            by_id[r.question_id]["no_context"] = {"refused": r.refused}
        return list(by_id.values())


def load_qa_entries(path: str | Path) -> list[QAEntry]:
    """Load a JSON list of QAEntry-shaped objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("qa set must be a JSON list")
    fields = set(QAEntry.__dataclass_fields__)
    out: list[QAEntry] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise TypeError("each qa entry must be an object")
        kwargs = {k: v for k, v in item.items() if k in fields}
        out.append(QAEntry(**kwargs))
    return out


def _content_tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) >= 2 and t not in _STOPWORDS
    }


def _overlap_ratio(needle: set[str], haystack: set[str]) -> float:
    if not needle:
        return 0.0
    return len(needle & haystack) / max(len(needle), 1)


def looks_like_unknown(answer: str) -> bool:
    low = (answer or "").strip().lower()
    if not low:
        return False
    return any(m in low for m in _UNKNOWN_MARKERS)


# ── Retrieval metrics ───────────────────────────────────────────────────

def evaluate_retrieval(
    rag_query: RagSearchEngine,
    qa: list[QAEntry],
    ks: Sequence[int] = (1, 3, 5),
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    """For each question, search and record whether expected source is in top-k."""
    ks_list = list(ks)
    results: list[RetrievalResult] = []
    recall = {k: 0 for k in ks_list}
    mrr_sum = 0.0
    max_k = max(ks_list) if ks_list else 5

    for entry in qa:
        hits = rag_query.search(entry.query, top_k=max_k)
        top_sources = [str(h.get("source", "")) for h in hits]
        top_scores = [float(h.get("score", 0.0)) for h in hits]
        rr = 0.0
        needle = entry.expected_source_contains or ""
        if needle:
            for rank, src in enumerate(top_sources, 1):
                if needle in src:
                    rr = 1.0 / rank
                    break
        hit_at_max = bool(needle) and any(needle in s for s in top_sources[:max_k])
        results.append(RetrievalResult(
            question_id=entry.id,
            query=entry.query,
            expected_source=needle,
            top_sources=top_sources,
            top_scores=top_scores,
            reciprocal_rank=rr,
            hit_at_k=hit_at_max,
        ))
        mrr_sum += rr
        for k in ks_list:
            if needle and any(needle in s for s in top_sources[:k]):
                recall[k] += 1

    n = max(1, len(qa))
    recall_avg = {k: round(recall[k] / n, 3) for k in ks_list}
    mrr = round(mrr_sum / n, 3)
    return results, {"recall_at_k": recall_avg, "mrr": mrr}


# ── Answer grounding / fact coverage (deterministic) ────────────────────

def evaluate_answer_grounding(
    rag_query: RagSearchEngine,
    qa: list[QAEntry],
    answers: Mapping[str, str],
    *,
    top_k: int = 5,
    grounding_threshold: float = 0.15,
) -> list[GroundingResult]:
    """Score model answers against retrieved context (lexical) + must_contain."""
    out: list[GroundingResult] = []
    for entry in qa:
        hits = rag_query.search(entry.query, top_k=top_k)
        ctx = rag_query.format_context(hits, max_chars=5000)
        ans = answers.get(entry.id, "")
        ctx_toks = _content_tokens(ctx)
        ans_toks = _content_tokens(ans)
        overlap = _overlap_ratio(ans_toks, ctx_toks)
        missing = [f for f in entry.must_contain if f.lower() not in ans.lower()]
        covered = 1.0 - (len(missing) / max(len(entry.must_contain), 1))
        if not entry.must_contain:
            covered = 1.0 if ans.strip() else 0.0
        out.append(GroundingResult(
            question_id=entry.id,
            query=entry.query,
            answer=ans,
            context=ctx,
            grounding_overlap=round(overlap, 4),
            fact_coverage=round(covered, 4),
            missing_facts=missing,
            grounded=overlap >= grounding_threshold and bool(ans_toks),
            facts_covered=not missing,
        ))
    return out


def evaluate_fact_coverage(
    answers: Mapping[str, str],
    qa: list[QAEntry],
) -> list[FactCoverageResult]:
    """Substring must_contain check — formerly mislabelled as LLM-as-judge."""
    out: list[FactCoverageResult] = []
    for entry in qa:
        ans = answers.get(entry.id, "")
        missing = [f for f in entry.must_contain if f.lower() not in ans.lower()]
        out.append(FactCoverageResult(
            question_id=entry.id,
            query=entry.query,
            answer=ans,
            must_contain=list(entry.must_contain),
            missing=missing,
            passed=not missing,
            method="must_contain_substring",
        ))
    return out


def evaluate_no_context_unknown(
    qa: list[QAEntry],
    answer_fn: AnswerFn,
) -> list[NoContextResult]:
    """With empty context, answers must refuse / claim unknown."""
    out: list[NoContextResult] = []
    for entry in qa:
        ans = answer_fn(entry.query, "")
        out.append(NoContextResult(
            question_id=entry.id,
            query=entry.query,
            answer=ans,
            refused=looks_like_unknown(ans),
            method="unknown_phrase_match",
        ))
    return out


# ── Model answer helper (optional) ──────────────────────────────────────

def collect_rag_answers(
    mgr: Any,
    rag_query: RagSearchEngine,
    qa: list[QAEntry],
    *,
    top_k: int = 5,
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> dict[str, str]:
    """Ask a chat manager to answer each question using retrieved context only."""
    answers: dict[str, str] = {}
    for entry in qa:
        hits = rag_query.search(entry.query, top_k=top_k)
        ctx = rag_query.format_context(hits, max_chars=5000)
        prompt = (
            "Use ONLY the context below. If the answer is not in the context, "
            f'reply exactly: "{UNKNOWN_REPLY}" Otherwise answer concisely.\n\n'
            f"CONTEXT:\n{ctx}\n\nQUESTION: {entry.query}"
        )
        ans = mgr.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the provided context. "
                        "If the answer isn't in the context, say you don't know."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        ).strip()
        answers[entry.id] = ans
    return answers


def llm_as_judge(
    mgr: Any,
    rag_query: RagSearchEngine,
    qa: list[QAEntry],
    top_k: int = 5,
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> list[FactCoverageResult]:
    """Back-compat wrapper: retrieve + answer + must_contain check.

    Despite the historical name, this does **not** invoke an LLM judge —
    it checks that required fact strings appear in the model answer.
    Prefer ``collect_rag_answers`` + ``evaluate_fact_coverage``.
    """
    answers = collect_rag_answers(
        mgr, rag_query, qa,
        top_k=top_k, max_tokens=max_tokens, temperature=temperature,
    )
    return evaluate_fact_coverage(answers, qa)


# ── Portability round-trip ─────────────────────────────────────────────

def portability_roundtrip(
    corpus: PortableRAG,
    *,
    probe_query: str = "portability probe query",
) -> dict[str, Any]:
    """Build corpus → tar → untar → load → compare top-1 + vectors.

    ``probe_query`` defaults to a generic string (not domain-specific).
    Status is explicit: skipped / ok / failed — never an empty success.
    """
    from finetune_studio.data.rag_portable import PortableRAG as PR

    if not corpus.exists():
        return {
            "status": "failed",
            "ok": False,
            "error": "no corpus built",
            "method": "tar_roundtrip",
        }

    baseline_q = corpus.load()
    baseline_hits = baseline_q.search(probe_query, top_k=1)
    if not baseline_hits:
        return {
            "status": "failed",
            "ok": False,
            "error": "empty search results on baseline",
            "method": "tar_roundtrip",
        }
    baseline_top1 = baseline_hits[0]["source"]

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "corpus.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(corpus.dir), arcname="corpus")
        fresh_dir = Path(tmp) / "fresh_copy"
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(fresh_dir)
        src_inner = fresh_dir / "corpus"
        if not src_inner.exists():
            src_inner = fresh_dir
        fresh = PR(src_inner)
        if not fresh.exists():
            return {
                "status": "failed",
                "ok": False,
                "error": "no corpus after untar",
                "method": "tar_roundtrip",
            }
        fresh_q = fresh.load()
        fresh_hits = fresh_q.search(probe_query, top_k=1)
        if not fresh_hits:
            return {
                "status": "failed",
                "ok": False,
                "error": "empty search results after untar",
                "method": "tar_roundtrip",
            }
        fresh_top1 = fresh_hits[0]["source"]
        ok_top = baseline_top1 == fresh_top1
        base_vec = np.load(corpus.dir / "vectors.npy")
        fresh_vec = np.load(src_inner / "vectors.npy")
        vec_match = (
            base_vec.shape == fresh_vec.shape
            and bool(np.allclose(base_vec, fresh_vec, atol=1e-6))
        )
        ok = bool(ok_top and vec_match)
        return {
            "status": "ok" if ok else "failed",
            "ok": ok,
            "baseline_top1": baseline_top1,
            "fresh_top1": fresh_top1,
            "vectors_identical": bool(vec_match),
            "shape": list(base_vec.shape),
            "size_bytes_corpus": sum(
                p.stat().st_size for p in corpus.dir.rglob("*") if p.is_file()
            ),
            "method": "tar_roundtrip",
            "probe_query": probe_query,
        }


# ── Results writer ─────────────────────────────────────────────────────

def write_results(corpus_dir: Path, report: EvalReport) -> Path:
    tests_dir = corpus_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = tests_dir / stamp
    run_dir.mkdir(exist_ok=True)
    out = run_dir / "results.json"
    out.write_text(
        json.dumps(report.to_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    history_path = tests_dir / "history.json"
    history: dict[str, Any] = {"runs": []}
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            history = {"runs": []}
    history.setdefault("runs", []).append({"run": stamp, **report.to_json()})
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


# ── First-class entrypoint ──────────────────────────────────────────────

def run_rag_evaluation(
    *,
    rag_query: RagSearchEngine,
    qa: list[QAEntry],
    corpus_name: str = "",
    corpus_path: str = "",
    embedding_model: str = "",
    qa_set_path: str = "",
    ks: Sequence[int] = (1, 3, 5),
    top_k_answer: int = 5,
    answers: Mapping[str, str] | None = None,
    answer_fn: AnswerFn | None = None,
    mgr: Any | None = None,
    portable_corpus: PortableRAG | None = None,
    run_portability: bool = False,
    run_no_context: bool = True,
    write_to: Path | None = None,
) -> EvalReport:
    """Evaluate retrieval + optional grounding / unknown / portability.

    Parameters
    ----------
    answers
        Precomputed id→answer map (preferred for unit tests).
    answer_fn
        ``(query, context) -> answer`` used for no-context checks and, when
        ``answers`` is omitted, for in-context answers via retrieved context.
    mgr
        Chat manager with ``.chat(messages, ...)`` — used only when neither
        ``answers`` nor a full ``answer_fn`` path supplies in-context answers.
    run_portability
        When True and ``portable_corpus`` is set, run tar round-trip.
        When False, ``portability_test.status`` is ``"skipped"`` (not success).
    """
    t0 = time.time()
    timing: dict[str, float] = {}

    t_ret = time.time()
    ret_results, ret_metrics = evaluate_retrieval(rag_query, qa, ks=ks)
    timing["retrieve"] = round(time.time() - t_ret, 4)

    resolved_answers: dict[str, str] = dict(answers or {})

    if not resolved_answers and answer_fn is not None:
        for entry in qa:
            hits = rag_query.search(entry.query, top_k=top_k_answer)
            ctx = rag_query.format_context(hits, max_chars=5000)
            resolved_answers[entry.id] = answer_fn(entry.query, ctx)

    if not resolved_answers and mgr is not None:
        resolved_answers = collect_rag_answers(
            mgr, rag_query, qa, top_k=top_k_answer,
        )

    grounding_results: list[GroundingResult] = []
    fact_results: list[FactCoverageResult] = []
    if resolved_answers:
        t_g = time.time()
        grounding_results = evaluate_answer_grounding(
            rag_query, qa, resolved_answers, top_k=top_k_answer,
        )
        fact_results = evaluate_fact_coverage(resolved_answers, qa)
        timing["grounding"] = round(time.time() - t_g, 4)

    no_ctx: list[NoContextResult] = []
    if run_no_context:
        t_nc = time.time()
        if answer_fn is not None:
            no_ctx = evaluate_no_context_unknown(qa, answer_fn)
        elif mgr is not None:
            def _mgr_fn(query: str, context: str) -> str:
                prompt = (
                    "Use ONLY the context below. If the answer is not in the "
                    f'context, reply exactly: "{UNKNOWN_REPLY}"\n\n'
                    f"CONTEXT:\n{context}\n\nQUESTION: {query}"
                )
                return mgr.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Answer using only the provided context. "
                                "If empty or insufficient, say you don't know."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=80,
                    temperature=0.0,
                    top_p=0.9,
                ).strip()

            no_ctx = evaluate_no_context_unknown(qa, _mgr_fn)
        timing["no_context"] = round(time.time() - t_nc, 4)

    if run_portability and portable_corpus is not None:
        t_p = time.time()
        portability = portability_roundtrip(portable_corpus)
        timing["portability"] = round(time.time() - t_p, 4)
    elif run_portability and portable_corpus is None:
        portability = {
            "status": "failed",
            "ok": False,
            "error": "run_portability=True but portable_corpus is None",
            "method": "tar_roundtrip",
        }
    else:
        portability = {
            "status": "skipped",
            "ok": None,
            "method": "tar_roundtrip",
            "note": "portability not requested for this run",
        }

    n = max(1, len(qa))
    g_pass = (
        round(sum(1 for r in grounding_results if r.grounded) / n, 3)
        if grounding_results else 0.0
    )
    f_pass = (
        round(sum(1 for r in fact_results if r.passed) / n, 3)
        if fact_results else 0.0
    )
    nc_pass = (
        round(sum(1 for r in no_ctx if r.refused) / n, 3)
        if no_ctx else 0.0
    )

    meta = EvalMetadata(
        corpus=corpus_name,
        corpus_path=corpus_path,
        embedding_model=embedding_model,
        qa_set_path=qa_set_path,
        total_questions=len(qa),
        ks=list(ks),
        top_k_answer=top_k_answer,
        ran_retrieval=True,
        ran_grounding=bool(grounding_results),
        ran_fact_coverage=bool(fact_results),
        ran_no_context=bool(no_ctx),
        ran_portability=portability.get("status") not in (None, "skipped"),
    )
    timing["total"] = round(time.time() - t0, 4)

    report = EvalReport(
        corpus=corpus_name,
        timestamp=time.time(),
        embedding_model=embedding_model,
        total_questions=len(qa),
        retrieval_results=ret_results,
        grounding_results=grounding_results,
        fact_coverage_results=fact_results,
        llm_judge_results=fact_results,
        no_context_results=no_ctx,
        recall_at_k=ret_metrics["recall_at_k"],
        mrr=ret_metrics["mrr"],
        grounding_pass_rate=g_pass,
        fact_coverage_pass_rate=f_pass,
        llm_pass_rate=f_pass,
        no_context_pass_rate=nc_pass,
        portability_test=portability,
        timing=timing,
        metadata=meta,
    )
    if write_to is not None:
        write_results(Path(write_to), report)
    return report


def run_eval_on_corpus(
    corpus: str,
    qa_set_path: str,
    *,
    run_portability: bool = True,
    use_llm: bool = True,
    ks: Sequence[int] = (1, 3, 5),
) -> EvalReport:
    """Stand-alone runner used by the portable-RAG CLI.

    Loads the corpus + QA set, optionally a chat manager for answers, and
    writes ``tests/<stamp>/results.json``.
    """
    from finetune_studio.data.rag_portable import PortableRAG as PR

    qa = load_qa_entries(qa_set_path)
    rag = PR(corpus)
    if not rag.exists():
        raise FileNotFoundError(f"corpus not built: {corpus}")
    q = rag.load()
    embed_name = ""
    try:
        embed_name = q.manifest.embedding_model.name
    except Exception:  # noqa: BLE001
        embed_name = ""

    mgr = None
    if use_llm:
        try:
            from finetune_studio.models.manager import get_manager
            mgr = get_manager()
            if mgr.active() is None:
                mgr.load("local-default")
        except Exception:  # noqa: BLE001
            mgr = None

    report = run_rag_evaluation(
        rag_query=q,
        qa=qa,
        corpus_name=Path(corpus).name,
        corpus_path=str(Path(corpus).resolve()),
        embedding_model=embed_name,
        qa_set_path=str(Path(qa_set_path).resolve()),
        ks=ks,
        mgr=mgr,
        portable_corpus=rag if run_portability else None,
        run_portability=run_portability,
        run_no_context=bool(mgr),
        write_to=Path(corpus),
    )
    return report
