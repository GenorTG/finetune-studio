"""Focused tests for first-class RAG evaluation with a fake corpus/engine."""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.data.rag_eval import (
    EVAL_SCHEMA_VERSION,
    UNKNOWN_REPLY,
    QAEntry,
    evaluate_no_context_unknown,
    evaluate_retrieval,
    looks_like_unknown,
    run_rag_evaluation,
)


class FakeRAGEngine:
    """Tiny in-memory search engine: keyword overlap over a fixed corpus."""

    def __init__(self, docs: list[dict[str, str]]) -> None:
        self.docs = docs

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_toks = set(query.lower().split())
        scored: list[tuple[float, dict[str, str]]] = []
        for doc in self.docs:
            text = doc["text"].lower()
            score = sum(1.0 for t in q_toks if t in text)
            scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        out: list[dict] = []
        for rank, (score, doc) in enumerate(scored[:top_k], start=1):
            out.append({
                "rank": rank,
                "score": float(score),
                "rrf_score": float(score),
                "text": doc["text"],
                "source": doc["source"],
                "filename": doc.get("filename", Path(doc["source"]).name),
                "chunk_id": doc.get("id", doc["source"]),
                "document_id": doc.get("id", "d1"),
                "chunk_index": 0,
            })
        return out

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        blocks: list[str] = []
        total = 0
        for r in results:
            block = f"[{r['rank']}] (source: {r['source']})\n{r['text']}"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n".join(blocks)


CORPUS = [
    {
        "id": "doc_helios",
        "source": "/corpus/helios_handbook.txt",
        "filename": "helios_handbook.txt",
        "text": (
            "Helios Industries CEO is Mira Chen. "
            "Aurora battery packs reach 400 Wh/kg."
        ),
    },
    {
        "id": "doc_other",
        "source": "/corpus/catering_menu.txt",
        "filename": "catering_menu.txt",
        "text": "Today's lunch special is tomato soup and grilled cheese.",
    },
]

QA = [
    QAEntry(
        id="q1",
        query="Who is the CEO of Helios Industries?",
        must_contain=["Mira Chen"],
        expected_source_contains="helios_handbook.txt",
    ),
    QAEntry(
        id="q2",
        query="What is the lunch special?",
        must_contain=["tomato soup"],
        expected_source_contains="catering_menu.txt",
    ),
]


def _oracle_answer(query: str, context: str) -> str:
    if not context.strip():
        return UNKNOWN_REPLY
    low = context.lower()
    if "mira chen" in low and "ceo" in query.lower():
        return "Mira Chen is the CEO according to helios_handbook.txt."
    if "tomato soup" in low:
        return "The lunch special is tomato soup and grilled cheese."
    return UNKNOWN_REPLY


class TestRetrievalMetrics:
    def test_recall_and_mrr(self) -> None:
        engine = FakeRAGEngine(CORPUS)
        results, metrics = evaluate_retrieval(engine, QA, ks=(1, 3, 5))
        assert len(results) == 2
        assert metrics["recall_at_k"][1] == 1.0
        assert metrics["mrr"] == 1.0
        assert all(r.hit_at_k for r in results)


class TestGroundingAndUnknown:
    def test_grounded_answers_and_fact_coverage(self) -> None:
        engine = FakeRAGEngine(CORPUS)
        report = run_rag_evaluation(
            rag_query=engine,
            qa=QA,
            corpus_name="fake",
            answers={
                "q1": "Mira Chen leads Helios Industries.",
                "q2": "tomato soup is the lunch special.",
            },
            run_no_context=False,
            run_portability=False,
        )
        assert report.metadata.schema_version == EVAL_SCHEMA_VERSION
        assert report.metadata.ran_grounding is True
        assert report.grounding_pass_rate == 1.0
        assert report.fact_coverage_pass_rate == 1.0
        # Honest alias — same value, labelled in metadata.
        assert report.llm_pass_rate == report.fact_coverage_pass_rate
        assert "not an LLM-as-judge" in report.metadata.legacy_llm_pass_rate_alias
        assert report.portability_test["status"] == "skipped"
        assert report.portability_test.get("ok") is None

    def test_ungrounded_answer_fails_grounding(self) -> None:
        engine = FakeRAGEngine(CORPUS)
        report = run_rag_evaluation(
            rag_query=engine,
            qa=[QA[0]],
            answers={"q1": "Quantum foam wobbles in eleven-dimensional calabi manifolds."},
            run_no_context=False,
            run_portability=False,
        )
        assert report.grounding_results[0].grounded is False
        assert report.fact_coverage_results[0].passed is False

    def test_no_context_unknown_behavior(self) -> None:
        results = evaluate_no_context_unknown(QA, _oracle_answer)
        assert all(r.refused for r in results)
        assert looks_like_unknown(UNKNOWN_REPLY)

    def test_run_rag_evaluation_with_answer_fn(self) -> None:
        engine = FakeRAGEngine(CORPUS)
        report = run_rag_evaluation(
            rag_query=engine,
            qa=QA,
            corpus_name="fake",
            embedding_model="none",
            answer_fn=_oracle_answer,
            run_portability=False,
            run_no_context=True,
        )
        assert report.recall_at_k[1] == 1.0
        assert report.mrr == 1.0
        assert report.no_context_pass_rate == 1.0
        assert report.metadata.ran_no_context is True
        assert report.metadata.ran_portability is False
        payload = report.to_json()
        assert "cases" in payload
        assert len(payload["cases"]) == 2
        # Legacy "llm" key exists but notes it is not a judge.
        assert payload["llm"][0]["method"] == "must_contain_substring"
        assert "legacy key" in payload["llm"][0]["note"]


class TestWriteResultsMetadata:
    def test_write_results_roundtrip(self, tmp_path: Path) -> None:

        engine = FakeRAGEngine(CORPUS)
        report = run_rag_evaluation(
            rag_query=engine,
            qa=QA[:1],
            corpus_name="fake",
            answers={"q1": "Mira Chen is CEO."},
            run_no_context=False,
            run_portability=False,
            write_to=tmp_path,
        )
        out = tmp_path / "tests"
        assert out.exists()
        runs = list(out.glob("*/results.json"))
        assert len(runs) == 1
        data = json.loads(runs[0].read_text(encoding="utf-8"))
        assert data["schema_version"] == EVAL_SCHEMA_VERSION
        assert data["metadata"]["ran_retrieval"] is True
        assert data["portability_test"]["status"] == "skipped"
        assert report.total_questions == 1
