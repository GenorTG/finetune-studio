"""Unit + route tests for retrieval-grounded QA (testing.rag_suite)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.rag_eval import UNKNOWN_REPLY
from finetune_studio.testing.rag_suite import (
    RagCaseResult,
    build_grounded_messages,
    compute_retrieval_metrics,
    hit_matches_source,
    provenance_from_hit,
    resolve_corpus_path,
    run_rag_suite,
    run_rag_suite_evaluation,
)
from finetune_studio.testing.suite import BenchmarkCase, CaseResult
from finetune_studio.webui import app as app_module
from finetune_studio.webui.app import app
from tests.testing_run_support import wait_run_finished


class FakeRag:
    """Minimal PortableRAGQuery stand-in."""

    def __init__(self, hits_by_query: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.hits_by_query = hits_by_query or {}
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.search_calls.append((query, top_k))
        hits = list(self.hits_by_query.get(query, []))
        return hits[:top_k]

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        parts: list[str] = []
        total = 0
        for r in results:
            block = f"[{r.get('rank', 0)}] {r.get('text', '')}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n---\n".join(parts)


class FakeEngine:
    """Inference stand-in that echoes context or returns UNKNOWN."""

    def __init__(self, answer: str = "Ada Smit", *, model_path: str = "/fake/model") -> None:
        self.answer = answer
        self.model = object()
        self.model_path = model_path
        self.is_gguf = False
        self.generate_calls: list[list[dict]] = []

    def load(self, path: str, **_kwargs: Any) -> None:
        self.model = object()
        self.model_path = path

    def unload(self) -> None:
        self.model = None
        self.model_path = None

    def generate(self, messages: list[dict], **_kwargs: Any) -> str:
        self.generate_calls.append(list(messages))
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        if "CONTEXT:\n" in user and self.answer.lower() in user.lower():
            return self.answer
        if not self.answer:
            return UNKNOWN_REPLY
        return self.answer


def test_build_grounded_messages_mentions_unknown_fallback() -> None:
    msgs = build_grounded_messages("Who owns RK-04?", "chunk about RK-04")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert UNKNOWN_REPLY in msgs[1]["content"]
    assert "Who owns RK-04?" in msgs[1]["content"]
    assert "chunk about RK-04" in msgs[1]["content"]


def test_build_grounded_messages_gives_generic_table_guidance() -> None:
    system = build_grounded_messages("What is the total actual_hours?", "actual_hours | variance_hours")[0]["content"]
    assert "use the row and column named by the question" in system
    assert "variance" not in system.lower()      # no corpus-specific column names in the shared prompt


def test_hit_matches_source_and_chunk() -> None:
    hit = {
        "document_id": "doc-rk04",
        "source": "/docs/rk04.md",
        "filename": "rk04.md",
        "chunk_index": 2,
        "chunk_id": "c2",
        "rank": 1,
        "score": 0.9,
    }
    assert hit_matches_source(hit, "doc-rk04", 2) is True
    assert hit_matches_source(hit, "doc-rk04", 9) is False
    assert hit_matches_source(hit, "other", 2) is False
    assert hit_matches_source(hit, "rk04.md", 0) is True


def test_provenance_from_hit_normalizes_scores() -> None:
    p = provenance_from_hit(
        {
            "document_id": "d1",
            "rrf_score": 0.42,
            "filename": "a.txt",
            "rank": 3,
            "chunk_index": 1,
        }
    )
    assert p["document_id"] == "d1"
    assert p["score"] == 0.42
    assert p["rank"] == 3
    assert p["chunk_index"] == 1


def test_compute_retrieval_metrics() -> None:
    def _wrap(name: str, source_id: str, hit: bool) -> RagCaseResult:
        return RagCaseResult(
            case_result=CaseResult(
                case_name=name,
                category="x",
                question="q",
                correct_answer="a",
                model_answer="a",
                source_id=source_id,
            ),
            retrieval_hit=hit,
        )

    metrics = compute_retrieval_metrics(
        [
            _wrap("a", "s1", True),
            _wrap("b", "s2", False),
            _wrap("c", "", True),  # ignored — no source_id
        ]
    )
    assert metrics["cases_with_source_id"] == 2
    assert metrics["retrieval_hits"] == 1
    assert metrics["retrieval_misses"] == 1
    assert metrics["recall_at_k"] == 0.5


def test_run_rag_suite_preserves_transcript_context_hits() -> None:
    question = "Who owns RK-04?"
    hits = [
        {
            "rank": 1,
            "document_id": "doc-rk04",
            "source": "rk04.md",
            "filename": "rk04.md",
            "chunk_index": 0,
            "chunk_id": "c0",
            "score": 1.0,
            "text": "RK-04 is owned by Ada Smit.",
        }
    ]
    rag = FakeRag({question: hits})
    engine = FakeEngine("Ada Smit", model_path="/models/merged")
    cases = [
        BenchmarkCase(
            name="rk04_owner",
            question=question,
            correct_answer="Ada Smit",
            keywords=["Ada Smit"],
            source_id="doc-rk04",
        )
    ]
    results = run_rag_suite(engine, rag, cases, top_k=3)
    assert len(results) == 1
    r = results[0]
    assert r.retrieval_hit is True
    assert r.context_text
    assert "Ada Smit" in r.context_text
    assert r.retrieval_hits[0]["document_id"] == "doc-rk04"
    assert r.case_result.model_answer == "Ada Smit"
    assert len(r.case_result.transcript) == 3
    assert r.case_result.transcript[0]["role"] == "system"
    assert r.case_result.transcript[-1]["role"] == "assistant"
    assert rag.search_calls == [(question, 3)]


def test_a_source_missed_at_top_k_is_reported_by_its_wide_rank_and_never_pulled_into_the_context() -> None:
    class ExpandingRag(FakeRag):
        def search(self, query: str, top_k: int = 5) -> list[dict]:
            self.search_calls.append((query, top_k))
            if top_k >= 20:
                return [{"document_id": "other", "chunk_index": 0, "text": "noise"},
                        {"document_id": "source-b", "chunk_index": 0, "text": "answer"}]
            return []

    rag = ExpandingRag()
    engine = FakeEngine(answer="answer")
    case = BenchmarkCase(
        name="expand", category="qa", question="question", correct_answer="answer",
        source_id="source-b",
    )
    results = run_rag_suite(engine, rag, [case], top_k=5)
    r = results[0]
    assert r.retrieval_hit is False and r.retrieval_hits == [] and r.context_text == ""   # the reader saw its real top-5
    assert r.gold_rank_wide == 2 and r.as_api_dict()["gold_rank_wide"] == 2
    assert rag.search_calls == [("question", 5), ("question", 40)]
    metrics = compute_retrieval_metrics(results)
    assert metrics["retrieval_misses"] == 1 and metrics["misses_found_wider"] == 1 and metrics["wide_search_top_k"] == 40


def test_no_wide_search_when_the_source_was_retrieved_or_the_case_names_none() -> None:
    hit = [{"document_id": "doc-a", "chunk_index": 0, "text": "x"}]
    rag = FakeRag({"q1": hit, "q2": hit})
    engine = FakeEngine(answer="x")
    cases = [BenchmarkCase(name="a", question="q1", correct_answer="x", source_id="doc-a"),
             BenchmarkCase(name="b", question="q2", correct_answer="x")]
    results = run_rag_suite(engine, rag, cases, top_k=3)
    assert [r.gold_rank_wide for r in results] == [None, None]
    assert rag.search_calls == [("q1", 3), ("q2", 3)]


def test_run_rag_suite_evaluation_scores_and_metrics(tmp_path: Path) -> None:
    suite = [
        {
            "name": "hit_case",
            "question": "Who owns RK-04?",
            "correct_answer": "Ada Smit",
            "keywords": ["Ada Smit"],
            "source_id": "doc-rk04",
        },
        {
            "name": "miss_case",
            "question": "What is CR-77?",
            "correct_answer": "cold chain exception",
            "keywords": ["cold"],
            "source_id": "doc-cr77",
        },
    ]
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    rag = FakeRag(
        {
            "Who owns RK-04?": [
                {
                    "rank": 1,
                    "document_id": "doc-rk04",
                    "source": "rk04.md",
                    "filename": "rk04.md",
                    "chunk_index": 0,
                    "chunk_id": "c0",
                    "score": 1.0,
                    "text": "RK-04 is owned by Ada Smit.",
                }
            ],
            "What is CR-77?": [
                {
                    "rank": 1,
                    "document_id": "unrelated",
                    "source": "other.md",
                    "filename": "other.md",
                    "chunk_index": 0,
                    "chunk_id": "x",
                    "score": 0.1,
                    "text": "Unrelated warehouse note.",
                }
            ],
        }
    )

    def _gen(messages: list[dict], **_kwargs: Any) -> str:
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "Ada Smit" in user:
            return "Ada Smit"
        return UNKNOWN_REPLY

    engine = MagicMock()
    engine.model_path = "/models/q8"
    engine.generate.side_effect = _gen

    report = run_rag_suite_evaluation(
        engine,
        suite_path=str(suite_path),
        corpus_path=str(tmp_path / "corpus"),
        project_id="p1",
        top_k=2,
        rag_query=rag,
    )
    payload = report.as_api_dict()
    assert payload["model_path"] == "/models/q8"
    assert payload["corpus_path"] == str(tmp_path / "corpus")
    assert payload["top_k"] == 2
    assert payload["unknown_reply"] == UNKNOWN_REPLY
    assert payload["retrieval"]["cases_with_source_id"] == 2
    assert payload["retrieval"]["retrieval_hits"] == 1
    assert payload["retrieval"]["recall_at_k"] == 0.5
    assert payload["scores"]["total"] == 2
    assert payload["scores"]["judged"] == 0 and payload["scores"]["awaiting"] == 2  # recorded, not judged
    assert payload["scores"]["pass_rate"] is None
    assert all(r["verdict"] == "" for r in payload["results"])
    hit_row = next(r for r in payload["results"] if r["name"] == "hit_case")
    assert hit_row["retrieval_hit"] is True
    assert hit_row["context_text"]
    assert hit_row["transcript"]
    miss_row = next(r for r in payload["results"] if r["name"] == "miss_case")
    assert miss_row["retrieval_hit"] is False


def test_resolve_corpus_path_explicit_and_default(tmp_path: Path) -> None:
    explicit = tmp_path / "custom-corpus"
    assert resolve_corpus_path("pid", str(explicit)) == explicit
    default = resolve_corpus_path("abc123", "")
    assert default.name == "abc123"
    assert "rag_corpora" in str(default)


@pytest.fixture
def client_and_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return TestClient(app), db_path


def test_run_rag_suite_route_starts_a_background_run_that_saves_transcripts_and_no_verdict(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = client.post(
        "/api/projects",
        json={"name": f"rag-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    suite = [
        {
            "name": "rk04_owner",
            "question": "Who owns RK-04?",
            "correct_answer": "Ada Smit",
            "keywords": ["Ada Smit"],
            "source_id": "doc-rk04",
        }
    ]
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("finetune_studio.webui.testing_models.local_model_missing", lambda _p: False)  # /override/model is a fake path
    fake = FakeEngine("Ada Smit", model_path="/override/model")
    monkeypatch.setattr(app_module, "inference_engine", fake)

    rag = FakeRag(
        {
            "Who owns RK-04?": [
                {
                    "rank": 1,
                    "document_id": "doc-rk04",
                    "source": "rk04.md",
                    "filename": "rk04.md",
                    "chunk_index": 0,
                    "chunk_id": "c0",
                    "score": 1.0,
                    "text": "RK-04 is owned by Ada Smit.",
                }
            ]
        }
    )
    monkeypatch.setattr(
        "finetune_studio.testing.rag_suite.load_portable_rag_query",
        lambda _path: rag,
    )

    resp = client.post(
        "/api/testing/run-rag-suite",
        json={
            "suite_path": str(suite_path),
            "project_id": pid,
            "model_path": "/override/model",
            "corpus_path": str(corpus),
            "top_k": 2,
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 202, resp.text
    bid = resp.json()["benchmark_id"]
    assert resp.json()["benchmark"]["status"] == "running" and resp.json()["benchmark"]["kind"] == "rag"
    done = wait_run_finished(bid)

    assert done["status"] == "done" and done["model_path"] == "/override/model"
    assert done["scores"]["corpus_path"] == str(corpus)
    assert done["scores"]["retrieval"]["retrieval_hits"] == 1
    assert done["scores"]["retrieval"]["recall_at_k"] == 1.0
    assert done["scores"]["total"] == 1 and done["scores"]["awaiting"] == 1 and done["scores"]["pass_rate"] is None
    assert done["config"]["rag"]["top_k"] == 2
    cases = db.list_cases(bid)
    assert len(cases) == 1
    assert cases[0]["source_id"] == "doc-rk04"
    assert cases[0]["transcript"]
    assert cases[0]["verdict"] == "" and cases[0]["model_answer"] == "Ada Smit"
    assert cases[0]["judge_input"]["retrieval_hit"] is True
    assert rag.search_calls

    listed = client.get(f"/api/testing/projects/{pid}/runs/{bid}/cases").json()
    assert listed[0]["question"] == "Who owns RK-04?" and listed[0]["judge_input"]["retrieval_hit"] is True
    assert "context_text" not in listed[0]["judge_input"]  # heavy fields only on the single-case fetch
    full = client.get(f"/api/testing/projects/{pid}/runs/{bid}/cases/{listed[0]['id']}").json()
    assert full["judge_input"]["context_text"] and full["judge_input"]["retrieval_hits"][0]["document_id"] == "doc-rk04"
    assert full["transcript"]


def test_run_rag_suite_route_requires_suite_path(
    client_and_db: tuple[TestClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = client_and_db
    fake = FakeEngine()
    monkeypatch.setattr(app_module, "inference_engine", fake)
    monkeypatch.setattr(
        "finetune_studio.webui.routes.testing.inference_engine",
        fake,
    )
    resp = client.post("/api/testing/run-rag-suite", json={})
    assert resp.status_code == 400
    assert "suite_path" in resp.json()["error"]


def test_run_rag_suite_route_missing_corpus_fails_the_run_with_the_reason(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = client.post("/api/projects", json={"name": f"rag-{uuid.uuid4().hex[:6]}", "base_model": "x/test"}).json()["id"]
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps([{"name": "q1", "question": "Q?", "correct_answer": "A", "keywords": ["A"]}]), encoding="utf-8",
    )
    monkeypatch.setattr("finetune_studio.webui.testing_models.local_model_missing", lambda _p: False)  # /override/model is a fake path
    monkeypatch.setattr(app_module, "inference_engine", FakeEngine(model_path="/override/model"))
    resp = client.post(
        "/api/testing/run-rag-suite",
        json={
            "suite_path": str(suite_path),
            "project_id": pid,
            "model_path": "/override/model",
            "corpus_path": str(tmp_path / "no-such-corpus"),
        },
    )
    assert resp.status_code == 202, resp.text
    done = wait_run_finished(resp.json()["benchmark_id"])
    assert done["status"] == "failed" and "RAG corpus not found" in done["error"]
    assert db.list_cases(done["id"]) == []


def test_run_rag_suite_route_needs_a_project(client_and_db: tuple[TestClient, Path], tmp_path: Path) -> None:
    client, _ = client_and_db
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps([{"name": "q1", "question": "Q?", "correct_answer": "A"}]), encoding="utf-8")
    resp = client.post("/api/testing/run-rag-suite", json={"suite_path": str(suite_path)})
    assert resp.status_code == 400 and "project_id" in resp.json()["error"]
