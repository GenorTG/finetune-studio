"""The default number of retrieved passages is 10 everywhere RAG chat / the RAG suite pick one themselves.

Why 10: on the Korvane quiz the same readers answered 84 -> 87 -> 89 of 102 at top-5 / 10 / 20, and ten chunks still fit a modern
window (tests/corpus/korvane/RESULTS.md §8). Values a user set stay as they were; only an unset one resolves to the new default.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from finetune_studio.smart_brain.constants import DEFAULT_TOP_K
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.query import BrainQuery
from finetune_studio.smart_brain.standalone_server import DEFAULT_CONFIG
from finetune_studio.testing import brain_suite
from finetune_studio.testing.suite import BenchmarkCase
from finetune_studio.webui.routes import chat_v2
from finetune_studio.webui.routes import brain as rag_routes
from finetune_studio.webui.testing_jobs import RunSpec, start_run_job
from tests.test_brain_suite import FakeEngine, FakeRag
from tests.test_testing_jobs import (
    _isolated_jobs,  # noqa: F401  (autouse fixture: fresh job registry + settings file)
)
from tests.testing_run_support import await_finished, install_engine, make_project_run

TEMPLATES = Path(brain_suite.__file__).resolve().parents[1] / "webui" / "templates"


def _default(fn: object, name: str = "top_k") -> object:
    return inspect.signature(fn).parameters[name].default


def test_the_default_is_ten_and_one_constant_feeds_the_python_defaults() -> None:
    assert DEFAULT_TOP_K == 10
    for fn in (BrainQuery.search, brain_suite.run_brain_case, brain_suite.run_brain_suite,
               brain_suite.run_brain_suite_evaluation, chat_v2._search_brain_index):
        assert _default(fn) == 10, fn
    assert brain_suite.BrainSuiteReport().top_k == 10
    assert rag_routes.SearchRequest(query="q").top_k == 10
    assert rag_routes.ChatRequest(messages=[]).top_k == 10


def test_export_and_standalone_defaults_follow() -> None:
    assert BrainExportConfig().top_k == 10
    assert DEFAULT_CONFIG["top_k"] == 10


def test_values_a_user_set_are_untouched() -> None:
    assert rag_routes.SearchRequest(query="q", top_k=3).top_k == 3
    assert rag_routes.ChatRequest(messages=[], top_k=20).top_k == 20
    assert BrainExportConfig(top_k=7).top_k == 7


def test_the_suite_asks_the_index_for_ten_unless_told_otherwise() -> None:
    hits = [{"rank": i + 1, "document_id": f"d{i}", "text": f"chunk {i}"} for i in range(15)]
    case = BenchmarkCase(name="c", category="qa", question="q", correct_answer="a")
    rag = FakeRag({"q": hits})
    (result,) = brain_suite.run_brain_suite(FakeEngine(answer="a"), rag, [case])
    assert rag.search_calls == [("q", 10)] and result.chunks_retrieved == 10

    rag = FakeRag({"q": hits})
    (result,) = brain_suite.run_brain_suite(FakeEngine(answer="a"), rag, [case], top_k=3)
    assert rag.search_calls == [("q", 3)] and result.chunks_retrieved == 3


@pytest.mark.asyncio
async def test_a_run_job_without_top_k_retrieves_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    pid, _rid = make_project_run()
    rag = FakeRag({"Who owns RK-04?": [{"rank": 1, "document_id": "doc-1", "text": "RK-04 is owned by Ada Smit."}]})
    monkeypatch.setattr("finetune_studio.testing.brain_suite.load_brain_query", lambda _p: rag)
    install_engine(monkeypatch, FakeEngine(lambda _q: "Ada Smit", model_path="/m/merged"))
    spec = RunSpec(project_id=pid, kind="brain", suite_name="rag · s.json", model_path="/m/merged",
                   cases=[BenchmarkCase(name="rk04", question="Who owns RK-04?", correct_answer="Ada Smit")],
                   brain={"corpus_path": "/corpus", "max_context_chars": 2000})
    row = await start_run_job(spec)
    await await_finished(row["id"])
    assert rag.search_calls == [("Who owns RK-04?", 10)]


def test_the_rag_page_defaults_match() -> None:
    html = (TEMPLATES / "brain.html").read_text(encoding="utf-8")
    assert re.search(r'id="mcp-topk"[^>]*value="10"', html)
    assert "|| 10," in html and "top_k: 10" in html
    assert not re.search(r"top_k:\s*5\b", html) and "|| 5," not in html


def test_project_testing_page_already_defaults_to_ten() -> None:
    html = (TEMPLATES / "project_testing.html").read_text(encoding="utf-8")
    assert re.search(r'id="t-brain-topk"[^>]*value="10"', html)
