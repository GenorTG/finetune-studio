"""Shared fakes for the run-then-judge tests: a scriptable inference engine, a fake judge and a job waiter.

Used by ``test_testing_jobs.py`` / ``test_testing_routes.py`` (and the route tests that start a test run).
The engine and the judge are the only things faked: the jobs, the DB rows and the routes are the real ones.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from finetune_studio import db
from finetune_studio.testing.judge import LoadedJudge
from finetune_studio.testing.suite import BenchmarkCase
from finetune_studio.webui import testing_jobs

TERMINAL = ("done", "failed", "cancelled")


class FakeEngine:
    """Stands in for ``InferenceEngine``: answers from a table (or a callable) and can call a hook per question."""

    def __init__(
        self,
        answers: dict[str, str] | Callable[[str], str] | None = None,
        *,
        model_path: str | None = "/fake/model",
        on_generate: Callable[[str], None] | None = None,
    ) -> None:
        self.answers = answers if answers is not None else (lambda q: f"answer to {q}")
        self.model: object | None = object() if model_path else None
        self.model_path = model_path
        self.is_gguf = False
        self.n_ctx = 4096
        self.on_generate = on_generate
        self.asked: list[str] = []
        self.load_calls: list[str] = []

    def load(self, path: str, **_kw: Any) -> None:
        self.load_calls.append(path)
        self.model, self.model_path = object(), path

    def unload(self) -> None:
        self.model, self.model_path = None, None

    def generate(self, messages: list[dict], **_kw: Any) -> str:
        question = messages[-1]["content"]
        self.asked.append(question)
        if self.on_generate is not None:
            self.on_generate(question)
        return self.answers(question) if callable(self.answers) else self.answers[question]


def install_engine(monkeypatch: pytest.MonkeyPatch, engine: FakeEngine | None = None) -> FakeEngine:
    """Make ``engine`` the app's inference engine (everything that reads it at call time sees the fake)."""
    engine = engine or FakeEngine()
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", engine)
    monkeypatch.setattr("finetune_studio.webui.routes.testing.inference_engine", engine)
    return engine


def cases_n(n: int, *, prefix: str = "Q") -> list[BenchmarkCase]:
    return [BenchmarkCase(name=f"c{i}", question=f"{prefix}{i}", correct_answer=f"key{i}", keywords=[f"key{i}"])
            for i in range(n)]


def judge_reply(verdict: str, reasoning: str = "") -> str:
    return json.dumps({"reasoning": reasoning or f"judged {verdict}", "verdict": verdict, "confidence": 0.9})


def install_judge(
    monkeypatch: pytest.MonkeyPatch,
    verdicts: dict[str, str] | None = None,
    *,
    concurrent: bool = False,
    model: str = "fake-judge",
    on_chat: Callable[[str], None] | None = None,
    reply: Callable[[str], str] | None = None,
) -> list[str]:
    """Replace ``testing_jobs.open_judge`` with a judge that answers by the question found in the prompt.

    ``verdicts`` maps a question to its verdict (default pass). Returns the list of questions the judge was asked.
    """
    asked: list[str] = []
    verdicts = verdicts or {}

    def chat(messages: list[dict]) -> str:
        prompt = messages[-1]["content"]
        question = prompt.split("QUESTION:\n", 1)[1].split("\n\n", 1)[0].strip()
        asked.append(question)
        if on_chat is not None:
            on_chat(question)
        return reply(question) if reply is not None else judge_reply(verdicts.get(question, "pass"))

    @contextmanager
    def fake_open(provider_id: str) -> Iterator[LoadedJudge]:
        yield LoadedJudge(chat=chat, provider_id=provider_id, model=model, label="Fake judge", concurrent=concurrent)

    monkeypatch.setattr("finetune_studio.webui.testing_jobs.open_judge", fake_open)
    return asked


def add_api_provider(provider_id: str = "api-judge") -> str:
    """An OpenAI-compatible (non-local) provider row, i.e. a judge that needs no GPU."""
    from finetune_studio.models.manager import get_manager

    get_manager().upsert_provider(id=provider_id, name="API judge", kind="openai_compat", model_id="judge-x",
                                  base_url="http://127.0.0.1:9/v1", api_key="k")
    return provider_id


def make_project_run() -> tuple[str, str]:
    """A project and one finished training run (owner of test runs)."""
    pid = db.create_project(name=f"p-{time.time_ns()}")["id"]
    return pid, db.create_run(pid, "train")["id"]


def saved_run(
    rid: str, answers: list[tuple[str, str, str]], *, status: str = "done", scoring: str = "judge",
) -> str:
    """A saved test run with ``(question, key, model answer)`` cases and no verdicts. Returns its id."""
    bench = db.create_benchmark(rid, "quiz", {}, status=status, kind="suite", scoring=scoring)
    for i, (q, key, ans) in enumerate(answers):
        db.create_case(bench["id"], rid, f"c{i}", "general", q, key, ans, [{"role": "user", "content": q}])
    return bench["id"]


def wait_until(predicate: Callable[[], bool], *, timeout: float = 15.0, what: str = "condition") -> None:
    """Poll from a synchronous test (the job runs on the app's own event loop thread)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def wait_run_finished(bid: str, *, judge: bool = False, timeout: float = 15.0) -> dict[str, Any]:
    """Block until the run (and, with ``judge``, its judge job) reached a terminal state and nothing is active."""
    def finished() -> bool:
        row = db.get_benchmark(bid) or {}
        if row.get("status") not in TERMINAL:
            return False
        if judge and row.get("judge_status") not in TERMINAL:
            return False
        return not testing_jobs.is_active(bid)

    wait_until(finished, timeout=timeout, what=f"run {bid} to finish")
    return db.get_benchmark(bid) or {}


async def await_finished(bid: str, *, judge: bool = False, timeout: float = 15.0) -> dict[str, Any]:
    """``wait_run_finished`` for tests that run on the event loop the jobs run on (never blocks the loop)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = db.get_benchmark(bid) or {}
        if row.get("status") in TERMINAL and (not judge or row.get("judge_status") in TERMINAL) \
                and not testing_jobs.is_active(bid):
            return row
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for run {bid}: {db.get_benchmark(bid)}")
