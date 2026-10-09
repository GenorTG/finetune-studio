"""Benchmarks route: public / built-in multiple-choice suites are exact-scored here; a project quiz is not (QABUG-006 / QABUG-010)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui.app import app


@pytest.fixture
def client_and_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    # The shared autouse fixture patches db.connection.settings separately;
    # keep this module's custom path aligned with the connection actually used.
    from finetune_studio.db import connection
    monkeypatch.setattr(connection.settings, "db_path", str(db_path))
    db.init_db()
    return TestClient(app), db_path


def _create_project_and_run(client: TestClient, tmp_path: Path) -> tuple[str, str]:
    r = client.post(
        "/api/projects",
        json={"name": f"bench-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    out = tmp_path / "out"
    out.mkdir()
    (out / "merged").mkdir()
    (out / "merged" / "config.json").write_text("{}", encoding="utf-8")
    run = db.create_run(pid, "train-1", base_model="x/test")
    db.update_run(run["id"], status="done", output_path=str(out))
    return pid, run["id"]


def _stub_engine(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """Replace InferenceEngine so generate returns a fixed answer."""

    class FakeEngine:
        def __init__(self) -> None:
            self.model = None
            self.model_path = None

        def load(self, path: str, **_kwargs: Any) -> None:
            self.model = object()
            self.model_path = path

        def unload(self) -> None:
            self.model = None

        def generate(self, messages: list, **_kwargs: Any) -> str:
            return answer

    monkeypatch.setattr(
        "finetune_studio.testing.inference.InferenceEngine",
        FakeEngine,
    )
    # Also silence global unload path.
    monkeypatch.setattr(
        "finetune_studio.webui.app.inference_engine",
        MagicMock(model=None, unload=MagicMock()),
        raising=False,
    )


def _exact_suite() -> str:
    """A built-in multiple-choice suite: the one kind of suite the benchmarks route runs and scores itself."""
    from finetune_studio.benchmarks.suite_defs import list_builtin_smoke_suites

    return list_builtin_smoke_suites()[0].path


def _write_quiz(tmp_path: Path) -> str:
    path = tmp_path / "quiz.json"
    path.write_text(json.dumps([{
        "name": "capital_fr", "category": "geo", "question": "Capital of France?",
        "correct_answer": "Paris", "keywords": ["Paris"],
    }]), encoding="utf-8")
    return str(path)


def _answer_key_engine(monkeypatch: pytest.MonkeyPatch, *, wrong: bool = False) -> None:
    """An engine that replies with each case's answer key (or a letter that is never right)."""
    from finetune_studio.testing.suite import load_test_suite

    key = {c.question: c.correct_answer for c in load_test_suite(_exact_suite())}

    class KeyEngine:
        model = None
        model_path = None

        def load(self, path: str, **_kw: Any) -> None:
            self.model, self.model_path = object(), path

        def unload(self) -> None:
            self.model = None

        def generate(self, messages: list, **_kw: Any) -> str:
            return "Answer: Z" if wrong else key[messages[-1]["content"]]

    monkeypatch.setattr("finetune_studio.testing.inference.InferenceEngine", KeyEngine)
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", MagicMock(model=None, unload=MagicMock()),
                        raising=False)


def test_public_suite_is_scored_by_exact_match_and_marked_so(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, db_path = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    _answer_key_engine(monkeypatch)

    r = client.post(
        f"/api/benchmarks/projects/{pid}/runs/{rid}/run",
        json={"suite_path": _exact_suite(), "suite_name": "mcq", "max_tokens": 32},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scores"]["pass_rate"] == 100.0 and body["scores"]["by_judge"] == {"exact": body["scores"]["total"]}
    assert {x["judge"] for x in body["results"]} == {"exact"}
    assert body["benchmark"]["scoring"] == "exact"

    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT verdict, judge FROM benchmark_cases WHERE benchmark_id = ?",
                           (body["benchmark_id"],)).fetchall()
        recorded = con.execute("SELECT kind, verdict FROM case_judgements WHERE benchmark_id = ?",
                               (body["benchmark_id"],)).fetchall()
    finally:
        con.close()
    assert rows and {r[1] for r in rows} == {"exact"} and {r[0] for r in rows} == {"pass"}
    assert len(recorded) == len(rows) and {k for k, _v in recorded} == {"exact"}


def test_public_suite_wrong_answers_fail_exactly(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _db_path = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    _answer_key_engine(monkeypatch, wrong=True)

    r = client.post(f"/api/benchmarks/projects/{pid}/runs/{rid}/run", json={"suite_path": _exact_suite()})
    assert r.status_code == 200, r.text
    assert {x["verdict"] for x in r.json()["results"]} == {"fail"}
    assert r.json()["scores"]["pass_rate"] == 0.0


def test_a_project_quiz_is_refused_here_and_points_at_the_testing_page(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _db_path = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    _stub_engine(monkeypatch, "Paris")

    r = client.post(f"/api/benchmarks/projects/{pid}/runs/{rid}/run", json={"suite_path": _write_quiz(tmp_path)})
    assert r.status_code == 400, r.text
    assert "Testing page" in r.json()["error"]
    assert db.list_benchmarks_for_project(pid) == []  # nothing was saved or scored


def test_the_scripted_rejudge_route_and_judge_mode_are_gone(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
) -> None:
    client, _db_path = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    bench = db.create_benchmark(rid, "x", {}, status="done", kind="suite")
    r = client.post(f"/api/benchmarks/projects/{pid}/benchmarks/{bench['id']}/judge",
                    json={"judge_mode": "heuristic"})
    assert r.status_code in (404, 405)


def test_benchmark_routes_reject_another_projects_benchmark(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
) -> None:
    client, _db_path = client_and_db
    owner_pid, rid = _create_project_and_run(client, tmp_path)
    outsider = client.post(
        "/api/projects",
        json={"name": f"outsider-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert outsider.status_code == 200, outsider.text
    outsider_pid = outsider.json()["id"]
    benchmark = db.create_benchmark(
        rid,
        "private-suite",
        {"total": 1, "judged": 1, "passed": 1, "pass_rate": 100.0},
        cases=[
            {
                "name": "secret-case",
                "question": "Secret question?",
                "correct_answer": "secret answer",
                "model_answer": "secret answer",
                "verdict": "pass",
                "judge": "heuristic",
            }
        ],
    )
    case = db.list_cases(benchmark["id"])[0]
    bid = benchmark["id"]

    assert client.get(
        f"/api/benchmarks/projects/{outsider_pid}/benchmarks/{bid}/cases"
    ).status_code == 404
    assert client.get(
        f"/api/benchmarks/projects/{outsider_pid}/benchmarks/{bid}/audit"
    ).status_code == 404
    assert client.post(
        f"/api/benchmarks/projects/{outsider_pid}/benchmarks/{bid}/cases/{case['id']}/verdict",
        json={"verdict": "fail", "reasoning": "tamper"},
    ).status_code == 404
    assert client.delete(
        f"/api/benchmarks/projects/{outsider_pid}/benchmarks/{bid}"
    ).status_code == 404

    assert db.get_benchmark(bid) is not None
    assert db.list_cases(bid)[0]["verdict"] == "pass"
    assert client.get(
        f"/api/benchmarks/projects/{owner_pid}/benchmarks/{bid}/cases"
    ).status_code == 200
