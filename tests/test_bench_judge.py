"""Regression tests for bench-exec heuristic judge (QABUG-006 / QABUG-010)."""

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


def _write_suite(tmp_path: Path, keywords: list[str], correct: str = "Paris") -> str:
    suite = [
        {
            "name": "capital_fr",
            "category": "geo",
            "question": "Capital of France?",
            "correct_answer": correct,
            "keywords": keywords,
        }
    ]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")
    return str(path)


def test_bench_judge_invokes_heuristic(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _db_path = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    _stub_engine(monkeypatch, "Paris")
    suite_path = _write_suite(tmp_path, ["Paris"])

    r = client.post(
        f"/api/benchmarks/projects/{pid}/runs/{rid}/run",
        json={
            "suite_path": suite_path,
            "suite_name": "kw-test",
            "judge_mode": "heuristic",
            "max_tokens": 32,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body or not body.get("error"), body
    results = body["results"]
    assert len(results) == 1
    assert results[0]["verdict"] == "pass"


def test_bench_judge_records_verdict_in_db(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, db_path = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    _stub_engine(monkeypatch, "Paris")
    suite_path = _write_suite(tmp_path, ["Paris"])

    r = client.post(
        f"/api/benchmarks/projects/{pid}/runs/{rid}/run",
        json={
            "suite_path": suite_path,
            "suite_name": "kw-db",
            "judge_mode": "heuristic",
            "max_tokens": 32,
        },
    )
    assert r.status_code == 200, r.text
    bid = r.json()["benchmark"]["id"]

    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT verdict, judge FROM benchmark_cases WHERE benchmark_id = ?",
            (bid,),
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    assert row[0] == "pass"
    assert row[1] == "heuristic"


def test_bench_judge_marks_fail_when_keywords_missed(
    client_and_db: tuple[TestClient, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _db_path = client_and_db
    pid, rid = _create_project_and_run(client, tmp_path)
    _stub_engine(monkeypatch, "London")
    suite_path = _write_suite(tmp_path, ["Paris"])

    r = client.post(
        f"/api/benchmarks/projects/{pid}/runs/{rid}/run",
        json={
            "suite_path": suite_path,
            "suite_name": "kw-fail",
            "judge_mode": "heuristic",
            "max_tokens": 32,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["verdict"] == "fail"
