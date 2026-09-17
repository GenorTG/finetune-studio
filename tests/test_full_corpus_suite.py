"""Tests for project-local full-ingested-corpus discovery/generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from finetune_studio import db
from finetune_studio.benchmarks.suite_defs import discover_suites
from finetune_studio.config import settings
from finetune_studio.data.fs import paths as paths_mod
from finetune_studio.data.fs import qa as qa_mod
from finetune_studio.testing.full_corpus_suite import (
    FULL_CORPUS_SUITE_NAME,
    case_from_pair,
    ensure_full_corpus_suite,
    ensure_full_corpus_suite_definition,
    suite_path_for_project,
)


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "full_corpus.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return db_path


@pytest.fixture
def project_fs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> Path:
    """Isolate project_dir under tmp_path for both paths and qa modules."""
    root = tmp_path / "projects"
    root.mkdir(parents=True)

    def _project_dir(pid: str) -> Path:
        p = root / pid
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(paths_mod, "project_dir", _project_dir)
    monkeypatch.setattr(qa_mod, "project_dir", _project_dir)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "benchmarks").mkdir(parents=True)
    return root


def _write_pair(
    project_root: Path,
    pid: str,
    pair_id: str,
    *,
    status: str = "approved",
    source_id: str | None = "src-a",
    question: str = "What is X?",
    answer: str = "X is Y",
    messages: list[dict[str, Any]] | None = None,
) -> None:
    pairs_dir = project_root / pid / "qa" / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "id": pair_id,
        "status": status,
        "question": question,
        "answer": answer,
    }
    if source_id is not None:
        payload["source_id"] = source_id
    if messages is not None:
        payload["messages"] = messages
    (pairs_dir / f"{pair_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_case_from_pair_requires_approved_and_source() -> None:
    assert case_from_pair({"status": "draft", "source_id": "s", "question": "q", "answer": "a"}) is None
    assert case_from_pair({"status": "approved", "question": "q", "answer": "a"}) is None
    case = case_from_pair(
        {
            "id": "p1",
            "status": "approved",
            "source_id": "src1",
            "question": "Q?",
            "answer": "A",
        }
    )
    assert case is not None
    assert case.name == "p1"
    assert case.source_id == "src1"
    assert case.question == "Q?"
    assert case.correct_answer == "A"


def test_case_from_pair_reads_messages_when_fields_empty() -> None:
    case = case_from_pair(
        {
            "id": "m1",
            "status": "approved",
            "source_id": "src1",
            "messages": [
                {"role": "user", "content": "From messages?"},
                {"role": "assistant", "content": "Yes"},
            ],
        }
    )
    assert case is not None
    assert case.question == "From messages?"
    assert case.correct_answer == "Yes"


def test_ensure_writes_suite_and_definition(project_fs: Path) -> None:
    pid = "proj-a"
    _write_pair(project_fs, pid, "qa1", question="Q1?", answer="A1")
    _write_pair(project_fs, pid, "qa2", question="Q2?", answer="A2", source_id="src-b")
    _write_pair(
        project_fs,
        pid,
        "qa-draft",
        status="draft",
        question="Skip?",
        answer="No",
    )
    _write_pair(
        project_fs,
        pid,
        "qa-nosource",
        source_id=None,
        question="Skip2?",
        answer="No",
    )

    result = ensure_full_corpus_suite(pid)
    assert result.written is True
    assert result.case_count == 2
    assert result.source_count == 2
    assert result.path == suite_path_for_project(pid)
    assert result.path.is_file()

    data = json.loads(result.path.read_text(encoding="utf-8"))
    assert data["name"] == FULL_CORPUS_SUITE_NAME
    assert data["case_count"] == 2
    assert len(data["cases"]) == 2
    names = {c["name"] for c in data["cases"]}
    assert names == {"qa1", "qa2"}

    definition = ensure_full_corpus_suite_definition(pid)
    assert definition is not None
    assert definition.name == FULL_CORPUS_SUITE_NAME
    assert definition.suite_type == "local"
    assert definition.source == "project_qa"
    assert definition.case_count == 2
    assert definition.path == str(result.path)
    assert "2 cases" in definition.label()


def test_ensure_no_pairs_returns_none_and_clears_stale(
    project_fs: Path,
) -> None:
    pid = "proj-empty"
    stale = suite_path_for_project(pid)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"name": "full-ingested-corpus", "cases": []}', encoding="utf-8")

    result = ensure_full_corpus_suite(pid)
    assert result.written is False
    assert result.case_count == 0
    assert not stale.exists()
    assert ensure_full_corpus_suite_definition(pid) is None


def test_discover_suites_includes_full_corpus_for_project(
    project_fs: Path,
) -> None:
    proj = db.create_project(name="corpus-proj", base_model="x/y")
    pid = proj["id"]
    for i in range(3):
        _write_pair(
            project_fs,
            pid,
            f"qa{i}",
            question=f"Q{i}?",
            answer=f"A{i}",
            source_id=f"src-{i}",
        )

    suites = discover_suites(pid)
    corpus = [
        s
        for s in suites
        if s.get("name") == FULL_CORPUS_SUITE_NAME and s.get("source") == "project_qa"
    ]
    assert len(corpus) == 1
    assert corpus[0]["suite_type"] == "local"
    assert corpus[0]["case_count"] == 3
    assert corpus[0]["label"] == f"local · {FULL_CORPUS_SUITE_NAME} (3 cases)"
    assert Path(corpus[0]["path"]).is_file()

    # Without project_id: no project_qa suite.
    global_suites = discover_suites()
    assert all(s.get("source") != "project_qa" for s in global_suites)


def test_discover_suites_keeps_auto_and_local(
    project_fs: Path, tmp_path: Path
) -> None:
    import time

    bench = tmp_path / "data" / "benchmarks" / "default.json"
    bench.write_text(
        json.dumps([{"name": "a", "question": "q", "correct_answer": "a"}]),
        encoding="utf-8",
    )

    proj = db.create_project(name="mixed-proj", base_model="x/y")
    pid = proj["id"]
    _write_pair(project_fs, pid, "qa1")

    run = db.create_run(pid, "r1", base_model="x/y")
    auto_path = tmp_path / "auto_eval.json"
    auto_path.write_text(
        json.dumps([{"name": "c1", "question": "q", "correct_answer": "a"}]),
        encoding="utf-8",
    )
    with db.cursor() as c:
        c.execute(
            "INSERT INTO auto_suites "
            "(id, run_id, project_id, suite_name, suite_path, case_count, "
            "categories_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "as-fc",
                run["id"],
                pid,
                "held_out",
                str(auto_path),
                1,
                "{}",
                time.time(),
            ),
        )

    suites = discover_suites(pid)
    by_source = {s.get("source") for s in suites}
    assert "project_qa" in by_source
    assert "auto_suites" in by_source
    assert "data_benchmarks" in by_source
    assert any(s.get("name") == "held_out" and s.get("suite_type") == "auto" for s in suites)
    assert any(s.get("name") == "default" and s.get("suite_type") == "local" for s in suites)
    assert any(
        s.get("name") == FULL_CORPUS_SUITE_NAME and s.get("source") == "project_qa"
        for s in suites
    )
