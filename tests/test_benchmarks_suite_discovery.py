"""Tests for benchmark suite discovery (E2E-43)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui.routes.benchmarks import _discover_suites


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "discover.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return db_path


def test_missing_known_suite_files_not_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> None:
    bench_dir = tmp_path / "data" / "benchmarks"
    bench_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    suites = _discover_suites()
    names = {s["name"] for s in suites}
    assert "default" not in names
    assert "tool_calling" not in names
    assert "chris_ai_v21" not in names
    assert suites == []


def test_existing_json_file_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> None:
    bench_dir = tmp_path / "data" / "benchmarks"
    bench_dir.mkdir(parents=True)
    suite_path = bench_dir / "default.json"
    suite_path.write_text(
        json.dumps([{"name": "a", "question": "q", "correct_answer": "a"}]),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    suites = _discover_suites()
    assert len(suites) == 1
    assert suites[0]["name"] == "default"
    assert suites[0]["path"] == "data/benchmarks/default.json"


def test_auto_suites_rows_listed_for_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "benchmarks").mkdir(parents=True)

    proj = db.create_project(name="auto-suite-proj", base_model="x/y")
    run = db.create_run(proj["id"], "r1", base_model="x/y")
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
                "as1",
                run["id"],
                proj["id"],
                "held_out",
                str(auto_path),
                3,
                "{}",
                time.time(),
            ),
        )

    suites = _discover_suites(proj["id"])
    assert len(suites) == 1
    assert suites[0]["name"] == "held_out"
    assert suites[0]["path"] == str(auto_path)
    assert suites[0]["label"] == "auto · held_out (3 cases)"


def test_auto_suites_not_listed_without_project_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "benchmarks").mkdir(parents=True)
    proj = db.create_project(name="p2", base_model="x/y")
    run = db.create_run(proj["id"], "r1", base_model="x/y")
    with db.cursor() as c:
        c.execute(
            "INSERT INTO auto_suites "
            "(id, run_id, project_id, suite_name, suite_path, case_count, "
            "categories_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("as2", run["id"], proj["id"], "held_out", "/tmp/x.json", 1, "{}", time.time()),
        )

    assert _discover_suites() == []
    assert _discover_suites(None) == []
