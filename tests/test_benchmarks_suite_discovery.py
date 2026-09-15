"""Tests for benchmark suite discovery (E2E-43).

Built-in industry smoke suites (package fixtures) are always discoverable.
Local ``data/benchmarks/*.json`` and project ``auto_suites`` remain optional.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui.routes.benchmarks import _discover_suites

_EXPECTED_INDUSTRY: frozenset[str] = frozenset(
    {"mmlu_smoke", "gsm8k_smoke", "hellaswag_smoke"}
)


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "discover.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return db_path


def _industry_names(suites: list[dict[str, Any]]) -> set[str]:
    return {
        str(s["name"])
        for s in suites
        if s.get("suite_type") == "industry_smoke"
    }


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
    # Industry smoke fixtures are package-local and always discoverable.
    assert _industry_names(suites) == set(_EXPECTED_INDUSTRY)
    assert all(s.get("suite_type") == "industry_smoke" for s in suites)
    assert all(str(s.get("label", "")).startswith("industry ·") for s in suites)


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
    by_name = {s["name"]: s for s in suites}
    assert "default" in by_name
    assert by_name["default"]["path"] == "data/benchmarks/default.json"
    assert by_name["default"]["suite_type"] == "local"
    assert by_name["default"]["label"].startswith("local ·")
    assert _industry_names(suites) == set(_EXPECTED_INDUSTRY)
    # Industry smoke sorts before local.
    assert suites[0]["suite_type"] == "industry_smoke"
    local_idx = next(i for i, s in enumerate(suites) if s["name"] == "default")
    assert local_idx >= len(_EXPECTED_INDUSTRY)


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
    auto = [s for s in suites if s.get("suite_type") == "auto"]
    assert len(auto) == 1
    assert auto[0]["name"] == "held_out"
    assert auto[0]["path"] == str(auto_path)
    assert auto[0]["label"] == "auto · held_out (3 cases)"
    assert _industry_names(suites) == set(_EXPECTED_INDUSTRY)
    assert suites[-1]["suite_type"] == "auto"


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

    suites = _discover_suites()
    assert all(s.get("suite_type") != "auto" for s in suites)
    assert all(s.get("suite_type") != "auto" for s in _discover_suites(None))
    assert _industry_names(suites) == set(_EXPECTED_INDUSTRY)
    assert _industry_names(_discover_suites(None)) == set(_EXPECTED_INDUSTRY)
