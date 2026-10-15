"""Tests for benchmark suite discovery (E2E-43).

Built-in synthetic smoke/offline suites (package fixtures) are always discoverable.
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

_EXPECTED_SMOKE: frozenset[str] = frozenset(
    {"mmlu_smoke", "gsm8k_smoke", "hellaswag_smoke"}
)
_EXPECTED_OFFLINE: frozenset[str] = frozenset(
    {"mmlu_offline", "gsm8k_offline", "hellaswag_offline"}
)


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "discover.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return db_path


def _names_of_type(suites: list[dict[str, Any]], suite_type: str) -> set[str]:
    return {
        str(s["name"])
        for s in suites
        if s.get("suite_type") == suite_type
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
    # Built-in fixtures are package-local and always discoverable.
    assert _names_of_type(suites, "synthetic_smoke") == set(_EXPECTED_SMOKE)
    assert _names_of_type(suites, "synthetic_offline") == set(_EXPECTED_OFFLINE)
    assert _names_of_type(suites, "real") == {
        "mmlu_real",
        "gsm8k_real",
        "hellaswag_real",
    }
    assert all(
        s.get("suite_type")
        in {"synthetic_smoke", "synthetic_offline", "real"}
        for s in suites
    )
    for s in suites:
        if s.get("suite_type") == "real":
            assert str(s.get("label", "")).startswith("real ·")
            assert s.get("is_real_benchmark") is True
        else:
            assert str(s.get("label", "")).startswith("synthetic ·")
            assert "not industry" in str(s.get("label", ""))


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
    assert _names_of_type(suites, "synthetic_smoke") == set(_EXPECTED_SMOKE)
    assert _names_of_type(suites, "synthetic_offline") == set(_EXPECTED_OFFLINE)
    # Real HF suites sort before synthetic offline; both before local.
    assert suites[0]["suite_type"] == "real"
    local_idx = next(i for i, s in enumerate(suites) if s["name"] == "default")
    assert local_idx >= 3 + len(_EXPECTED_SMOKE) + len(_EXPECTED_OFFLINE)


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
    assert _names_of_type(suites, "synthetic_smoke") == set(_EXPECTED_SMOKE)
    assert _names_of_type(suites, "synthetic_offline") == set(_EXPECTED_OFFLINE)
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
    assert _names_of_type(suites, "synthetic_smoke") == set(_EXPECTED_SMOKE)
    assert _names_of_type(suites, "synthetic_offline") == set(_EXPECTED_OFFLINE)
    assert _names_of_type(_discover_suites(None), "synthetic_smoke") == set(
        _EXPECTED_SMOKE
    )


def test_auto_suite_regeneration_lists_newest_case_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> None:
    """Same suite file re-generated (500→554) must show the NEWEST row.

    Regression for the full-coverage rollout: the old capped row shadowed the
    fresh full-coverage row in the picker because the DESC loop kept
    overwriting with older rows.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "benchmarks").mkdir(parents=True)

    proj = db.create_project(name="regen-proj", base_model="x/y")
    run = db.create_run(proj["id"], "r1", base_model="x/y")
    auto_path = tmp_path / "suite_regen.json"
    auto_path.write_text(json.dumps([]), encoding="utf-8")
    now = time.time()
    with db.cursor() as c:
        for sid, cnt, ts in (("as-old", 500, now - 100), ("as-new", 554, now)):
            c.execute(
                "INSERT INTO auto_suites "
                "(id, run_id, project_id, suite_name, suite_path, case_count, "
                "categories_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sid, run["id"], proj["id"], "regen", str(auto_path), cnt, "{}", ts),
            )

    suites = _discover_suites(proj["id"])
    auto = [s for s in suites if s.get("suite_type") == "auto" and s["name"] == "regen"]
    assert len(auto) == 1, "one entry per suite file"
    assert auto[0]["case_count"] == 554, "newest regeneration wins"
