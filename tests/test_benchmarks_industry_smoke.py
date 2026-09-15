"""Tests for industry smoke suite discovery, labels, and selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.benchmarks.suite_defs import (
    discover_suites,
    fixtures_dir,
    is_selectable_suite,
    list_builtin_smoke_suites,
)
from finetune_studio.config import settings
from finetune_studio.testing.suite import load_test_suite
from finetune_studio.webui.routes.benchmarks import _validate_suite_file


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "smoke.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    return db_path


def test_builtin_smoke_suites_are_present() -> None:
    suites = list_builtin_smoke_suites()
    names = {s.name for s in suites}
    assert names == {"mmlu_smoke", "gsm8k_smoke", "hellaswag_smoke"}
    for s in suites:
        assert s.suite_type == "industry_smoke"
        assert s.source == "builtin"
        assert s.version == 1
        assert s.family in {"mmlu", "gsm8k", "hellaswag"}
        assert Path(s.path).is_file()
        assert s.case_count is not None and s.case_count >= 1


def test_smoke_labels_are_readable() -> None:
    suites = list_builtin_smoke_suites()
    by_name = {s.name: s for s in suites}
    assert "industry ·" in by_name["mmlu_smoke"].label()
    assert "MMLU-style" in by_name["mmlu_smoke"].label()
    assert "smoke v1" in by_name["mmlu_smoke"].label()
    assert "GSM8K-style" in by_name["gsm8k_smoke"].label()
    assert "HellaSwag-style" in by_name["hellaswag_smoke"].label()


def test_discover_includes_industry_without_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    suites = discover_suites()
    types = {s["suite_type"] for s in suites}
    assert "industry_smoke" in types
    assert "local" not in types
    names = {s["name"] for s in suites}
    assert {"mmlu_smoke", "gsm8k_smoke", "hellaswag_smoke"} <= names
    for s in suites:
        if s["suite_type"] == "industry_smoke":
            assert s["label"].startswith("industry ·")
            assert s["source"] == "builtin"


def test_discover_keeps_local_and_auto_alongside_industry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> None:
    bench = tmp_path / "data" / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "default.json").write_text(
        json.dumps([{"name": "a", "question": "q", "correct_answer": "a"}]),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    proj = db.create_project(name="smoke-proj", base_model="x/y")
    run = db.create_run(proj["id"], "r1", base_model="x/y")
    auto_path = tmp_path / "auto.json"
    auto_path.write_text(
        json.dumps([{"name": "c1", "question": "q", "correct_answer": "a"}]),
        encoding="utf-8",
    )
    with db.cursor() as c:
        c.execute(
            "INSERT INTO auto_suites "
            "(id, run_id, project_id, suite_name, suite_path, case_count, "
            "categories_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("as1", run["id"], proj["id"], "held_out", str(auto_path), 2, "{}", 0),
        )

    suites = discover_suites(proj["id"])
    by_type: dict[str, list[dict]] = {}
    for s in suites:
        by_type.setdefault(s["suite_type"], []).append(s)

    assert len(by_type["industry_smoke"]) == 3
    assert any(s["name"] == "default" for s in by_type["local"])
    assert any(s["name"] == "held_out" for s in by_type["auto"])
    assert by_type["auto"][0]["label"].startswith("auto ·")
    assert by_type["local"][0]["label"].startswith("local ·")


def test_versioned_fixtures_load_via_load_test_suite() -> None:
    for path in sorted(fixtures_dir().glob("*.v1.json")):
        cases = load_test_suite(str(path))
        assert len(cases) >= 1
        assert cases[0].question
        assert cases[0].correct_answer


def test_selection_validation_accepts_industry_rejects_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    smoke = list_builtin_smoke_suites()[0]
    assert is_selectable_suite(smoke.path) is True
    assert is_selectable_suite(str(tmp_path / "nope.json")) is False

    cases, err = _validate_suite_file(
        smoke.path, require_selectable=True, project_id=None
    )
    assert err is None
    assert cases is not None
    assert len(cases) >= 1

    rogue = tmp_path / "rogue.json"
    rogue.write_text(
        json.dumps([{"name": "x", "question": "q", "correct_answer": "a"}]),
        encoding="utf-8",
    )
    _, err2 = _validate_suite_file(
        str(rogue), require_selectable=True, project_id=None
    )
    assert err2 is not None
    assert err2.status_code == 400
    assert "not selectable" in json.loads(err2.body.decode())["error"]


def test_api_suites_lists_industry_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_db: Path
) -> None:
    from fastapi.testclient import TestClient

    from finetune_studio.webui.app import app

    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    r = client.get("/api/benchmarks/suites")
    assert r.status_code == 200
    suites = r.json()
    labels = [s["label"] for s in suites]
    assert any("MMLU-style" in lb for lb in labels)
    assert any("GSM8K-style" in lb for lb in labels)
    assert any("HellaSwag-style" in lb for lb in labels)
    for s in suites:
        if s.get("suite_type") == "industry_smoke":
            assert s.get("source") == "builtin"
            assert "{" not in s["label"]
