"""Tests for substantive synthetic offline benchmark fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio.benchmarks.offline_suites import (
    MIN_OFFLINE_CASES,
    ensure_offline_fixtures,
    offline_suite_metas,
    write_offline_fixtures,
)
from finetune_studio.benchmarks.suite_defs import (
    discover_suites,
    list_builtin_offline_suites,
    list_builtin_smoke_suites,
)
from finetune_studio.testing.suite import (
    CaseResult,
    apply_heuristic_judging,
    load_test_suite,
    score_results,
)


def test_offline_fixtures_are_substantive() -> None:
    paths = write_offline_fixtures(force=False)
    assert len(paths) == 3
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["suite_type"] == "industry_offline"
        assert data["synthetic"] is True
        assert data["offline"] is True
        assert data["licensed_external_data"] is False
        assert data["scoring"] == "deterministic_keywords"
        assert len(data["cases"]) >= MIN_OFFLINE_CASES
        cases = load_test_suite(str(path))
        assert len(cases) >= MIN_OFFLINE_CASES
        assert all(c.keywords for c in cases)


def test_offline_suites_discovered_and_labeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ensure_offline_fixtures()
    offline = list_builtin_offline_suites()
    names = {s.name for s in offline}
    assert names == {"mmlu_offline", "gsm8k_offline", "hellaswag_offline"}
    for s in offline:
        assert s.suite_type == "industry_offline"
        assert "offline synthetic" in s.label()
        assert "synthetic" in s.description.lower() or "offline" in s.description.lower()
        assert s.case_count is not None and s.case_count >= MIN_OFFLINE_CASES

    discovered = discover_suites()
    types = {s["suite_type"] for s in discovered}
    assert "industry_offline" in types
    assert "industry_smoke" in types
    assert any("offline synthetic" in s["label"] for s in discovered)


def test_smoke_labels_mark_synthetic_offline() -> None:
    for s in list_builtin_smoke_suites():
        assert "synthetic/offline" in s.label()


def test_offline_keyword_scoring_is_deterministic() -> None:
    meta = offline_suite_metas()[0]
    path = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "benchmarks" / "fixtures" / meta["filename"]
    cases = load_test_suite(str(path))
    results = [
        CaseResult(
            case_name=c.name,
            category=c.category,
            question=c.question,
            correct_answer=c.correct_answer,
            model_answer=c.correct_answer,
            keywords=list(c.keywords),
        )
        for c in cases[:5]
    ]
    apply_heuristic_judging(results)
    scores = score_results(results)
    assert scores["judged"] == 5
    assert scores["passed"] == 5
    assert scores["unjudged"] == 0
    assert scores["pass_rate"] == 100.0
