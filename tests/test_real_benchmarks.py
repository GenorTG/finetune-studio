"""Focused tests for real MMLU / GSM8K / HellaSwag evaluators (mocked HF)."""

from __future__ import annotations

from typing import Any

import pytest

from finetune_studio.benchmarks.real_benchmarks import (
    DEFAULT_SAMPLE_LIMIT,
    RealBenchmarkSuite,
    extract_gsm8k_gold,
    format_gsm8k_prompt,
    format_hellaswag_prompt,
    format_mmlu_prompt,
    is_real_suite_path,
    list_real_families,
    overall_accuracy_from_run_all,
    parse_real_suite_path,
    real_suite_path,
    resolve_sample_count,
    select_indices,
)
from finetune_studio.benchmarks.suite_defs import (
    discover_suites,
    list_builtin_real_suites,
)
from finetune_studio.testing.strict_scoring import (
    score_multiple_choice,
    score_numeric,
)


class _FakeSplit(list):
    """Minimal sequence mimicking a HuggingFace Dataset split."""


def _fake_loader_factory(splits: dict[str, _FakeSplit]):
    def _loader(
        path: str,
        name: str | None = None,
        *,
        split: str,
        cache_dir: str | None = None,
        revision: str | None = None,
    ) -> _FakeSplit:
        key = f"{path}|{name}|{split}"
        if key not in splits:
            raise KeyError(f"unexpected load: {key}")
        return splits[key]

    return _loader


class _ScriptedEngine:
    """Returns canned responses in call order."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self.answers:
            return ""
        return self.answers.pop(0)


def test_real_suite_paths_and_catalog() -> None:
    assert list_real_families() == ["mmlu", "gsm8k", "hellaswag"]
    assert real_suite_path("mmlu") == "real://mmlu"
    assert parse_real_suite_path("real://gsm8k") == "gsm8k"
    assert is_real_suite_path("real://hellaswag") is True
    assert is_real_suite_path("data/benchmarks/x.json") is False
    suites = list_builtin_real_suites()
    assert {s.name for s in suites} == {
        "mmlu_real",
        "gsm8k_real",
        "hellaswag_real",
    }
    for s in suites:
        assert s.suite_type == "real"
        assert s.is_real_benchmark is True
        assert s.is_industry_benchmark is True
        assert s.label().startswith("real ·")
        assert s.default_sample_limit == DEFAULT_SAMPLE_LIMIT


def test_discover_includes_real_suites(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    suites = discover_suites()
    real = [s for s in suites if s["suite_type"] == "real"]
    assert len(real) == 3
    assert all(s["is_real_benchmark"] for s in real)
    assert all(s["path"].startswith("real://") for s in real)
    assert suites[0]["suite_type"] == "real"


def test_select_indices_deterministic() -> None:
    a = select_indices(10, 4, seed=7, order="seeded_shuffle")
    b = select_indices(10, 4, seed=7, order="seeded_shuffle")
    assert a == b
    assert len(a) == 4
    assert select_indices(10, 3, order="dataset") == [0, 1, 2]
    assert resolve_sample_count(split_size=100, num_samples=50, full_run=False) == 50
    assert resolve_sample_count(split_size=100, num_samples=50, full_run=True) == 100


def test_strict_mcq_rejects_ambiguous_and_substring() -> None:
    scored = score_multiple_choice(
        correct_answer="B",
        model_answer="The answer is A. Also answer is B.",
    )
    assert scored.verdict == "fail"
    assert scored.validity in {"ambiguous", "wrong_extra"}
    # Substring of option text alone is not enough without a letter.
    scored2 = score_multiple_choice(
        correct_answer="C",
        model_answer="Paris is lovely in the spring",
    )
    assert scored2.verdict == "fail"
    assert scored2.validity == "no_answer"
    scored3 = score_multiple_choice(correct_answer="C", model_answer="C")
    assert scored3.verdict == "pass"


def test_gsm8k_exact_normalized_final() -> None:
    assert extract_gsm8k_gold("step\n#### 1,234") == "1234"
    scored = score_numeric(
        correct_answer="42",
        model_answer="Working… #### 42",
    )
    assert scored.verdict == "pass"
    # Keyword / substring credit must not pass (42 inside 420).
    scored_bad = score_numeric(
        correct_answer="42",
        model_answer="The answer is 420",
    )
    assert scored_bad.verdict == "fail"


def test_evaluate_mmlu_with_mocked_dataset() -> None:
    rows = _FakeSplit(
        [
            {
                "question": "Capital of France?",
                "choices": ["Berlin", "Madrid", "Paris", "Rome"],
                "answer": 2,
                "subject": "geography",
            },
            {
                "question": "2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
                "subject": "math",
            },
        ]
    )
    loader = _fake_loader_factory(
        {"cais/mmlu|all|test": rows}
    )
    suite = RealBenchmarkSuite(dataset_loader=loader)
    engine = _ScriptedEngine(["C", "B"])
    out = suite.evaluate_mmlu(engine, num_samples=2)
    assert out["total"] == 2
    assert out["correct"] == 2
    assert out["accuracy"] == 100.0
    assert out["metadata"]["is_real_benchmark"] is True
    assert out["metadata"]["dataset_id"] == "cais/mmlu"
    assert out["metadata"]["split"] == "test"
    assert out["metadata"]["scoring_method"] == "strict_mcq"
    assert out["metadata"]["sample_count"] == 2
    assert "Reply with ONLY the letter" in format_mmlu_prompt("q", ["a", "b", "c", "d"])


def test_evaluate_gsm8k_and_hellaswag_mocked() -> None:
    gsm = _FakeSplit(
        [
            {
                "question": "What is 6*7?",
                "answer": "Reasoning\n#### 42",
            }
        ]
    )
    hs = _FakeSplit(
        [
            {
                "ctx": "A person is eating.",
                "endings": ["flies", "continues eating", "vanishes", "sings"],
                "label": "1",
                "activity_label": "eating",
            }
        ]
    )
    loader = _fake_loader_factory(
        {
            "openai/gsm8k|main|test": gsm,
            "Rowan/hellaswag|None|validation": hs,
        }
    )
    suite = RealBenchmarkSuite(dataset_loader=loader)

    gsm_out = suite.evaluate_gsm8k(
        _ScriptedEngine(["step #### 42"]), num_samples=1
    )
    assert gsm_out["correct"] == 1
    assert gsm_out["metadata"]["dataset_id"] == "openai/gsm8k"
    assert gsm_out["metadata"]["scoring_method"] == "strict_numeric"
    assert "####" in format_gsm8k_prompt("q")

    hs_out = suite.evaluate_hellaswag(
        _ScriptedEngine(["B"]), num_samples=1
    )
    assert hs_out["correct"] == 1
    assert hs_out["metadata"]["split"] == "validation"
    assert "What happens next?" in format_hellaswag_prompt("ctx", ["a", "b", "c", "d"])


def test_run_all_summary_not_scalar_map() -> None:
    rows = _FakeSplit(
        [
            {
                "question": "Capital?",
                "choices": ["A1", "A2", "A3", "A4"],
                "answer": 0,
                "subject": "x",
            }
        ]
    )
    loader = _fake_loader_factory(
        {
            "cais/mmlu|all|test": rows,
            "openai/gsm8k|main|test": _FakeSplit(
                [{"question": "1+1?", "answer": "#### 2"}]
            ),
            "Rowan/hellaswag|None|validation": _FakeSplit(
                [
                    {
                        "ctx": "c",
                        "endings": ["w", "x", "y", "z"],
                        "label": 0,
                        "activity_label": "a",
                    }
                ]
            ),
        }
    )
    suite = RealBenchmarkSuite(dataset_loader=loader)
    engine = _ScriptedEngine(["A", "#### 2", "A"])
    payload = suite.run_all(
        engine, num_samples=1, benchmarks=["mmlu", "gsm8k", "hellaswag"]
    )
    assert "benchmarks" in payload
    assert "summary" in payload
    overall = overall_accuracy_from_run_all(payload)
    assert overall == 100.0
    # The broken endpoint did sum(results.values()) — must not be valid here.
    with pytest.raises(TypeError):
        sum(payload.values())  # type: ignore[arg-type]


def test_inference_benchmark_endpoint_uses_overall_helper(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from finetune_studio.benchmarks import real_benchmarks as rb
    from finetune_studio.webui import app as app_mod

    rows = _FakeSplit(
        [
            {
                "question": "Q?",
                "choices": ["a", "b", "c", "d"],
                "answer": 0,
                "subject": "s",
            }
        ]
    )

    def _loader(path, name=None, *, split, cache_dir=None, revision=None):
        return rows

    orig = rb.RealBenchmarkSuite

    def _factory(*_a, **_k):
        return orig(dataset_loader=_loader)

    monkeypatch.setattr(rb, "RealBenchmarkSuite", _factory)

    class _Eng:
        model = object()

        def generate(self, messages, **kwargs):
            return "A"

    monkeypatch.setattr(app_mod, "inference_engine", _Eng())

    client = TestClient(app_mod.app)
    r = client.post(
        "/api/chat-v2/inference/benchmark",
        json={"num_samples": 1, "benchmarks": ["mmlu"]},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "error" not in data
    assert isinstance(data["overall"], (int, float))
    assert "benchmarks" in data["results"]
    assert "summary" in data["results"]
