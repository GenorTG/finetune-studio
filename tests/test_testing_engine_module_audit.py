"""Regression tests for the E1 testing/ module audit (2026-10-01).

Pins three fixes:
1. ``score_results`` no longer excludes errored/unjudged cases from the
   ``pass_rate`` / ``weighted_score`` denominator (was inflating accuracy).
2. ``InferenceEngine._generate_hf`` now honors ``stop`` sequences instead of
   silently dropping them (GGUF models already honored ``stop``; HF models
   did not).
3. ``full_corpus_suite.cases_from_pairs_directory`` logs (instead of
   silently swallowing) unreadable/malformed pair files while still
   returning the cases it could build.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from finetune_studio.testing.full_corpus_suite import cases_from_pairs_directory
from finetune_studio.testing.inference import InferenceEngine
from finetune_studio.testing.suite import CaseResult, score_results


def test_score_results_does_not_inflate_pass_rate_with_errored_cases() -> None:
    """An errored case with no verdict must count against pass_rate, not be excluded."""
    results = [
        CaseResult(
            case_name="ok",
            category="general",
            question="q1",
            correct_answer="c1",
            model_answer="good answer",
            verdict="pass",
        ),
        CaseResult(
            case_name="crashed",
            category="general",
            question="q2",
            correct_answer="c2",
            model_answer="",
            error="engine crashed mid-generation",
            verdict="",  # judging never produced a verdict
        ),
    ]
    scores = score_results(results)

    assert scores["total"] == 2
    assert scores["judged"] == 1
    assert scores["unjudged"] == 1
    assert scores["passed"] == 1

    # Before the fix: pass_rate = 1/1 * 100 = 100.0 (errored case excluded).
    # After the fix: pass_rate = 1/2 * 100 = 50.0 (errored case counts).
    assert scores["pass_rate"] == 50.0
    assert scores["weighted_score"] == 50.0


def test_score_results_all_judged_is_unaffected() -> None:
    """When every case has a verdict, judged == total and behavior is unchanged."""
    results = [
        CaseResult(
            case_name="a", category="g", question="q", correct_answer="c",
            model_answer="x", verdict="pass",
        ),
        CaseResult(
            case_name="b", category="g", question="q", correct_answer="c",
            model_answer="x", verdict="fail",
        ),
    ]
    scores = score_results(results)
    assert scores["judged"] == scores["total"] == 2
    assert scores["unjudged"] == 0
    assert scores["pass_rate"] == 50.0


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                              enable_thinking=None):
        return "PROMPT"

    def __call__(self, text, return_tensors="pt"):
        return _FakeBatch({"input_ids": torch.zeros((1, 3), dtype=torch.long)})

    def decode(self, generated, skip_special_tokens=True):
        return "answer before STOPMARK answer after"


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return torch.zeros((1, 8), dtype=torch.long)


def _fake_engine() -> InferenceEngine:
    engine = InferenceEngine()
    engine.tokenizer = _FakeTokenizer()
    engine.model = _FakeModel()
    engine.is_gguf = False
    return engine


def test_generate_hf_honors_stop_sequence() -> None:
    """Previously `stop` was accepted but never used for HF-loaded models."""
    engine = _fake_engine()
    response = engine._generate_hf(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16, temperature=0.0, top_p=0.9, top_k=40,
        repeat_penalty=1.1, stop=["STOPMARK"],
    )
    assert response == "answer before "
    assert "STOPMARK" not in response
    assert "answer after" not in response


def test_generate_hf_without_stop_returns_full_text() -> None:
    engine = _fake_engine()
    response = engine._generate_hf(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16, temperature=0.0, top_p=0.9, top_k=40,
        repeat_penalty=1.1, stop=None,
    )
    assert response == "answer before STOPMARK answer after"


def test_generate_hf_stop_picks_earliest_match_among_multiple() -> None:
    engine = _fake_engine()
    response = engine._generate_hf(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16, temperature=0.0, top_p=0.9, top_k=40,
        repeat_penalty=1.1, stop=["answer after", "STOPMARK"],
    )
    # "STOPMARK" occurs earlier in the fixed decode string than "answer after".
    assert response == "answer before "


def test_cases_from_pairs_directory_logs_unreadable_files(
    tmp_path: Path, caplog: logging.LogCaptureFixture,
) -> None:
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    good = {
        "status": "approved",
        "source_id": "doc-1",
        "question": "What is X?",
        "answer": "X is Y.",
    }
    (pairs_dir / "good.json").write_text(json.dumps(good), encoding="utf-8")
    (pairs_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    (pairs_dir / "not_an_object.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        cases = cases_from_pairs_directory(pairs_dir)

    assert len(cases) == 1
    assert cases[0].source_id == "doc-1"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 2
    assert any("broken.json" in r.message for r in warnings)
    assert any("not_an_object.json" in r.message for r in warnings)
