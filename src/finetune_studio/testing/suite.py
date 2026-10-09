"""Test suite — Q&A cases, run against a model, judged afterwards.

WHAT THIS FILE DOES
===================
  - BenchmarkCase: a question + the correct answer from the data (+ optional key values)
  - CaseResult: what the model answered, with the full transcript — and, once judged, the verdict
  - run_suite(): asks the model every question and records the raw transcripts. It never scores.

FLOW
====
  1. run_suite()      -> raw transcripts (question, model answer, answer key), no verdict
  2. judging          -> a separate step (testing/judge.py + testing/judging.py): an AI judge on any provider row,
                         and/or a human in the UI, reads each saved case and decides pass / partial / fail
  3. score_results()  -> aggregates whatever verdicts exist; unjudged cases are counted as "awaiting"

Nothing here compares strings to decide correctness. The one exception is the official public benchmarks
(MMLU/GSM8K ...), which keep their standard exact-match scoring in testing/strict_scoring.py.

Suite JSON format (v2):
[
  {
    "name": "lora_definition",
    "category": "knowledge",
    "question": "What is LoRA and how does it work?",
    "correct_answer": "LoRA is a parameter-efficient training method that adds small adapter matrices...",
    "keywords": ["optional", "key values the judge should look for"],
    "expect_abstain": false
  }
]
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from finetune_studio.training.formatting import with_system_prompt

JudgeType = Literal["none", "ai", "human", "exact", "scripted"]
Verdict = Literal["pass", "fail", "partial", ""]


@dataclass
class BenchmarkCase:
    """A single test — question + known-correct answer."""
    name: str
    question: str
    correct_answer: str
    category: str = "general"
    context: str = ""  # optional extra context for the judge
    keywords: list[str] = field(default_factory=list)
    source_id: str = ""
    chunk_idx: int = 0
    row_index: int = -1  # 0-based line in the source dataset (full-coverage audit)
    # System turn the row was trained with (the retrieved CONTEXT of a grounded row). Such a row
    # teaches "answer from the context", so quizzing it bare measures the wrong skill.
    system_prompt: str = ""
    # An unanswerable question: the right behaviour is to say the documents do not cover it, so an abstention PASSES
    # and any confident answer FAILS (the opposite of every other case).
    expect_abstain: bool = False


@dataclass
class CaseResult:
    """Result of running + judging one case."""
    case_name: str
    category: str
    question: str
    correct_answer: str
    model_answer: str
    transcript: list = field(default_factory=list)  # [{role, content}, ...]
    judge: JudgeType = "none"
    judge_model: str = ""
    verdict: Verdict = ""
    judge_reasoning: str = ""
    time_ms: float = 0.0
    error: str = ""
    keywords: list[str] = field(default_factory=list)
    scoring_method: str = ""
    validity: str = ""
    source_id: str = ""
    chunk_idx: int = 0
    expect_abstain: bool = False


def load_test_suite(path: str) -> list[BenchmarkCase]:
    """Load a v2 Q&A benchmark suite from JSON.

    Accepts either a bare case list, or a versioned suite definition object
    with a ``cases`` array (built-in synthetic or local evaluation fixtures).
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        raw_cases = data.get("cases")
        if not isinstance(raw_cases, list):
            raise TypeError(
                f"suite definition at {path!r} must contain a 'cases' list"
            )
        data = raw_cases
    if not isinstance(data, list):
        raise TypeError(f"suite at {path!r} must be a JSON list or definition")
    cases = []
    for item in data:
        # Support both v2 (question/correct_answer) and v1 (messages/expected_keywords)
        if "question" in item:
            kws = item.get("keywords") or item.get("expected_keywords") or []
            if not isinstance(kws, list):
                kws = []
            cases.append(BenchmarkCase(
                name=item["name"],
                question=item["question"],
                correct_answer=item.get("correct_answer", ""),
                category=item.get("category", "general"),
                context=item.get("context", ""),
                keywords=[str(k) for k in kws],
                source_id=str(item.get("source_id") or ""),
                chunk_idx=int(item.get("chunk_idx") or 0),
                row_index=int(item.get("row_index", -1)),
                system_prompt=str(item.get("system_prompt") or ""),
                expect_abstain=bool(item.get("expect_abstain")),
            ))
        elif "messages" in item:
            # v1 fallback: extract from messages format
            msgs = item["messages"]
            user_msg = next((m for m in msgs if m.get("role") == "user"), None)
            assistant_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
            question = user_msg["content"] if user_msg else ""
            kws = item.get("expected_keywords") or item.get("keywords") or []
            if not isinstance(kws, list):
                kws = []
            correct = assistant_msg["content"] if assistant_msg else (kws[0] if kws else "")
            cases.append(BenchmarkCase(
                name=item["name"],
                question=question,
                correct_answer=correct,
                category=item.get("category", "general"),
                keywords=[str(k) for k in kws],
                source_id=str(item.get("source_id") or ""),
                chunk_idx=int(item.get("chunk_idx") or 0),
                row_index=int(item.get("row_index", -1)),
            ))
    return cases


def run_case(engine, case: BenchmarkCase, *, max_tokens: int = 512, temperature: float = 0.3,
             think: bool = False, system_prompt: str = "") -> CaseResult:
    """Ask the model one question and record the raw transcript. No verdict."""
    start = time.time()
    messages = with_system_prompt([{"role": "user", "content": case.question}],
                                  case.system_prompt or system_prompt)
    common = {
        "case_name": case.name, "category": case.category, "question": case.question,
        "correct_answer": case.correct_answer, "keywords": list(case.keywords), "source_id": case.source_id,
        "chunk_idx": case.chunk_idx, "expect_abstain": case.expect_abstain,
    }
    try:
        response = engine.generate(messages, max_tokens=max_tokens, temperature=temperature, think=think)
    except Exception as e:  # noqa: BLE001
        return CaseResult(
            model_answer="", transcript=[*messages, {"role": "assistant", "content": ""}],
            error=str(e), time_ms=round((time.time() - start) * 1000, 1), **common,
        )
    return CaseResult(
        model_answer=response, transcript=[*messages, {"role": "assistant", "content": response}],
        time_ms=round((time.time() - start) * 1000, 1), **common,
    )


def run_suite(engine, cases: list[BenchmarkCase], max_tokens: int = 512,
              temperature: float = 0.3, think: bool = False,
              system_prompt: str = "", *,
              on_result: Callable[[CaseResult], None] | None = None,
              should_stop: Callable[[], bool] | None = None) -> list[CaseResult]:
    """Run each case through the model and collect transcripts. Judging is a separate step.

    ``on_result`` is called as each case finishes (the test job saves it at once, so a crash loses nothing);
    ``should_stop`` ends the run early between cases.
    """
    results: list[CaseResult] = []
    for case in cases:
        if should_stop is not None and should_stop():
            break
        result = run_case(engine, case, max_tokens=max_tokens, temperature=temperature,
                          think=think, system_prompt=system_prompt)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def score_results(results: list[CaseResult]) -> dict:
    """Aggregate stats over whatever verdicts exist.

    A case without a verdict is *awaiting* judgement (``awaiting``/``unjudged``): it is not a failure, but it
    still counts in ``total`` — ``pass_rate`` and ``weighted_score`` are shares of ALL cases, so a half-judged
    run (or one where cases errored) never reports an inflated score. ``pass_rate`` is ``None`` while nothing is
    judged, because a rate computed from zero verdicts is not a measurement. ``by_judge`` counts verdicts per
    source (human / ai / exact / scripted) so a score shows what it rests on.
    """
    judged = [r for r in results if r.verdict]
    total = len(results)
    n_judged = len(judged)
    passed = sum(1 for r in judged if r.verdict == "pass")
    partial = sum(1 for r in judged if r.verdict == "partial")
    failed = sum(1 for r in judged if r.verdict == "fail")
    unjudged = total - n_judged
    avg_time = (sum(r.time_ms for r in results) / max(total, 1)) if total else 0

    # pass=1.0, partial=0.5, fail=0.0, awaiting=0.0 over ALL cases.
    weighted = (passed * 1.0 + partial * 0.5) / max(total, 1) * 100 if total else 0

    cats: dict[str, dict] = {}
    for r in judged:
        if r.category not in cats:
            cats[r.category] = {"passed": 0, "total": 0}
        cats[r.category]["total"] += 1
        if r.verdict == "pass":
            cats[r.category]["passed"] += 1

    by_judge: dict[str, int] = {}
    for r in judged:
        by_judge[r.judge or "none"] = by_judge.get(r.judge or "none", 0) + 1

    return {
        "total": total,
        "judged": n_judged,
        "unjudged": unjudged,
        "awaiting": unjudged,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "pass_rate": round(passed / max(total, 1) * 100, 1) if n_judged else None,
        "weighted_score": round(weighted, 1) if n_judged else None,
        "avg_time_ms": round(avg_time, 1),
        "categories": cats,
        "by_judge": by_judge,
    }
