"""Test suite — Q&A benchmark cases with AI/human judging.

WHAT THIS FILE DOES
==================
Defines benchmark cases as Q&A pairs (not keyword matching):
  - BenchmarkCase: a question + the correct answer from training data
  - CaseResult: the model's response + judge verdict + reasoning
  - SuiteRunner: executes a suite, collects transcripts, then judges

KEY CONCEPTS
============
- Each case is a Q&A pair — we know the correct answer because we created
  the training data (or extracted it from existing data).
- The model under test is asked the question. Its full transcript is saved.
- A judge (AI model or human) evaluates: did the model answer correctly?
- Cases start with judge="none". The judge step runs separately so you
  can use a powerful external model (GPT-4, Claude) to score cheaply.

FLOW
====
  1. run_suite() → ask model each question, store transcript + model answer
  2. judge_suite() → send each transcript to an AI judge (or queue for human)
  3. score_results() → aggregate pass/fail/partial + judge confidence

Suite JSON format (v2):
[
  {
    "name": "lora_definition",
    "category": "knowledge",
    "question": "What is LoRA and how does it work?",
    "correct_answer": "LoRA is a parameter-efficient training method that adds small adapter matrices...",
    "context": "optional extra context for the judge"
  }
]
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal

JudgeType = Literal["none", "ai", "human", "heuristic", "local"]
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
            ))
    return cases


def extract_answer(transcript: list) -> str:
    """Pull the last assistant message from a transcript as the model's answer."""
    for msg in reversed(transcript):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def run_suite(engine, cases: list[BenchmarkCase], max_tokens: int = 512,
              temperature: float = 0.3, think: bool = False) -> list[CaseResult]:
    """Run each case through the model. No judging yet — just collect transcripts."""
    results = []
    for case in cases:
        start = time.time()
        try:
            messages = [{"role": "user", "content": case.question}]
            response = engine.generate(messages, max_tokens=max_tokens,
                                       temperature=temperature, think=think)
            elapsed_ms = (time.time() - start) * 1000
            # Build transcript: user question + model answer
            transcript = messages + [{"role": "assistant", "content": response}]
            results.append(CaseResult(
                case_name=case.name,
                category=case.category,
                question=case.question,
                correct_answer=case.correct_answer,
                model_answer=response,
                transcript=transcript,
                time_ms=round(elapsed_ms, 1),
                keywords=list(case.keywords),
                source_id=case.source_id,
                chunk_idx=case.chunk_idx,
            ))
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.time() - start) * 1000
            results.append(CaseResult(
                case_name=case.name,
                category=case.category,
                question=case.question,
                correct_answer=case.correct_answer,
                model_answer="",
                transcript=[{"role": "user", "content": case.question},
                           {"role": "assistant", "content": ""}],
                error=str(e),
                time_ms=round(elapsed_ms, 1),
                keywords=list(case.keywords),
                source_id=case.source_id,
                chunk_idx=case.chunk_idx,
            ))
    return results


def apply_heuristic_judging(results: list[CaseResult]) -> None:
    """Mutate results in place: set verdict/judge via strict or legacy scoring.

    Prefer task-aware strict scoring for multiple-choice and numeric cases
    (exact option / normalized final number; rejects wrong extras). Open-ended
    cases keep keyword matching when ``keywords`` is non-empty, else
    ``judge_case_heuristic``.

    Skips cases that already have a verdict or that errored with an empty answer.
    """
    from finetune_studio.testing.judge import judge_case_heuristic
    from finetune_studio.testing.strict_scoring import (
        score_source_grounded,
        score_strict,
    )

    for r in results:
        if r.verdict:
            continue
        if r.error and not r.model_answer:
            continue

        if r.source_id:
            source_score = score_source_grounded(
                correct_answer=r.correct_answer,
                model_answer=r.model_answer,
            )
            if source_score is not None:
                r.verdict = source_score.verdict
                r.judge = "heuristic"
                r.judge_model = "heuristic"
                r.scoring_method = source_score.scoring_method
                r.validity = source_score.validity
                r.judge_reasoning = (
                    f"[{source_score.scoring_method}; validity={source_score.validity}] "
                    f"{source_score.reasoning}"
                )
                continue

        strict = score_strict(
            question=r.question,
            correct_answer=r.correct_answer,
            model_answer=r.model_answer,
        )
        if strict is not None:
            r.verdict = strict.verdict
            r.judge = "heuristic"
            r.judge_model = "heuristic"
            r.scoring_method = strict.scoring_method
            r.validity = strict.validity
            r.judge_reasoning = (
                f"[{strict.scoring_method}; validity={strict.validity}] "
                f"{strict.reasoning}"
            )
            continue

        if r.keywords:
            lower = (r.model_answer or "").lower()
            hits = [k for k in r.keywords if k.lower() in lower]
            n = len(r.keywords)
            n_hits = len(hits)
            if n_hits == n and n > 0:
                r.verdict = "pass"
            elif n_hits > 0:
                r.verdict = "partial"
            else:
                r.verdict = "fail"
            r.judge = "heuristic"
            r.judge_model = "heuristic"
            r.scoring_method = "keyword_substring"
            r.validity = "valid" if n_hits == n else ("partial" if n_hits else "valid")
            r.judge_reasoning = (
                f"[keyword_substring; validity={r.validity}] "
                f"keywords matched {n_hits}/{n}"
            )
            continue
        verdict, reasoning, _conf = judge_case_heuristic(
            question=r.question,
            correct_answer=r.correct_answer,
            model_answer=r.model_answer,
        )
        if not verdict:
            continue
        r.verdict = verdict
        r.judge = "heuristic"
        r.judge_model = "heuristic"
        r.scoring_method = "heuristic_overlap"
        r.validity = "valid"
        r.judge_reasoning = f"[heuristic_overlap; validity=valid] {reasoning}"


def score_results(results: list[CaseResult]) -> dict:
    """Aggregate stats over judged results. Only counts cases with a verdict."""
    judged = [r for r in results if r.verdict]
    total = len(results)
    n_judged = len(judged)
    passed = sum(1 for r in judged if r.verdict == "pass")
    partial = sum(1 for r in judged if r.verdict == "partial")
    failed = sum(1 for r in judged if r.verdict == "fail")
    unjudged = total - n_judged
    avg_time = (sum(r.time_ms for r in results) / max(total, 1)) if total else 0

    # Compute weighted score: pass=1.0, partial=0.5, fail=0.0
    weighted = (passed * 1.0 + partial * 0.5) / max(n_judged, 1) * 100 if n_judged else 0

    # Category breakdown
    cats: dict[str, dict] = {}
    for r in judged:
        if r.category not in cats:
            cats[r.category] = {"passed": 0, "total": 0}
        cats[r.category]["total"] += 1
        if r.verdict == "pass":
            cats[r.category]["passed"] += 1

    return {
        "total": total,
        "judged": n_judged,
        "unjudged": unjudged,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "pass_rate": round(passed / max(n_judged, 1) * 100, 1) if n_judged else 0,
        "weighted_score": round(weighted, 1),
        "avg_time_ms": round(avg_time, 1),
        "categories": cats,
    }
