"""Aggregate scores of a saved test run, recomputed from the verdicts that exist on its cases.

Every verdict change — an AI judge finishing a case, a human overriding one — goes through
:func:`rescore_benchmark`, so the headline score on the Testing page never goes stale.
"""
from __future__ import annotations

from typing import Any

from finetune_studio import db
from finetune_studio.testing.suite import CaseResult, score_results

# Case rows written before judgements were tracked name the old matcher "heuristic"/"local".
_JUDGE_KIND = {"heuristic": "scripted", "local": "ai"}


def case_result_from_row(row: dict[str, Any]) -> CaseResult:
    """A ``CaseResult`` view of a stored case row (verdict and judge included)."""
    judge = str(row.get("judge") or "none")
    return CaseResult(
        case_name=row.get("case_name") or row.get("name") or "",
        category=row.get("category") or "general",
        question=row.get("question") or "",
        correct_answer=row.get("correct_answer") or "",
        model_answer=row.get("model_answer") or "",
        transcript=row.get("transcript") or [],
        judge=_JUDGE_KIND.get(judge, judge),  # type: ignore[arg-type]
        judge_model=row.get("judge_model") or "",
        verdict=row.get("verdict") or "",
        judge_reasoning=row.get("judge_reasoning") or "",
        time_ms=float(row.get("time_ms") or 0),
        error=row.get("error") or "",
        keywords=list(row.get("keywords") or []),
        expect_abstain=bool(row.get("expect_abstain")),
    )


def rescore_benchmark(bid: str) -> dict[str, Any]:
    """Recompute and persist a run's aggregate scores from its stored case verdicts.

    Non-score metadata already in the scores (``eval_kind``, ``retrieval``, ``model_path`` ...) is kept.
    """
    results = [case_result_from_row(c) for c in db.list_cases(bid)]
    old = (db.get_benchmark(bid) or {}).get("scores") or {}
    scores = {**(old if isinstance(old, dict) else {}), **score_results(results)}
    db.update_benchmark_scores(bid, scores)
    return scores


def rescore_legacy_benchmarks() -> int:
    """Startup: runs saved before verdict sources were tracked get ``by_judge``/``awaiting`` so the UI can say
    where each verdict came from (old scripted matcher vs AI vs human). Returns how many runs were rescored."""
    n = 0
    for bench in db.list_benchmarks():
        scores = bench.get("scores")
        if isinstance(scores, dict) and "by_judge" not in scores and bench.get("scoring") != "exact":
            rescore_benchmark(bench["id"])
            n += 1
    return n
