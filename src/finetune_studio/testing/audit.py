"""Independent audit helpers for persisted benchmark runs."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from finetune_studio.testing.suite import (
    CaseResult,
    apply_heuristic_judging,
    score_results,
)


def recompute_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-score persisted raw cases without trusting stored verdict fields."""
    results: list[CaseResult] = []
    malformed: list[str] = []
    for case in cases:
        transcript = case.get("transcript", [])
        if isinstance(transcript, str):
            try:
                transcript = json.loads(transcript)
            except json.JSONDecodeError:
                malformed.append(str(case.get("id", case.get("case_name", ""))))
                transcript = []
        results.append(CaseResult(
            case_name=str(case.get("case_name") or case.get("name") or ""),
            category=str(case.get("category") or "general"),
            question=str(case.get("question") or ""),
            correct_answer=str(case.get("correct_answer") or ""),
            model_answer=str(case.get("model_answer") or ""),
            transcript=transcript if isinstance(transcript, list) else [],
            error=str(case.get("error") or ""),
            keywords=[str(k) for k in case.get("keywords", []) or []],
        ))
    apply_heuristic_judging(results)
    recomputed = score_results(results)
    disagreements = []
    for original, current in zip(cases, results):
        stored = str(original.get("verdict") or "")
        if stored != current.verdict:
            disagreements.append({
                "id": original.get("id", original.get("case_name", "")),
                "stored": stored,
                "recomputed": current.verdict,
                "stored_reasoning": original.get("judge_reasoning", ""),
                "recomputed_reasoning": current.judge_reasoning,
            })
    return {
        "case_count": len(cases),
        "malformed_transcripts": malformed,
        "stored_verdict_counts": dict(Counter(str(c.get("verdict") or "") for c in cases)),
        "recomputed_scores": recomputed,
        "disagreements": disagreements,
        "passed": not malformed and not disagreements,
    }
