"""Weighted scoring for model comparisons.

WHAT THIS FILE DOES
==================
Takes comparison results and computes a score for each model based
on multiple criteria:
  - Time to respond (faster = better)
  - Keyword match percentage
  - Length appropriateness
  - Forbidden word penalty

KEY CONCEPTS
============
- Weighted average: combine multiple scores into one final score.
- Normalization: different criteria have different scales (time is
  in seconds, length is in characters). We normalize to 0-1.
- User-configurable weights: users can adjust the importance of
  each criterion via config.
"""

"""Scorer — score comparison results with multiple metrics."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreResult:
    test_name: str = ""
    source_name: str = ""
    response: str = ""
    keyword_score: float = 0.0
    forbidden_penalty: float = 0.0
    length_score: float = 0.0
    time_ms: float = 0.0
    total_score: float = 0.0
    passed: bool = False
    details: dict = field(default_factory=dict)


class Scorer:
    """Score comparison results."""

    def __init__(self, keyword_weight: float = 0.5, length_weight: float = 0.2,
                 time_weight: float = 0.1, forbidden_weight: float = 0.2):
        self.keyword_weight = keyword_weight
        self.length_weight = length_weight
        self.time_weight = time_weight
        self.forbidden_weight = forbidden_weight

    def score_keyword_match(self, response: str, expected: list, forbidden: list) -> tuple:
        """Score keyword presence/absence."""
        resp_lower = response.lower()

        if expected:
            hits = sum(1 for k in expected if k.lower() in resp_lower)
            keyword_score = hits / len(expected)
        else:
            keyword_score = 1.0  # No keywords to check

        if forbidden:
            forbidden_hits = sum(1 for k in forbidden if k.lower() in resp_lower)
            forbidden_penalty = forbidden_hits / len(forbidden)
        else:
            forbidden_penalty = 0.0

        return keyword_score, forbidden_penalty

    def score_length(self, response: str, ideal_length: int = 200) -> float:
        """Score response length (penalize too short or too long)."""
        length = len(response)
        if length == 0:
            return 0.0

        ratio = length / ideal_length
        if ratio < 0.1:
            return 0.2
        elif ratio < 0.5:
            return 0.6
        elif ratio <= 2.0:
            return 1.0
        elif ratio <= 3.0:
            return 0.7
        else:
            return 0.4

    def score_time(self, time_ms: float, baseline_ms: float = 1000) -> float:
        """Score response time (faster is better, but not suspiciously fast)."""
        if time_ms <= 0:
            return 0.0
        ratio = baseline_ms / time_ms
        if ratio > 3:
            return 0.8  # Suspiciously fast
        elif ratio > 1.5:
            return 1.0
        elif ratio > 0.5:
            return 0.8
        else:
            return 0.5

    def score_response(self, test_name: str, source_name: str, response: str,
                       expected: list, forbidden: list, time_ms: float,
                       ideal_length: int = 200) -> ScoreResult:
        """Score a single response."""
        keyword_score, forbidden_penalty = self.score_keyword_match(
            response, expected, forbidden
        )
        length_score = self.score_length(response, ideal_length)
        time_score = self.score_time(time_ms)

        total = (
            keyword_score * self.keyword_weight +
            length_score * self.length_weight +
            time_score * self.time_weight -
            forbidden_penalty * self.forbidden_weight
        )
        total = max(0.0, min(1.0, total))

        return ScoreResult(
            test_name=test_name,
            source_name=source_name,
            response=response,
            keyword_score=round(keyword_score, 3),
            forbidden_penalty=round(forbidden_penalty, 3),
            length_score=round(length_score, 3),
            time_ms=round(time_ms, 1),
            total_score=round(total, 3),
            passed=total >= 0.5 and forbidden_penalty == 0,
            details={
                "keyword_hits": [k for k in expected if k.lower() in response.lower()],
                "keyword_misses": [k for k in expected if k.lower() not in response.lower()],
                "forbidden_hits": [k for k in forbidden if k.lower() in response.lower()],
            },
        )

    def score_comparison(self, comparison_results: list[dict]) -> dict:
        """Score all comparison results."""
        all_scores = []

        for test in comparison_results:
            for source_name, responses in test["responses"].items():
                for i, result in enumerate(responses):
                    if result["error"]:
                        all_scores.append(ScoreResult(
                            test_name=test["name"],
                            source_name=source_name,
                            response="",
                            total_score=0.0,
                            passed=False,
                            details={"error": result["error"]},
                        ))
                    else:
                        score = self.score_response(
                            test["name"],
                            source_name,
                            result["response"],
                            test["expected_keywords"],
                            test["forbidden_keywords"],
                            result["time_ms"],
                        )
                        all_scores.append(score)

        # Aggregate by source
        by_source: dict[str, dict[str, Any]] = {}
        for score in all_scores:
            if score.source_name not in by_source:
                by_source[score.source_name] = {
                    "scores": [],
                    "total": 0,
                    "passed": 0,
                    "avg_score": 0,
                    "avg_time_ms": 0,
                }
            by_source[score.source_name]["scores"].append(score)
            by_source[score.source_name]["total"] += 1
            if score.passed:
                by_source[score.source_name]["passed"] += 1

        for source_name, data in by_source.items():
            scores = data["scores"]
            data["avg_score"] = round(
                sum(s.total_score for s in scores) / max(len(scores), 1), 3
            )
            data["avg_time_ms"] = round(
                sum(s.time_ms for s in scores) / max(len(scores), 1), 1
            )
            data["pass_rate"] = round(
                data["passed"] / max(data["total"], 1) * 100, 1
            )

        return {
            "all_scores": all_scores,
            "by_source": by_source,
        }
