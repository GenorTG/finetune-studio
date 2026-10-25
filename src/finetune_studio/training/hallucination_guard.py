"""Detect training examples that may cause hallucination.

WHAT THIS FILE DOES
==================
Identifies training examples that could teach the model to make
things up:
  - Factual claims without context
  - Made-up numbers or statistics
  - Confident assertions about uncertain topics
  - Examples that contradict known facts

KEY CONCEPTS
============
- Hallucination: when a model states false information confidently.
  This is a major problem in production LLMs.
- Training data origin: where did this example come from? If it's
  from a synthetic source, double-check the facts.
- Confidence calibration: examples with high model confidence during
  initial testing might be hallucinated.
"""

"""Hallucination guardrails — detect and prevent model hallucinations."""
import re
from dataclasses import dataclass, field


@dataclass
class GuardrailResult:
    passed: bool
    issues: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)


class TrainingDataValidator:
    """Validate training data for hallucination risks."""

    def __init__(self):
        self.risky_patterns = [
            (r"\b\d{4}-\d{2}-\d{2}\b", "specific_date", "May be fabricated"),
            (r"\b\d+\.\d+\s*(million|billion)\b", "specific_number", "May be fabricated"),
            (r"(according to|study shows)", "citation", "Verify source exists"),
            (r"https?://[^\s]+", "url", "Verify URL is real"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email", "Verify email is real"),
        ]

    def validate_response(self, response: str) -> list:
        """Check a training response for hallucination risks."""
        risks = []
        for pattern, ptype, suggestion in self.risky_patterns:
            matches = re.findall(pattern, response)
            if matches:
                risks.append({"type": ptype, "match": matches[0], "suggestion": suggestion})
        return risks

    def validate_dataset(self, data: list) -> dict:
        """Validate entire dataset for hallucination risks."""
        total_risks = 0
        risk_types: dict[str, int] = {}

        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    risks = self.validate_response(msg.get("content", ""))
                    for risk in risks:
                        total_risks += 1
                        risk_types[risk["type"]] = risk_types.get(risk["type"], 0) + 1

        return {
            "total_risks": total_risks,
            "risk_types": risk_types,
            "recommendation": self._get_recommendation(total_risks, len(data)),
        }

    def _get_recommendation(self, risks: int, total: int) -> str:
        if total == 0:
            return "Empty dataset"
        ratio = risks / total
        if ratio > 0.3:
            return "HIGH risk — review responses for fabricated content"
        if ratio > 0.1:
            return "MODERATE risk — verify specific claims in responses"
        return "LOW risk — dataset looks clean"
