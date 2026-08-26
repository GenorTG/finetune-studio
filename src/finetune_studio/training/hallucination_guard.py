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


class HallucinationGuardrail:
    """Post-processing guardrails to detect hallucinations."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._load_patterns()

    def _load_patterns(self):
        """Load patterns that indicate potential hallucinations."""
        self.hallucination_patterns = [
            # Confidence without evidence
            (r"\b(definitely|certainly|absolutely|100%)\b", "high_confidence"),
            # Fabricated specifics
            (r"\b\d{4}-\d{2}-\d{2}\b", "specific_date"),
            (r"\b\d+\.\d+\s*(million|billion|trillion)\b", "specific_number"),
            # Fabricated citations
            (r"(according to|as reported by|study shows|research indicates)", "citation"),
            # Hedging on facts
            (r"\b(might be|could be|possibly|perhaps)\b", "hedging"),
        ]

        self.refusal_patterns = [
            (r"\b(I don't know|I'm not sure|I can't|I don't have)\b", "refusal"),
            (r"\b(outside my scope|not my area|beyond my)\b", "scope_refusal"),
        ]

        self.confidence_markers = [
            "is", "are", "was", "were", "will", "has", "have",
        ]

    def check_response(self, question: str, response: str,
                       context: list | None = None) -> GuardrailResult:
        """Check a response for potential hallucinations."""
        issues = []
        suggestions = []

        # Check for hallucination patterns
        for pattern, ptype in self.hallucination_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                issues.append(f"Potential {ptype}: {matches[0]}")

        # Check if response is too short (possible refusal)
        if len(response.split()) < 3:
            issues.append("Very short response — possible refusal or empty")
            suggestions.append("Check if the model refused to answer")

        # Check if response matches question topic
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        overlap = question_words & response_words
        if len(overlap) < 2 and len(question.split()) > 5:
            issues.append("Response may not address the question")
            suggestions.append("Verify the response is relevant")

        # Check for fabricated URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, response)
        if urls:
            issues.append(f"Contains URLs that may be fabricated: {urls[0]}")
            suggestions.append("Verify URLs are real")

        # Check for fabricated emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, response)
        if emails:
            issues.append(f"Contains email addresses that may be fabricated: {emails[0]}")
            suggestions.append("Verify emails are real")

        # Overall assessment
        passed = len(issues) == 0
        if len(issues) > 2:
            suggestions.append("Consider rephrasing or adding more context")

        return GuardrailResult(passed=passed, issues=issues, suggestions=suggestions)

    def check_batch(self, qa_pairs: list) -> dict:
        """Check a batch of QA pairs for hallucinations."""
        results = []
        for q, a in qa_pairs:
            result = self.check_response(q, a)
            results.append({"question": q, "answer": a[:200],
                          "passed": result.passed, "issues": result.issues})

        passed = sum(1 for r in results if r["passed"])
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / max(len(results), 1) * 100, 1),
            "results": results,
        }


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
