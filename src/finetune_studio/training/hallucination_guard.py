"""Flag training answers that contain claims worth checking by hand (`fts validate-hallucination`).

A pattern scan, not a fact check: it counts answers with an ISO date, a "N.N million/billion" figure, a
"according to" / "study shows" citation, a URL or an e-mail address, so a reviewer knows where to look. It
cannot tell a true value from an invented one; a dataset of real company facts is full of dates and
scores high on purpose. Whether a model learned or invents facts is measured on the Testing page.
"""
import re


class TrainingDataValidator:
    """Count checkable claims (dates, figures, citations, URLs, e-mails) in assistant answers."""

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
            return "MANY answers with dates, figures, citations, URLs or e-mails: check them against the sources"
        if ratio > 0.1:
            return "SOME answers with dates, figures, citations, URLs or e-mails: spot-check them"
        return "FEW answers with checkable claims (this is a pattern scan, not a fact check)"
