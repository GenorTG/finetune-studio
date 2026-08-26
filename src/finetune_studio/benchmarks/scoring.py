"""Scoring heuristics for model outputs.

WHAT THIS FILE DOES
==================
Implements various ways to score a model's response:
  - Keyword matching: does the response contain expected keywords?
  - Forbidden words: does it contain words it shouldn't?
  - Length scoring: is the response the right length?
  - Truthfulness scoring: are the claims factually accurate?

KEY CONCEPTS
============
- Weighted scoring: different criteria have different weights
  (e.g., keyword match is 70%, length is 20%, forbidden is 10%).
- Fuzzy matching: keywords can match with case differences or partial
  matches (e.g., "Python" matches "python").
- Penalties vs bonuses: correct keywords ADD to the score, forbidden
  words SUBTRACT.
"""

"""Benchmark scoring module for Finetune Studio WebUI."""
import re
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    benchmark: str
    question: str
    prediction: str
    expected: str
    correct: bool
    score: float = 0.0
    details: dict = field(default_factory=dict)


class BenchmarkScorer:
    """Industry-standard scoring for LLM benchmarks."""

    def score_mcq(self, response: str, expected: str, choices: list | None = None) -> dict:
        """Score multiple-choice question response."""
        pred_letter = self.extract_mcq_letter(response, choices)
        correct = pred_letter == expected
        return {
            "correct": correct,
            "prediction": pred_letter,
            "expected": expected,
            "method": "mcq_extraction",
        }

    def extract_mcq_letter(self, response: str, choices: list | None = None) -> str:
        """Extract chosen letter from MCQ response using industry-standard patterns."""
        response = response.strip()

        # Pattern 1: Direct letter at start
        match = re.match(r'^([A-D])\b', response)
        if match:
            return match.group(1)

        # Pattern 2: Letter with delimiters
        match = re.match(r'^([A-D])[\)\.\:\s]', response)
        if match:
            return match.group(1)

        # Pattern 3: "answer is X"
        match = re.search(r'(?:the answer is|answer is|answer:?)\s*([A-D])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 4: "X)" or "X." anywhere
        match = re.search(r'\b([A-D])[\)\.]', response)
        if match:
            return match.group(1).upper()

        # Pattern 5: Just the letter
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return match.group(1).upper()

        # Pattern 6: Choice text match
        if choices:
            for i, choice in enumerate(choices):
                if choice.lower() in response.lower():
                    return chr(65 + i)

        return ""

    def score_math(self, response: str, expected: str) -> dict:
        """Score math response with normalization."""
        pred = self.extract_math_answer(response)
        pred_norm = self.normalize_number(pred)
        expected_norm = self.normalize_number(expected)
        correct = pred_norm == expected_norm
        return {
            "correct": correct,
            "prediction": pred,
            "expected": expected,
            "method": "math_extraction",
        }

    def extract_math_answer(self, response: str) -> str:
        """Extract numeric answer from math response."""
        response = response.strip()

        # After #### marker
        if "####" in response:
            after = response.split("####")[-1].strip()
            nums = re.findall(r'[-+]?\d+\.?\d*', after)
            if nums:
                return nums[-1].replace(",", "")

        # Boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            return match.group(1).replace(",", "")

        # "answer is X"
        match = re.search(r'(?:answer is|=)\s*([-+]?\d+\.?\d*)', response, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")

        # Last number
        nums = re.findall(r'[-+]?\d+\.?\d*', response)
        if nums:
            return nums[-1].replace(",", "")

        return ""

    def normalize_number(self, num_str: str) -> str:
        """Normalize number for comparison."""
        if not num_str:
            return ""
        num_str = num_str.replace(",", "").replace(" ", "").replace("$", "")
        try:
            return str(float(num_str))
        except ValueError:
            return num_str

    def score_truthful(self, response: str, correct_answers: list, incorrect_answers: list) -> dict:
        """Score TruthfulQA response."""
        response_lower = response.lower().strip()

        matches_correct = [ca for ca in correct_answers if ca.lower() in response_lower]
        matches_incorrect = [ia for ia in incorrect_answers if ia.lower() in response_lower]

        is_truthful = len(matches_correct) > 0 and len(matches_incorrect) == 0

        return {
            "correct": is_truthful,
            "matches_correct": matches_correct,
            "matches_incorrect": matches_incorrect,
            "method": "truthfulqa_keywords",
        }

    def score_winogrande(self, response: str, option1: str, option2: str) -> dict:
        """Score Winogrande response."""
        response = response.strip()

        if response.startswith("1"):
            pred = "1"
        elif response.startswith("2"):
            pred = "2"
        elif option1.lower() in response.lower():
            pred = "1"
        elif option2.lower() in response.lower():
            pred = "2"
        else:
            match = re.search(r'\b(1|2)\b', response)
            pred = match.group(1) if match else ""

        return {
            "prediction": pred,
            "method": "winogrande_extraction",
        }

    def score_open_ended(self, response: str, reference: str, keywords: list | None = None,
                         forbidden: list | None = None) -> dict:
        """Score open-ended response against reference."""
        response_lower = response.lower()

        # Keyword scoring
        keyword_hits = []
        keyword_misses = []
        if keywords:
            for kw in keywords:
                if kw.lower() in response_lower:
                    keyword_hits.append(kw)
                else:
                    keyword_misses.append(kw)
            keyword_score = len(keyword_hits) / len(keywords) if keywords else 1.0
        else:
            keyword_score = 1.0

        # Forbidden penalty
        forbidden_hits = []
        if forbidden:
            for kw in forbidden:
                if kw.lower() in response_lower:
                    forbidden_hits.append(kw)
            forbidden_penalty = len(forbidden_hits) / len(forbidden) if forbidden else 0
        else:
            forbidden_penalty = 0

        # Length scoring
        length = len(response.split())
        if length < 3:
            length_score = 0.2
        elif length < 10:
            length_score = 0.8
        elif length < 200:
            length_score = 1.0
        else:
            length_score = 0.7

        total = max(0, keyword_score * 0.6 + length_score * 0.3 - forbidden_penalty * 0.4)

        return {
            "correct": total >= 0.5,
            "score": round(total, 3),
            "keyword_score": round(keyword_score, 3),
            "length_score": round(length_score, 3),
            "forbidden_hits": forbidden_hits,
            "keyword_hits": keyword_hits,
            "keyword_misses": keyword_misses,
            "method": "open_ended_keywords",
        }


# Singleton instance
scorer = BenchmarkScorer()
