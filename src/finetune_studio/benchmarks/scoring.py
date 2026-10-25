"""Exact-match scoring for the public multiple-choice / numeric benchmarks (MMLU, GSM8K, HellaSwag ...).

Open-ended answers are never scored here: whether a model's own-data answer is correct is decided afterwards by an
AI judge or a person (``finetune_studio.testing.judge``), not by matching keywords.
"""

import re


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


# Singleton instance
scorer = BenchmarkScorer()
