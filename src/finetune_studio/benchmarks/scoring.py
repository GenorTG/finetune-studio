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


# Singleton instance
scorer = BenchmarkScorer()
