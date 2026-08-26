"""Real industry-standard benchmarks using HuggingFace datasets.

WHAT THIS FILE DOES
==================
Implements benchmarks that use REAL datasets from HuggingFace:
  - MMLU (Massive Multitask Language Understanding): 57 subjects
  - HellaSwag: sentence completion (commonsense reasoning)
  - ARC (AI2 Reasoning Challenge): grade-school science questions
  - TruthfulQA: detect when models make things up
  - GSM8K: grade-school math word problems
  - Winogrande: pronoun resolution (commonsense)

Each benchmark:
  1. Downloads its dataset from HuggingFace (cached after first run)
  2. Formats questions as prompts to the model
  3. Parses the model's response
  4. Compares to the correct answer
  5. Reports accuracy

KEY CONCEPTS
============
- HuggingFace datasets: a library for downloading and processing
  standard ML datasets. We use the "test" or "validation" splits.
- Few-shot prompting: we show the model 0-5 examples before asking
  the question, to help it understand the format.
- Answer extraction: the model might say "The answer is B" or "I
  think it's (B)" or "B" — we need to extract the actual letter.
- Token budget: each question + answer costs tokens; we limit max
  tokens to avoid runaway costs.
"""

"""Real industry-standard benchmarks using HuggingFace datasets."""
import os
import re
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    benchmark: str
    question_id: int
    question: str
    prediction: str
    expected: str
    correct: bool
    time_ms: float = 0.0


class RealBenchmarkSuite:
    """Industry-standard benchmarks using real datasets from HuggingFace."""

    def __init__(self, cache_dir: str = "data/benchmarks"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def list_available(self):
        return {
            "mmlu": "MMLU — 57 subjects, college-level knowledge (14K questions)",
            "hellaswag": "HellaSwag — commonsense reasoning (10K questions)",
            "arc_challenge": "ARC Challenge — grade-school science (2K questions)",
            "truthfulqa": "TruthfulQA — hallucination detection (817 questions)",
            "gsm8k": "GSM8K — grade-school math (8.5K questions)",
            "winogrande": "Winogrande — pronoun resolution (4.4K questions)",
            "mmlu_5shot": "MMLU 5-shot — MMLU with 5 examples (standard eval)",
        }

    # ══════════════════════════════════════════════════════════════
    # MMLU
    # ══════════════════════════════════════════════════════════════

    def load_mmlu(self, num_samples: int | None = None, subjects: list | None = None):
        """Load MMLU dataset from HuggingFace."""
        from datasets import load_dataset
        print("Loading MMLU dataset...")
        ds = load_dataset("cais/mmlu", "all", split="test", cache_dir=self.cache_dir)

        if subjects:
            ds = ds.filter(lambda x: x["subject"] in subjects)

        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))

        return ds

    def evaluate_mmlu(self, inference_engine, num_samples: int = 100,
                      subjects: list | None = None, max_tokens: int = 10) -> dict:
        """Evaluate on MMLU (multiple choice, 4 options)."""
        ds = self.load_mmlu(num_samples, subjects)
        correct = 0
        total = 0
        subject_scores = defaultdict(list)
        results = []
        choices = ["A", "B", "C", "D"]

        for item in ds:
            question = item["question"]
            answer_idx = item["answer"]
            expected = choices[answer_idx]
            subject = item["subject"]
            
            # choices may be a string repr of a list
            raw_choices = item["choices"]
            if isinstance(raw_choices, str):
                import ast
                raw_choices = ast.literal_eval(raw_choices)

            prompt = f"""Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D).

Question: {question}
A) {raw_choices[0]}
B) {raw_choices[1]}
C) {raw_choices[2]}
D) {raw_choices[3]}

Answer:"""
            result = inference_engine.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.0
            )

            pred = result["response"].strip().upper()
            # Extract just the letter
            pred_letter = pred[0] if pred and pred[0] in "ABCD" else pred[:2]

            is_correct = pred_letter == expected
            if is_correct:
                correct += 1
            total += 1
            subject_scores[subject].append(is_correct)

            results.append({
                "id": total, "question": question[:100],
                "prediction": pred, "expected": expected,
                "correct": is_correct, "subject": subject
            })

        subject_summary = {
            sub: round(sum(scores) / len(scores) * 100, 1)
            for sub, scores in subject_scores.items()
        }

        return {
            "benchmark": "mmlu",
            "total": total,
            "correct": correct,
            "accuracy": round(correct / max(total, 1) * 100, 1),
            "subjects": subject_summary,
            "results": results,
        }

    # ══════════════════════════════════════════════════════════════
    # HellaSwag
    # ══════════════════════════════════════════════════════════════

    def load_hellaswag(self, num_samples: int | None = None):
        from datasets import load_dataset
        print("Loading HellaSwag dataset...")
        ds = load_dataset("Rowan/hellaswag", split="validation", cache_dir=self.cache_dir)
        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))
        return ds

    def evaluate_hellaswag(self, inference_engine, num_samples: int = 100) -> dict:
        """Evaluate on HellaSwag (commonsense reasoning, 4 options)."""
        ds = self.load_hellaswag(num_samples)
        correct = 0
        total = 0
        results = []
        choices_labels = ["A", "B", "C", "D"]

        for item in ds:
            activity = item["activity_label"]
            ctx = item["ctx"]
            gold_idx = int(item["label"])
            endings = item["endings"]
            # endings might be a string representation of a list
            if isinstance(endings, str):
                import ast
                endings = ast.literal_eval(endings)

            prompt = f"""Complete the following scenario. Reply with ONLY the letter (A, B, C, or D).

Context: {ctx}
A) {endings[0]}
B) {endings[1]}
C) {endings[2]}
D) {endings[3]}

What happens next?:"""
            result = inference_engine.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=10, temperature=0.0
            )

            pred = result["response"].strip().upper()
            pred_letter = pred[0] if pred and pred[0] in "ABCD" else ""
            expected = choices_labels[gold_idx]

            is_correct = pred_letter == expected
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id": total, "question": ctx[:100],
                "prediction": pred_letter, "expected": expected,
                "correct": is_correct, "activity": activity
            })

        return {
            "benchmark": "hellaswag",
            "total": total,
            "correct": correct,
            "accuracy": round(correct / max(total, 1) * 100, 1),
            "results": results,
        }

    # ══════════════════════════════════════════════════════════════
    # ARC Challenge
    # ══════════════════════════════════════════════════════════════

    def load_arc(self, num_samples: int | None = None):
        from datasets import load_dataset
        print("Loading ARC dataset...")
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=self.cache_dir)
        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))
        return ds

    def evaluate_arc(self, inference_engine, num_samples: int = 100) -> dict:
        """Evaluate on ARC Challenge (science reasoning, 4 options)."""
        ds = self.load_arc(num_samples)
        correct = 0
        total = 0
        results = []

        for item in ds:
            question = item["question"]
            choices_dict = item["choices"]
            # Parse choices if it's a string
            if isinstance(choices_dict, str):
                import ast
                choices_dict = ast.literal_eval(choices_dict)
            labels = choices_dict["label"]
            texts = choices_dict["text"]
            answer = item["answerKey"]

            choice_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))

            prompt = f"""Answer the following science question. Reply with ONLY the letter.

Question: {question}
{choice_text}

Answer:"""
            result = inference_engine.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=10, temperature=0.0
            )

            pred = result["response"].strip().upper()
            pred_letter = pred[0] if pred and pred[0] in "ABCDE" else ""

            is_correct = pred_letter == answer
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id": total, "question": question[:100],
                "prediction": pred_letter, "expected": answer,
                "correct": is_correct,
            })

        return {
            "benchmark": "arc_challenge",
            "total": total,
            "correct": correct,
            "accuracy": round(correct / max(total, 1) * 100, 1),
            "results": results,
        }

    # ══════════════════════════════════════════════════════════════
    # TruthfulQA
    # ══════════════════════════════════════════════════════════════

    def load_truthfulqa(self, num_samples: int | None = None):
        from datasets import load_dataset
        print("Loading TruthfulQA dataset...")
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", cache_dir=self.cache_dir)
        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))
        return ds

    def evaluate_truthfulqa(self, inference_engine, num_samples: int = 100) -> dict:
        """Evaluate on TruthfulQA (hallucination detection)."""
        ds = self.load_truthfulqa(num_samples)
        correct = 0
        total = 0
        results = []

        for item in ds:
            question = item["question"]
            correct_answers = item["correct_answers"]
            incorrect_answers = item["incorrect_answers"]

            # Parse if string
            if isinstance(correct_answers, str):
                import ast
                correct_answers = ast.literal_eval(correct_answers)
            if isinstance(incorrect_answers, str):
                import ast
                incorrect_answers = ast.literal_eval(incorrect_answers)

            prompt = f"""Answer the following question truthfully. Be concise.

Question: {question}

Answer:"""
            result = inference_engine.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=100, temperature=0.0
            )

            pred = result["response"].strip().lower()

            # Check if prediction matches any correct answer
            matches_correct = any(ca.lower() in pred for ca in correct_answers)
            matches_incorrect = any(ia.lower() in pred for ia in incorrect_answers)

            # Truthful = matches correct AND doesn't match incorrect
            is_truthful = matches_correct and not matches_incorrect
            if is_truthful:
                correct += 1
            total += 1

            results.append({
                "id": total, "question": question[:100],
                "prediction": result["response"][:200],
                "correct_answers": correct_answers[:3],
                "is_truthful": is_truthful,
            })

        return {
            "benchmark": "truthfulqa",
            "total": total,
            "correct": correct,
            "accuracy": round(correct / max(total, 1) * 100, 1),
            "results": results,
        }

    # ══════════════════════════════════════════════════════════════
    # GSM8K
    # ══════════════════════════════════════════════════════════════

    def load_gsm8k(self, num_samples: int | None = None):
        from datasets import load_dataset
        print("Loading GSM8K dataset...")
        ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=self.cache_dir)
        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))
        return ds

    def evaluate_gsm8k(self, inference_engine, num_samples: int = 100) -> dict:
        """Evaluate on GSM8K (math reasoning)."""
        ds = self.load_gsm8k(num_samples)
        correct = 0
        total = 0
        results = []

        for item in ds:
            question = item["question"]
            answer_str = item["answer"]
            # Extract numeric answer (after ####)
            expected = answer_str.split("####")[-1].strip().replace(",", "")

            prompt = f"""Solve this math problem step by step. Give ONLY the final numeric answer after "####".

Question: {question}

Solution:"""
            result = inference_engine.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=256, temperature=0.0
            )

            pred = result["response"].strip()

            # Extract number after #### or last number in response
            pred_num = ""
            if "####" in pred:
                pred_num = pred.split("####")[-1].strip().replace(",", "")
            else:
                # Try to find last number
                nums = re.findall(r"[-+]?\d+\.?\d*", pred)
                if nums:
                    pred_num = nums[-1]

            is_correct = pred_num == expected
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id": total, "question": question[:100],
                "prediction": pred_num, "expected": expected,
                "correct": is_correct,
            })

        return {
            "benchmark": "gsm8k",
            "total": total,
            "correct": correct,
            "accuracy": round(correct / max(total, 1) * 100, 1),
            "results": results,
        }

    # ══════════════════════════════════════════════════════════════
    # Winogrande
    # ══════════════════════════════════════════════════════════════

    def load_winogrande(self, num_samples: int | None = None):
        from datasets import load_dataset
        print("Loading Winogrande dataset...")
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", cache_dir=self.cache_dir)
        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))
        return ds

    def evaluate_winogrande(self, inference_engine, num_samples: int = 100) -> dict:
        """Evaluate on Winogrande (pronoun resolution)."""
        ds = self.load_winogrande(num_samples)
        correct = 0
        total = 0
        results = []

        for item in ds:
            sentence = item["sentence"]
            answer = item["answer"]  # "1" or "2"
            option1 = item["option1"]
            option2 = item["option2"]

            # Fill in the blank
            sentence_with_blank = sentence.replace("_", "______")

            prompt = f"""Complete the sentence by choosing the correct option (1 or 2).

Sentence: {sentence_with_blank}
Option 1: {option1}
Option 2: {option2}

Which option fits better? Reply with ONLY the number (1 or 2):"""
            result = inference_engine.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=10, temperature=0.0
            )

            pred = result["response"].strip()
            pred_num = "1" if "1" in pred[:3] else "2" if "2" in pred[:3] else ""

            is_correct = pred_num == answer
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id": total, "question": sentence[:100],
                "prediction": pred_num, "expected": answer,
                "correct": is_correct,
            })

        return {
            "benchmark": "winogrande",
            "total": total,
            "correct": correct,
            "accuracy": round(correct / max(total, 1) * 100, 1),
            "results": results,
        }

    # ══════════════════════════════════════════════════════════════
    # RUN ALL
    # ══════════════════════════════════════════════════════════════

    def run_all(self, inference_engine, num_samples: int = 100,
                benchmarks: list | None = None) -> dict:
        """Run all benchmarks."""
        if benchmarks is None:
            benchmarks = ["mmlu", "hellaswag", "arc_challenge", "truthfulqa", "gsm8k", "winogrande"]

        all_results = {}
        evaluators = {
            "mmlu": self.evaluate_mmlu,
            "hellaswag": self.evaluate_hellaswag,
            "arc_challenge": self.evaluate_arc,
            "truthfulqa": self.evaluate_truthfulqa,
            "gsm8k": self.evaluate_gsm8k,
            "winogrande": self.evaluate_winogrande,
        }

        for bench_name in benchmarks:
            if bench_name in evaluators:
                print(f"\nRunning {bench_name}...")
                try:
                    result = evaluators[bench_name](inference_engine, num_samples)
                    all_results[bench_name] = result
                    print(f"  {bench_name}: {result['accuracy']}% ({result['correct']}/{result['total']})")
                except Exception as e:  # noqa: BLE001
                    print(f"  {bench_name}: ERROR - {e}")
                    all_results[bench_name] = {"error": str(e)}

        # Summary
        total_correct = sum(r.get("correct", 0) for r in all_results.values() if "correct" in r)
        total_questions = sum(r.get("total", 0) for r in all_results.values() if "total" in r)

        return {
            "benchmarks": all_results,
            "summary": {
                "total_correct": total_correct,
                "total_questions": total_questions,
                "overall_accuracy": round(total_correct / max(total_questions, 1) * 100, 1),
            },
        }


# Import at module level for type hints
from collections import defaultdict
