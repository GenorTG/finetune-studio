"""Model comparison infrastructure.

WHAT THIS FILE DOES
==================
Defines the data structures for comparing models side-by-side:
  - ComparisonResult: one model's response + score for a single test
  - ComparisonEngine: runs all models on the same test suite
  - Scoring: keywords, length, time, forbidden words

KEY CONCEPTS
============
- Side-by-side comparison: run the same prompt through multiple models
  and compare their outputs.
- Pairwise scoring: not just "is this good?" but "is this better than
  the other model?"
- Statistical significance: with 20+ samples, we can detect small
  differences. With 5 samples, results are noisy.
"""

"""Comparison module for Finetune Studio — compare models side-by-side."""
import time
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    source_name: str
    question: str
    response: str
    score: dict
    time_ms: float = 0.0
    error: str = ""


class ModelComparator:
    """Compare models on the same test suite."""

    def __init__(self):
        self.engines = {}

    def load_model(self, name: str, path: str):
        from finetune_studio.testing.inference import InferenceEngine
        engine = InferenceEngine()
        engine.load(path)
        self.engines[name] = engine

    def unload_all(self):
        for engine in self.engines.values():
            engine.unload()
        self.engines.clear()

    def cleanup(self):
        for engine in self.engines.values():
            try:
                engine.unload()
            except Exception:  # noqa: BLE001, S110
                pass
        self.engines.clear()

    def run_comparison(self, test_suite, config: dict | None = None) -> dict:
        config = config or {"max_tokens": 512, "temperature": 0.7}
        results = []

        for test in test_suite:
            # Handle both TestCase objects and dicts
            if hasattr(test, 'name'):
                test_name = test.name
                test_messages = test.messages
                test_expected = {
                    "keywords": getattr(test, 'expected_keywords', []),
                    "forbidden": getattr(test, 'forbidden_keywords', []),
                }
            else:
                test_name = test["name"]
                test_messages = test["messages"]
                test_expected = test.get("expected", {})

            test_result = {
                "name": test_name,
                "messages": test_messages,
                "expected": test_expected,
                "responses": {},
            }

            for model_name, engine in self.engines.items():
                start = time.time()
                try:
                    response = engine.generate(
                        test_messages,
                        max_tokens=config["max_tokens"],
                        temperature=config["temperature"],
                    )
                    elapsed = (time.time() - start) * 1000

                    score = self._score_response(response, test_expected)

                    test_result["responses"][model_name] = {
                        "response": response if isinstance(response, str) else str(response),
                        "score": score,
                        "time_ms": round(elapsed, 1),
                    }
                except Exception as e:  # noqa: BLE001
                    test_result["responses"][model_name] = {
                        "response": "",
                        "score": {"correct": False, "error": str(e)},
                        "time_ms": 0,
                        "error": str(e),
                    }

            results.append(test_result)

        # Aggregate scores
        summary = {}
        for model_name in self.engines:
            scores = []
            times = []
            for r in results:
                resp = r["responses"].get(model_name, {})
                if "score" in resp:
                    scores.append(1 if resp["score"].get("correct", False) else 0)
                if "time_ms" in resp:
                    times.append(resp["time_ms"])

            summary[model_name] = {
                "accuracy": round(sum(scores) / max(len(scores), 1) * 100, 1),
                "avg_time_ms": round(sum(times) / max(len(times), 1), 1),
                "total": len(results),
                "passed": sum(scores),
            }

        return {"results": results, "summary": summary}

    def _score_response(self, response, expected: dict) -> dict:
        if isinstance(response, dict):
            response = response.get("response", str(response))

        from finetune_studio.benchmarks.scoring import scorer

        keywords = expected.get("keywords", [])
        forbidden = expected.get("forbidden", [])

        if keywords or forbidden:
            return scorer.score_open_ended(response, "", keywords, forbidden)

        return {"correct": True, "score": 1.0, "method": "no_check"}


# Module-level singleton instance for easy import
# This matches the pattern used by callers: `from finetune_studio.benchmarks.comparison import comparator`
comparator = ModelComparator()
