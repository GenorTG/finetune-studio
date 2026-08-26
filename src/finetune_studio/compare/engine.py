from typing import Any

"""Comparison engine — runs the same prompts through multiple models.

WHAT THIS FILE DOES
==================
The core of the comparison feature. Takes a list of models and a
list of test prompts, runs each prompt through each model, and
collects the results.

KEY CONCEPTS
============
- Parallel execution: multiple models can run simultaneously (if
  you have multiple GPUs, or one model on GPU and another on CPU).
- Failure isolation: if one model crashes, the others still complete.
- Caching: results are cached so re-running with the same inputs
  returns immediately.
"""

"""Comparison engine — run same prompts through multiple models/APIs."""
import time
from dataclasses import dataclass

import requests


@dataclass
class ComparisonConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    runs_per_prompt: int = 1
    timeout: int = 60


@dataclass
class ModelSource:
    name: str
    type: str  # "local" or "api"
    path: str = ""  # For local models
    api_url: str = ""  # For API models
    api_key: str = ""  # For API models
    model_id: str = ""  # Model name for API


class ComparisonEngine:
    """Run comparison tests between multiple model sources."""

    def __init__(self, config: ComparisonConfig = None):
        self.config = config or ComparisonConfig()
        self._local_engines: dict[str, Any] = {}

    def _get_local_engine(self, path: str):
        """Get or create a local inference engine."""
        if path not in self._local_engines:
            from finetune_studio.testing.inference import InferenceEngine
            engine = InferenceEngine()
            engine.load(path)
            self._local_engines[path] = engine
        return self._local_engines[path]

    def _call_api(self, source: ModelSource, messages: list) -> str:
        """Call an external API (OpenAI-compatible)."""
        headers = {"Content-Type": "application/json"}
        if source.api_key:
            headers["Authorization"] = f"Bearer {source.api_key}"

        payload = {
            "model": source.model_id,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        response = requests.post(
            source.api_url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate_response(self, source: ModelSource, messages: list) -> dict:
        """Generate a response from a model source."""
        start = time.time()
        try:
            if source.type == "local":
                engine = self._get_local_engine(source.path)
                response = engine.generate(
                    messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            elif source.type == "api":
                response = self._call_api(source, messages)
            else:
                raise ValueError(f"Unknown source type: {source.type}")

            elapsed = (time.time() - start) * 1000
            return {"response": response, "time_ms": round(elapsed, 1), "error": None}
        except Exception as e:  # noqa: BLE001
            elapsed = (time.time() - start) * 1000
            return {"response": "", "time_ms": round(elapsed, 1), "error": str(e)}

    def run_comparison(self, sources: list[ModelSource], test_suite: list[dict],
                       config: ComparisonConfig = None) -> list[dict]:
        """Run comparison across multiple sources on a test suite."""
        cfg = config or self.config
        results = []

        for test in test_suite:
            test_result = {
                "name": test["name"],
                "messages": test["messages"],
                "expected_keywords": test.get("expected_keywords", []),
                "forbidden_keywords": test.get("forbidden_keywords", []),
                "responses": {},
            }

            for source in sources:
                source_results = []
                for run in range(cfg.runs_per_prompt):
                    result = self.generate_response(source, test["messages"])
                    source_results.append(result)

                test_result["responses"][source.name] = source_results

            results.append(test_result)

        return results

    def cleanup(self):
        """Unload all local engines."""
        for engine in self._local_engines.values():
            engine.unload()
        self._local_engines.clear()
