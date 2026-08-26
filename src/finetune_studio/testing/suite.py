"""Test suite — predefined test cases.

WHAT THIS FILE DOES
==================
Defines test cases that can be run against any model:
  - TestCase: a single test (prompt + expected keywords + forbidden words)
  - TestSuite: a collection of TestCases
  - TestRunner: executes a suite against a model and reports results

KEY CONCEPTS
============
- Keyword matching: the response must contain all expected keywords.
- Forbidden words: the response must NOT contain any forbidden words.
- Pass/fail tracking: each test is marked pass or fail with details.
- Aggregate metrics: overall pass rate, average time per test.
"""

import json
import time
from dataclasses import dataclass, field


@dataclass
class TestCase:
    name: str
    messages: list
    expected_keywords: list = field(default_factory=list)
    forbidden_keywords: list = field(default_factory=list)
    category: str = "general"

@dataclass
class TestResult:
    test_name: str
    response: str
    passed: bool
    keyword_hits: list = field(default_factory=list)
    keyword_misses: list = field(default_factory=list)
    forbidden_hits: list = field(default_factory=list)
    time_ms: float = 0
    error: str = ""

def load_test_suite(path):
    with open(path) as f:
        data = json.load(f)
    return [TestCase(
        name=item["name"], messages=item["messages"],
        expected_keywords=item.get("expected_keywords", []),
        forbidden_keywords=item.get("forbidden_keywords", []),
        category=item.get("category", "general"),
    ) for item in data]

def run_suite(inference_engine, cases, max_tokens=512):
    results = []
    for case in cases:
        start = time.time()
        try:
            response = inference_engine.generate(case.messages, max_tokens=max_tokens)
            elapsed_ms = (time.time() - start) * 1000
            resp_lower = response.lower()
            hits = [k for k in case.expected_keywords if k.lower() in resp_lower]
            misses = [k for k in case.expected_keywords if k.lower() not in resp_lower]
            forbidden = [k for k in case.forbidden_keywords if k.lower() in resp_lower]
            results.append(TestResult(
                test_name=case.name, response=response, passed=len(misses) == 0 and len(forbidden) == 0,
                keyword_hits=hits, keyword_misses=misses, forbidden_hits=forbidden,
                time_ms=round(elapsed_ms, 1),
            ))
        except Exception as e:  # noqa: BLE001
            results.append(TestResult(
                test_name=case.name, response="", passed=False, error=str(e),
                time_ms=round((time.time() - start) * 1000, 1),
            ))
    return results

def score_results(results):
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_time = sum(r.time_ms for r in results) / max(total, 1)
    return {
        "total": total, "passed": passed, "failed": total - passed,
        "pass_rate": round(passed / max(total, 1) * 100, 1),
        "avg_time_ms": round(avg_time, 1),
    }
