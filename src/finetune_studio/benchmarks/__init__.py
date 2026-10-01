"""Benchmarks subpackage — public benchmark suites and scoring.

The actual benchmark execution paths used by the live app are:
  - ``finetune_studio.benchmarks.real_benchmarks`` — official HuggingFace
    MMLU / GSM8K / HellaSwag suites (``RealBenchmarkSuite``).
  - ``finetune_studio.benchmarks.offline_suites`` /
    ``offline_fixture_data`` — synthetic offline fixtures.
  - ``finetune_studio.benchmarks.suite_defs`` — suite discovery
    (``discover_suites``) used by the WebUI benchmark routes.
  - ``finetune_studio.benchmarks.scoring`` — response scoring helpers.
  - ``finetune_studio.benchmarks.comparison`` — side-by-side model
    comparison (``comparator`` singleton), used by ``fts compare`` and
    the WebUI ``/compare`` routes.

This file intentionally re-exports nothing: callers import from the
specific submodule above.
"""
