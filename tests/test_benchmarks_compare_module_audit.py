"""Regression tests for the E2 benchmarks/compare dead-code audit.

Pins two removals:
  - ``finetune_studio.benchmarks`` package __init__ no longer exposes the
    legacy ``BenchmarkSuite``/``BaseBenchmark``/``MMLUSample``-style class
    hierarchy — it had zero production or test callers anywhere in the
    repo and was a parallel, unused implementation of what
    ``real_benchmarks.py`` / ``offline_suites.py`` / ``suite_defs.py``
    actually do for the live WebUI benchmark routes.
  - ``finetune_studio.compare.engine`` / ``.reporter`` / ``.scorer`` no
    longer exist — they were an entirely unused duplicate of the live
    ``finetune_studio.benchmarks.comparison`` (``ModelComparator`` /
    ``comparator`` singleton), itself since replaced by the run-then-judge
    Compare tab (``finetune_studio.compare`` + ``webui/compare_jobs.py``).
  - ``finetune_studio.benchmarks.samplers`` / ``.tool_calling`` no longer
    exist — both were fully orphaned (zero callers anywhere, including
    tests).

Also pins that the surviving live code paths still work.
"""

from __future__ import annotations

import importlib

import pytest


def test_benchmarks_init_has_no_dead_class_hierarchy() -> None:
    import finetune_studio.benchmarks as benchmarks_pkg

    dead_names = [
        "BenchmarkSuite",
        "BaseBenchmark",
        "MMLUSample",
        "HellaSwagSample",
        "ARCChallengeSample",
        "TriviaQASample",
        "WinoGrandeSample",
        "IFEvalSample",
        "ToolBenchSample",
        "GSM8KSample",
        "HumanEvalSample",
        "TruthfulQASample",
        "PersonaTest",
        "BenchmarkResult",
    ]
    for name in dead_names:
        assert not hasattr(benchmarks_pkg, name), f"{name} should have been removed"


@pytest.mark.parametrize(
    "module_name",
    [
        "finetune_studio.benchmarks.samplers",
        "finetune_studio.benchmarks.tool_calling",
        "finetune_studio.compare.engine",
        "finetune_studio.compare.reporter",
        "finetune_studio.compare.scorer",
        "finetune_studio.benchmarks.comparison",
    ],
)
def test_dead_modules_no_longer_importable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_scoring_module_has_no_dead_benchmark_result_dataclass() -> None:
    from finetune_studio.benchmarks import scoring

    assert not hasattr(scoring, "BenchmarkResult")
    # The exact-match scorer used by the public benchmarks still works.
    assert scoring.scorer.score_mcq("B) Paris", "B")["correct"] is True


def test_compare_package_still_imports_cleanly() -> None:
    # The __init__.py docstring was updated to document the retirement;
    # the package itself must still import without error.
    import finetune_studio.compare  # noqa: F401
