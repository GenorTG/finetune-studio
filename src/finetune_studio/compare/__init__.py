"""Compare subpackage — retired.

This package used to hold a parallel model-comparison implementation
(``engine.py`` ``ComparisonEngine``, ``scorer.py`` ``Scorer``,
``reporter.py``) that duplicated ``finetune_studio.benchmarks.comparison``
(``ModelComparator`` / the ``comparator`` singleton) and had zero callers
anywhere in the app or tests. It was removed; use
``finetune_studio.benchmarks.comparison`` instead — that is what
``fts compare`` and the WebUI ``/compare/*`` routes actually use.
"""
