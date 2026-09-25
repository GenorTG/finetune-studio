"""Guard the unsloth import-order precondition reported by the training engine.

Background (2026-09-25): unsloth patches trl/transformers/peft at import time
and only emits a UserWarning if they loaded first — so a run can silently lose
its optimizations. An earlier note claimed ``training/engine.py`` had this
order wrong (transformers at :578 before unsloth at :614). That was a false
positive: :578 is in ``_load_model_with_fallback``, called only from
``_train_standard``; ``_train_unsloth`` imports unsloth first.

What remains real is process history — ``training_engine`` is a module-level
singleton in ``webui/app.py``, so an earlier request that imported the trio
poisons every later unsloth run in that worker. ``_train_unsloth`` therefore
logs the precondition; these tests pin the helper that computes it.

Deliberately CPU-only: importing unsloth or torch here would be slow, and the
helper only reads ``sys.modules``.
"""

from __future__ import annotations

import sys

import pytest

from finetune_studio.training.engine import (
    _UNSLOTH_CRITICAL_MODULES,
    _unsloth_preimport_blockers,
)


def test_critical_modules_match_unsloth_itself() -> None:
    """The list must stay aligned with unsloth's own trigger condition.

    unsloth/_gpu_init.py uses ["trl", "transformers", "peft"] and notably does
    NOT include torch. If unsloth changes that set, a warning here is a signal
    to re-check rather than ignore.
    """
    assert set(_UNSLOTH_CRITICAL_MODULES) == {"trl", "transformers", "peft"}
    assert "torch" not in _UNSLOTH_CRITICAL_MODULES, (
        "torch-first is harmless — unsloth does not warn on it. Including it "
        "would make every run look degraded."
    )


def test_clean_process_reports_no_blockers() -> None:
    """A fresh process must report clean, so a first unsloth run is not flagged."""
    saved = {m: sys.modules.pop(m, None) for m in _UNSLOTH_CRITICAL_MODULES}
    try:
        assert _unsloth_preimport_blockers() == []
    finally:
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod


@pytest.mark.parametrize("poisoned", list(_UNSLOTH_CRITICAL_MODULES))
def test_poisoned_process_is_detected(poisoned: str) -> None:
    """Each critical module, once loaded, must be reported as a blocker."""
    saved = {m: sys.modules.pop(m, None) for m in _UNSLOTH_CRITICAL_MODULES}
    sentinel = type(sys)(poisoned)
    sys.modules[poisoned] = sentinel
    try:
        assert _unsloth_preimport_blockers() == [poisoned]
    finally:
        sys.modules.pop(poisoned, None)
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod


def test_order_follows_the_declared_sequence() -> None:
    """With all three loaded, the report is stable and in declaration order."""
    saved = {m: sys.modules.pop(m, None) for m in _UNSLOTH_CRITICAL_MODULES}
    for name in _UNSLOTH_CRITICAL_MODULES:
        sys.modules[name] = type(sys)(name)
    try:
        assert _unsloth_preimport_blockers() == list(_UNSLOTH_CRITICAL_MODULES)
        assert _unsloth_preimport_blockers() == _unsloth_preimport_blockers()
    finally:
        for name in _UNSLOTH_CRITICAL_MODULES:
            sys.modules.pop(name, None)
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod


def test_train_unsloth_reads_precondition_before_importing_unsloth() -> None:
    """The check must come BEFORE ``from unsloth import ...`` in the source.

    Once unsloth is in sys.modules the check is meaningless, so an in-source
    ordering guard is the only thing that catches a later refactor moving the
    log line down. Static by design: importing the engine's real dependency
    chain here would be far too slow for a unit test.
    """
    import inspect

    from finetune_studio.training.engine import TrainingEngine

    src = inspect.getsource(TrainingEngine._train_unsloth)
    check_at = src.find("_unsloth_preimport_blockers()")
    unsloth_at = src.find("from unsloth import")
    assert check_at != -1, "precondition check missing from _train_unsloth"
    assert unsloth_at != -1, "unsloth import missing from _train_unsloth"
    assert check_at < unsloth_at, (
        "the precondition check must run before `from unsloth import` — after "
        "that import the check is meaningless"
    )
