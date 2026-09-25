"""Import health guard for the training stack.

On 2026-09-25 the venv carried ``torchao 0.18`` against a CUDA-pinned
``torch 2.6.0+cu124``. torchao >=0.17 calls
``torch.utils._pytree.register_constant``, which only exists in torch 2.7+,
so **every** transformers import raised ``AttributeError`` at module scope:
``TrainingArguments``, ``peft``, and the model classes all died with the
misleading "Are this object's requirements defined correctly?" hint.

The venv "looked fine" because pip had installed cleanly, and the test suite
never imported the training stack at import time — so the breakage was
invisible until something actually tried to fine-tune.

This test imports the critical stack and fails with an actionable message, so
a dependency-resolution regression is caught by ``make test`` rather than by a
user clicking Train. Optional accelerators are skipped when genuinely absent
(air-gapped / CPU-only installs); only the core must import.
"""

from __future__ import annotations

import importlib

import pytest

# (module, attribute) pairs that must import on any working install. The
# attribute check matters: several of these packages import their top-level
# module fine and only fail on the specific symbol.
CORE: tuple[tuple[str, str], ...] = (
    ("torch", None),
    ("transformers", "TrainingArguments"),
    ("transformers", "AutoModelForCausalLM"),
    ("peft", "LoraConfig"),
    ("peft", "get_peft_model"),
    ("datasets", "load_dataset"),
    ("accelerate", "Accelerator"),
)

# Absent on a minimal/CPU install — reported, never failed.
OPTIONAL: tuple[tuple[str, str | None], ...] = (
    ("trl", "SFTTrainer"),
    ("bitsandbytes", None),
    ("unsloth", "FastLanguageModel"),
    ("torchao", None),
)


def _load(module: str, attribute: str | None) -> None:
    mod = importlib.import_module(module)
    if attribute is not None:
        getattr(mod, attribute)


def test_core_training_stack_imports() -> None:
    """The core stack must import. This is the torchao-class regression guard."""
    failures: list[str] = []
    for module, attribute in CORE:
        try:
            _load(module, attribute)
        except Exception as exc:  # noqa: BLE001 - report any import-time failure
            target = f"{module}.{attribute}" if attribute else module
            failures.append(f"  {target}: {type(exc).__name__}: {exc}")

    assert not failures, (
        "the training stack does not import — nothing can fine-tune until this "
        "is fixed. Dependency-resolution regressions usually mean a package "
        "targets a newer torch than the one pinned for this driver. Re-sync "
        "with `bash update.sh` so the torch constraints (including the "
        "torchao<0.17 cap on torch<2.7) are applied.\n"
        + "\n".join(failures)
    )


def test_optional_accelerator_stack_is_reported() -> None:
    """Optional packages are informational — a missing one must not fail CI."""
    missing: list[str] = []
    for module, attribute in OPTIONAL:
        try:
            _load(module, attribute)
        except Exception:  # noqa: BLE001 - absence is acceptable here
            missing.append(module)

    # Never fails; recorded so a broken-optional regression is visible in -v.
    if missing:
        pytest.skip(f"optional stack not installed on this host: {', '.join(missing)}")
