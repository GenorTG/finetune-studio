"""Regression E2E-27: in-server training must not fork dataset worker pools.

Unsloth's patched SFTTrainer auto-sized dataset_num_proc to 8; datasets forked
a pool inside the threaded uvicorn process with CUDA initialised and every
worker deadlocked, freezing the run at "Loading model…" with no error.
"""
from __future__ import annotations

import importlib
import os

import pytest


def test_engine_import_pins_serial_tokenization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNSLOTH_DATASET_NUM_PROC", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    from finetune_studio.training import engine

    importlib.reload(engine)
    assert os.environ["UNSLOTH_DATASET_NUM_PROC"] == "0"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_explicit_user_override_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNSLOTH_DATASET_NUM_PROC", "4")
    from finetune_studio.training import engine

    importlib.reload(engine)
    assert os.environ["UNSLOTH_DATASET_NUM_PROC"] == "4"
