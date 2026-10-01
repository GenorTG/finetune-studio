"""Regressions from the training/ module documentation+correctness audit.

Covers:
- TrainingEngine._persist_run_output must not let a secondary NameError
  mask the original failure when the db.runs import itself fails.
- convert_merged_to_gguf's success path must return `files` and `quants`
  as parallel same-order, same-length lists (load-bearing for the
  one-DB-row-per-quant export fix in webui/routes/exports.py).
- The dead `training/unsloth_engine.py` duplicate of
  TrainingEngine._train_unsloth must stay removed.
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import patch

from finetune_studio.training.engine import TrainingConfig, TrainingEngine, TrainingState
from finetune_studio.training.gguf_convert import convert_merged_to_gguf


def test_persist_run_output_survives_db_import_failure() -> None:
    """run_id must be bound before the try so the except branch can log it.

    Previously `run_id = ...` was the first statement inside the `try:`
    block. If `from finetune_studio.db.runs import update_run` itself
    raised (e.g. ImportError), `run_id` was never assigned, and the
    `except Exception: log.exception("...", run_id)` handler raised an
    unrelated NameError that propagated out of `_persist_run_output`
    uncaught — silently turning a successful training run into a
    reported failure in `_train()`'s outer except block.
    """
    engine = TrainingEngine()
    engine.current_run_id = "proj123-abc456"
    engine.config = TrainingConfig(output_dir="/tmp/fts-audit-does-not-exist")
    engine.state = TrainingState(status="done")

    with patch.dict(sys.modules, {"finetune_studio.db.runs": None}):
        engine._persist_run_output()  # must not raise


def test_convert_merged_to_gguf_files_quants_parallel(tmp_path) -> None:
    """On success, `files[i]` must correspond to `quants[i]` in order.

    webui/routes/exports.py's export_run zips quants with files 1:1 to
    register one DB row per quant. If the two lists ever drifted in
    order or length, that fix would silently mismatch quant labels to
    the wrong file path.
    """
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "model.safetensors").write_bytes(b"\x00")
    gguf_dir = tmp_path / "gguf"

    requested = ["q4_k_m", "f16", "q5_k_m"]
    old_env = os.environ.get("FTS_SKIP_EXPORT")
    os.environ["FTS_SKIP_EXPORT"] = "1"
    try:
        result = convert_merged_to_gguf(str(merged_dir), str(gguf_dir), requested)
    finally:
        if old_env is None:
            os.environ.pop("FTS_SKIP_EXPORT", None)
        else:
            os.environ["FTS_SKIP_EXPORT"] = old_env

    assert result["ok"] is True
    assert result["quants"] == requested
    assert len(result["files"]) == len(requested)
    for quant, file_path in zip(result["quants"], result["files"]):
        assert os.path.basename(file_path) == f"model-{quant}.gguf"


def test_unsloth_engine_duplicate_module_removed() -> None:
    """training/unsloth_engine.py duplicated TrainingEngine._train_unsloth
    with zero real callers anywhere in the repo (confirmed via full-repo
    grep) — a dead 'one true way' violation. It was deleted; this pins
    that it does not silently reappear."""
    import importlib.util

    spec = importlib.util.find_spec("finetune_studio.training.unsloth_engine")
    assert spec is None
