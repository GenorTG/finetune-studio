"""Models whose repos carry stale code or a vision tower train, merge and open like any other.

Found by training one real run per model family (2026-11-05, RESULTS §37): Phi-4-mini's own modeling_phi3.py fails on
this transformers, and a merged Gemma 3 lacked the processor settings anything building its processor needs.
"""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.hf_env import trust_remote_code_for
from finetune_studio.training.merge_base import copy_processor_files


def _config(d: Path, **cfg: object) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def test_native_types_never_run_the_repos_code(tmp_path: Path) -> None:
    assert trust_remote_code_for(str(_config(tmp_path / "phi", model_type="phi3", auto_map={"AutoConfig": "x.Y"}))) is False


def test_unknown_types_still_can(tmp_path: Path) -> None:
    assert trust_remote_code_for(str(_config(tmp_path / "new", model_type="never-heard-of-it"))) is True
    assert trust_remote_code_for("org/not-a-folder") is True


def test_a_merge_keeping_its_vision_tower_gets_the_processor_files(tmp_path: Path) -> None:
    base = _config(tmp_path / "base", model_type="gemma3")
    (base / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (base / "processor_config.json").write_text("{}", encoding="utf-8")
    merged = _config(tmp_path / "merged", model_type="gemma3", vision_config={"x": 1})
    assert copy_processor_files(str(base), str(merged)) == ["preprocessor_config.json", "processor_config.json"]
    assert (merged / "processor_config.json").is_file()


def test_a_text_only_merge_stays_text_only(tmp_path: Path) -> None:
    base = _config(tmp_path / "base", model_type="qwen3_5", vision_config={"x": 1})
    (base / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    merged = _config(tmp_path / "merged", model_type="qwen3_5_text")
    assert copy_processor_files(str(base), str(merged)) == []
    assert not (merged / "preprocessor_config.json").exists()
