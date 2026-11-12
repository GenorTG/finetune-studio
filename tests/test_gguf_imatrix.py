"""Importance-matrix GGUF export: calibration text, which quants it applies to, missing-tool refusals."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio.training import gguf_convert
from finetune_studio.training.run_export import imatrix_calibration_text


def _jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n")
    return str(path)


def test_calibration_text_reads_every_training_shape(tmp_path: Path) -> None:
    data = _jsonl(tmp_path / "d.jsonl", [
        {"messages": [{"role": "user", "content": "Q one?"}, {"role": "assistant", "content": "A one."}]},
        {"conversations": [{"from": "human", "value": "Q two?"}, {"from": "gpt", "value": "A two."}]},
        {"text": "Raw domain text."},
        {"prompt": "P", "completion": "C"},
        {"messages": []},
    ])
    text = imatrix_calibration_text(data)
    assert text == "Q one?\nA one.\n\nQ two?\nA two.\n\nRaw domain text.\n\nP\nC"


def test_calibration_text_is_capped_and_empty_for_a_missing_file(tmp_path: Path) -> None:
    data = _jsonl(tmp_path / "d.jsonl", [{"text": "x" * 100}] * 50)
    assert len(imatrix_calibration_text(data, max_chars=250)) == 250
    assert imatrix_calibration_text(str(tmp_path / "gone.jsonl")) == ""
    assert imatrix_calibration_text("") == ""


def test_imatrix_applies_only_to_llama_quantize_types() -> None:
    assert gguf_convert.imatrix_applies(["q4_k_m"])
    assert gguf_convert.imatrix_applies(["f16", "q8_0", "q3_k_m"])
    assert not gguf_convert.imatrix_applies(["f16", "q8_0", "bf16"])
    assert not gguf_convert.imatrix_applies([])


def test_compute_imatrix_without_the_tool_says_how_to_get_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gguf_convert, "find_llama_imatrix", lambda: None)
    with pytest.raises(RuntimeError, match="llama-imatrix binary not found"):
        gguf_convert.compute_imatrix(str(tmp_path / "m.gguf"), "text", str(tmp_path / "imatrix.dat"))


def test_imatrix_file_is_never_listed_as_a_model() -> None:
    assert not gguf_convert.IMATRIX_FILENAME.endswith(".gguf")
