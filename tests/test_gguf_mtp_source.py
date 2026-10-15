"""GGUF conversion: a Qwen3.5 merged model without mtp weights must be converted with the converter's --no-mtp."""
from __future__ import annotations

import json
import struct
from pathlib import Path

from finetune_studio.training.gguf_convert import converter_extra_args


def _shard(path: Path, names: list[str]) -> None:
    header = json.dumps({n: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for n in names}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")


def _model(d: Path, names: list[str], mtp: int) -> Path:
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": mtp}}))
    _shard(d / "model.safetensors", names)
    return d


def _script(tmp: Path, flag: bool) -> str:
    p = tmp / "convert_hf_to_gguf.py"
    p.write_text('parser.add_argument("--no-mtp")\n' if flag else "# old converter\n")
    return str(p)


def test_missing_mtp_weights_get_no_mtp(tmp_path: Path) -> None:
    d = _model(tmp_path / "m", ["model.layers.0.w", "lm_head.weight"], mtp=1)
    assert converter_extra_args(str(d), _script(tmp_path, True)) == ["--no-mtp"]


def test_a_model_with_mtp_weights_or_declaring_none_converts_as_before(tmp_path: Path) -> None:
    script = _script(tmp_path, True)
    assert converter_extra_args(str(_model(tmp_path / "a", ["mtp.layers.0.w", "model.layers.0.w"], mtp=1)), script) == []
    assert converter_extra_args(str(_model(tmp_path / "b", ["model.layers.0.w"], mtp=0)), script) == []


def test_an_old_converter_without_the_flag_and_an_unreadable_dir_add_nothing(tmp_path: Path) -> None:
    d = _model(tmp_path / "m", ["model.layers.0.w"], mtp=1)
    assert converter_extra_args(str(d), _script(tmp_path, False)) == []
    assert converter_extra_args(str(tmp_path / "missing"), _script(tmp_path, True)) == []
