"""Tests: real GGUF topology resolution + n_gpu_layers semantics.

Genor's rule: no magic 99 = "all layers". The loader must read the model's
real block count from the GGUF header, translate legacy 99 → -1 (llama.cpp
native "all"), and expose the true count (block_count / gpu_layers_total)
instead of a fake number.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from finetune_studio.models.gguf_layers import (
    gguf_header_values,
    is_gguf_path,
    resolve_block_count,
)


def _write_minimal_gguf(path: Path, arch: str = "qwen3",
                        block_count: int = 36, ctx: int = 40960) -> None:
    """Hand-rolled minimal GGUF v3 header with two uint32 scalar fields."""
    buf = b"GGUF" + struct.pack("<I", 3)          # magic + version
    buf += struct.pack("<Q", 0)                    # tensor count (header only)
    buf += struct.pack("<Q", 2)                    # kv pair count
    for name, val in ((f"{arch}.block_count", block_count),
                      (f"{arch}.context_length", ctx)):
        nk = name.encode()
        buf += struct.pack("<Q", len(nk)) + nk     # key
        buf += struct.pack("<I", 4)                # value type: UINT32
        buf += struct.pack("<I", val)              # scalar value
    path.write_bytes(buf)


def test_is_gguf_path(tmp_path: Path) -> None:
    assert is_gguf_path("/x/model.gguf")
    assert is_gguf_path("/x/MODEL.GGUF")
    assert not is_gguf_path("/x/model.safetensors")


def test_resolve_block_count_reads_real_value(tmp_path: Path) -> None:
    p = tmp_path / "m.gguf"
    _write_minimal_gguf(p, block_count=36, ctx=40960)
    got = resolve_block_count(str(p))
    assert got["block_count"] == 36
    assert got["context_length"] == 40960


def test_resolve_llama_arch_prefix(tmp_path: Path) -> None:
    p = tmp_path / "l.gguf"
    _write_minimal_gguf(p, arch="llama", block_count=32)
    assert resolve_block_count(str(p))["block_count"] == 32


def test_resolve_missing_file_is_empty(tmp_path: Path) -> None:
    got = resolve_block_count(str(tmp_path / "nope.gguf"))
    assert got == {"block_count": None, "context_length": None}


def test_non_gguf_paths_return_none() -> None:
    assert resolve_block_count("/x/model.safetensors") == {
        "block_count": None, "context_length": None,
    }


def test_gguf_header_values_scalar_only(tmp_path: Path) -> None:
    p = tmp_path / "m.gguf"
    _write_minimal_gguf(p, block_count=7)
    vals = gguf_header_values(str(p))
    assert vals["qwen3.block_count"] == 7

def test_manager_translates_legacy_99(monkeypatch: pytest.MonkeyPatch) -> None:
    """load() must rewrite legacy n_gpu_layers=99 to -1, never pass 99 on."""
    from finetune_studio.models import manager as mgr_mod

    class FakeProv:
        def __init__(self, cfg):
            self.config = cfg
            self._used_extra = dict(cfg.extra)
            self._loaded_at = 0.0

        def load(self) -> None:
            FakeProv.last_extra = dict(self.config.extra)

        def is_loaded(self) -> bool:
            return True

        def describe(self) -> dict:
            return {"id": self.config.id, "loaded": True,
                    "n_gpu_layers": self.config.extra["n_gpu_layers"]}

    class FakeMgr2(mgr_mod.ModelManager):
        def __init__(self):  # skip DB init
            import threading
            self._lock = threading.RLock()
            self._provider = None
            self._active_id = ""

    m = FakeMgr2()
    monkeypatch.setattr(mgr_mod, "build_provider", lambda cfg: FakeProv(cfg))
    monkeypatch.setattr(
        m, "get_provider",
        lambda pid: {"id": pid, "name": "t", "kind": "local_gguf",
                     "model_id": "/x/model.safetensors", "base_url": "",
                     "api_key": "", "extra": {"n_gpu_layers": 99}},
    )
    m.load("p1")
    extra = FakeProv.last_extra
    assert extra["n_gpu_layers"] == -1, extra
