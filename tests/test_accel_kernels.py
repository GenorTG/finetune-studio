"""Linear-attention kernels are hidden from transformers on devices that cannot run them (accel/kernels.py)."""
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from finetune_studio.accel import kernels


def _acc(kind: str, cc: tuple[int, int] = (0, 0)) -> SimpleNamespace:
    return SimpleNamespace(kind=kind, compute_capability=cc, name=f"{kind} {cc}")


@pytest.mark.parametrize(("kind", "cc", "ok"), [
    ("cuda", (8, 6), True), ("cuda", (7, 5), True), ("cuda", (6, 1), False),   # GTX 1070: Triton/causal-conv1d SMs start at 7.5
    ("cpu", (0, 0), False), ("mps", (0, 0), False), ("xpu", (0, 0), False), ("rocm", (0, 0), True),
])
def test_supported_devices(kind: str, cc: tuple[int, int], ok: bool) -> None:
    assert kernels.linear_attention_supported(_acc(kind, cc)) is ok


def test_unsupported_device_blocks_the_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in kernels.LINEAR_ATTN_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert kernels.guard_linear_attention(_acc("cuda", (6, 1))) == ["fla", "causal_conv1d"]
    with pytest.raises(ImportError):
        importlib.import_module("fla")              # transformers catches this and keeps its PyTorch function
    for name in kernels.LINEAR_ATTN_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_supported_device_blocks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in kernels.LINEAR_ATTN_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert kernels.guard_linear_attention(_acc("cuda", (8, 6))) == []
    assert all(name not in sys.modules or sys.modules[name] is not None for name in kernels.LINEAR_ATTN_MODULES)


def test_status_reports_installed_kernels_and_whether_this_device_uses_them(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import metadata

    versions = {"flash-linear-attention": "0.5.2"}

    def fake_version(dist: str) -> str:
        if dist not in versions:
            raise metadata.PackageNotFoundError(dist)
        return versions[dist]

    monkeypatch.setattr(metadata, "version", fake_version)
    assert kernels.linear_attention_status(_acc("cuda", (8, 6))) == {"installed": versions, "active": True}
    assert kernels.linear_attention_status(_acc("cuda", (6, 1)))["active"] is False
    versions.clear()
    assert kernels.linear_attention_status(_acc("cuda", (8, 6))) == {"installed": {}, "active": False}


def test_fts_accel_report_line() -> None:
    from finetune_studio.cli.commands.accel import format_report

    base = {"accelerator": {"is_gpu": True, "kind": "cuda", "name": "RTX 3090", "torch_device": "cuda:0", "total_gb": 24,
                            "free_gb": 20, "runtime": "CUDA 13.0", "degraded_reason": "", "compute_capability": [8, 6],
                            "supports_bf16": True, "device_count": 1},
            "torch": {"version": "2.14.1", "cuda": "13.0"}, "llama_cpp": {"installed": False}, "hardware": ["nvidia"],
            "policy": {}}
    on = format_report({**base, "linear_attention": {"installed": {"flash-linear-attention": "0.5.2"}, "active": True}})
    assert any(line.startswith("Qwen3.5 kern: flash-linear-attention 0.5.2 · active") for line in on)
    off = format_report({**base, "linear_attention": {"installed": {}, "active": False}})
    assert any("not installed" in line and "install.sh --repair" in line for line in off)
