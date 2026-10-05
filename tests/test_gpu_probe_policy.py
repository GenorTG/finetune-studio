"""The nvidia-smi fallback sees every physical card; it must still respect the
GPU policy env (a CPU-forced host listed the excluded GTX 1070 in Host Resources)."""
from __future__ import annotations

import pytest

from finetune_studio.webui import gpu_probe

SMI = "0, NVIDIA GeForce RTX 3090, 900, 24576\n1, NVIDIA GeForce GTX 1070, 5800, 8192\n"


@pytest.fixture
def smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_probe, "_run", lambda *_a, **_k: SMI)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("FTS_GPU_EXCLUDE", raising=False)


def test_no_policy_lists_all(smi: None) -> None:
    assert [d["index"] for d in gpu_probe._nvidia_smi()] == [0, 1]


def test_cuda_visible_devices_limits_listing(smi: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert [d["index"] for d in gpu_probe._nvidia_smi()] == [0]


def test_gpu_exclude_by_name(smi: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FTS_GPU_EXCLUDE", "GTX 1070")
    assert [d["name"] for d in gpu_probe._nvidia_smi()] == ["NVIDIA GeForce RTX 3090"]
