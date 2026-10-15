"""Training loads a base in 4-bit up front when bf16 weights + a training step cannot fit in free VRAM.

Live finding (Qwen3.5-9B DPO on a 24 GB card): the bf16 load succeeds (18 GiB), so the load-time OOM
ladder never fires, and the first forward pass dies with CUDA OOM.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio.accel.device import Accelerator
from finetune_studio.models import hf_loader
from finetune_studio.training import accel_plan


def _acc(free_gb: float, *, four_bit: bool = True) -> Accelerator:
    return Accelerator(
        kind="cuda", index=0, name="fake", total_gb=24.0, free_gb=free_gb, compute_capability=(8, 6),
        supports_bf16=True, supports_flash_attention=True, supports_4bit=four_bit, runtime="CUDA 13.0",
    )


def _model_dir(tmp_path: Path, gib: float) -> str:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    with shard.open("wb") as fh:
        fh.truncate(int(gib * 1024**3))  # sparse file: size without disk use
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _bnb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel_plan, "_has_module", lambda name: name == "bitsandbytes")


def test_nine_billion_base_on_24gb_card_goes_4bit(tmp_path: Path) -> None:
    needs, why = hf_loader.training_needs_4bit(_model_dir(tmp_path, 18.0), _acc(22.4))
    assert needs
    assert "18.0 GiB" in why and "4-bit" in why


def test_small_base_stays_bf16(tmp_path: Path) -> None:
    assert hf_loader.training_needs_4bit(_model_dir(tmp_path, 1.2), _acc(22.4)) == (False, "")


def test_boundary_uses_headroom(tmp_path: Path) -> None:
    path = _model_dir(tmp_path, 14.5)
    assert hf_loader.training_needs_4bit(path, _acc(22.4))[0] is False  # 14.5 + 6 <= 22.4
    assert hf_loader.training_needs_4bit(path, _acc(20.0))[0] is True


def test_no_4bit_support_never_forces_it(tmp_path: Path) -> None:
    assert hf_loader.training_needs_4bit(_model_dir(tmp_path, 18.0), _acc(22.4, four_bit=False)) == (False, "")


def test_hub_id_without_local_weights_is_left_alone() -> None:
    assert hf_loader.training_needs_4bit("Qwen/Qwen3-0.6B", _acc(22.4)) == (False, "")
