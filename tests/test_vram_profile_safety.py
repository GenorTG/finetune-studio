"""Safety regressions for training.vram.profile.profile_training (CUDA is mocked)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from finetune_studio.training.vram import profile as profile_mod


def _fake_torch(reset_error: Exception | None = None) -> types.ModuleType:
    torch = types.ModuleType("torch")
    cuda = types.SimpleNamespace()

    def reset_peak_memory_stats(device: int | None = None) -> None:
        if reset_error is not None:
            raise reset_error

    cuda.reset_peak_memory_stats = reset_peak_memory_stats
    cuda.is_available = lambda: False
    torch.cuda = cuda  # type: ignore[attr-defined]
    torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    return torch


def _pin_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the process-wide accelerator to a CUDA box so the test is host-independent."""
    from finetune_studio.accel import device as accel_device
    monkeypatch.setattr(accel_device, "_cached", accel_device.Accelerator(
        kind="cuda", index=0, name="Fake GPU", total_gb=24.0, free_gb=20.0,
        compute_capability=(8, 6), supports_bf16=True, supports_flash_attention=True,
        supports_4bit=True, runtime="CUDA 12.8"))


@pytest.fixture
def no_ml_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the heavy imports fail fast so the profiling body errors deterministically."""
    for name in ("peft", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)


def test_cuda_init_failure_is_structured_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _pin_cuda(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(RuntimeError("CUDA driver init failed")))
    result = profile_mod.profile_training("some/model", output_dir=str(tmp_path))
    assert result.success is False
    assert "CUDA driver init failed" in result.error
    assert result.peak_vram_gb == 0


def test_caller_supplied_output_dir_is_never_deleted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_ml_stack: None
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    user_dir = tmp_path / "my-precious-data"
    user_dir.mkdir()
    (user_dir / "keep.txt").write_text("important")
    (user_dir / "sub").mkdir()
    (user_dir / "sub" / "nested.txt").write_text("also important")

    result = profile_mod.profile_training("some/model", output_dir=str(user_dir))

    assert result.success is False  # ML stack stubbed out; body fails after dir setup
    assert (user_dir / "keep.txt").read_text() == "important"
    assert (user_dir / "sub" / "nested.txt").read_text() == "also important"


def test_profiler_removes_only_scratch_dir_it_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_ml_stack: None
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    created: list[str] = []
    real_mkdtemp = profile_mod.tempfile.mkdtemp

    def spy(*a, **kw):  # type: ignore[no-untyped-def]
        path = real_mkdtemp(*a, **kw)
        created.append(path)
        return path

    monkeypatch.setattr(profile_mod.tempfile, "mkdtemp", spy)
    base = tmp_path / "base"
    profile_mod.profile_training("some/model", output_dir=str(base))

    assert len(created) == 1
    assert Path(created[0]).parent == base
    assert not Path(created[0]).exists()  # scratch removed
    assert base.exists()  # caller-supplied parent kept
