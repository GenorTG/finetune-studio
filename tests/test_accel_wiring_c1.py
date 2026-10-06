"""Lane C1: training / loading / export route through ``finetune_studio.accel``.

Fake accelerators and fake ``transformers`` / ``llama_cpp`` prove, for every
vendor, that device_map / dtype / precision / optimizer / 4-bit / unsloth
decisions follow the hardware, that a CPU-only host gets fp32 with no bnb, no
unsloth and no GPU kwargs, and that the OOM ladder runs in the right order.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from finetune_studio import accel
from finetune_studio.accel import device as accel_device
from finetune_studio.accel.device import Accelerator
from finetune_studio.models import hf_loader
from finetune_studio.training import accel_plan


def _acc(kind: str, *, bf16: bool = True, idx: int = 0, cc: tuple[int, int] = (8, 6),
         four_bit: bool | None = None, reason: str = "") -> Accelerator:
    gpu = kind != "cpu"
    return Accelerator(
        kind=kind, index=idx, name=f"fake-{kind}", total_gb=24.0 if gpu else 0.0,
        free_gb=20.0 if gpu else 0.0, compute_capability=cc if kind == "cuda" else (0, 0),
        supports_bf16=bf16 and gpu, supports_flash_attention=kind == "cuda",
        supports_4bit=(kind in ("cuda", "rocm", "xpu")) if four_bit is None else four_bit,
        runtime=kind, degraded_reason=reason,
    )


CUDA, ROCM, XPU = _acc("cuda"), _acc("rocm"), _acc("xpu", idx=1)
MPS, CPU = _acc("mps", bf16=False), _acc("cpu")
PASCAL = _acc("cuda", bf16=False, cc=(6, 1))


@pytest.fixture
def pin(monkeypatch: pytest.MonkeyPatch):
    def _pin(acc: Accelerator, *, bnb: bool = True, unsloth: bool = True) -> Accelerator:
        monkeypatch.setattr(accel_device, "_cached", acc)
        monkeypatch.setattr(accel_plan, "_has_module",
                            lambda n: {"bitsandbytes": bnb, "unsloth": unsloth}.get(n, False))
        return acc
    return _pin


# --------------------------------------------------------------------------- plan

@pytest.mark.parametrize("acc,bf16,fp16,optim,unsloth", [
    (CUDA, True, False, "adamw_torch_fused", True),
    (ROCM, True, False, "adamw_torch_fused", False),   # unsloth is NVIDIA-only
    (XPU, True, False, "adamw_torch", False),
    (MPS, False, True, "adamw_torch", False),           # fp16 AMP, no bnb on Metal
    (PASCAL, False, True, "adamw_torch_fused", True),   # no bf16 -> fp16, never bf16=True
])
def test_gpu_plan_follows_vendor(acc, bf16, fp16, optim, unsloth, monkeypatch) -> None:
    monkeypatch.setattr(accel_plan, "_has_module", lambda _n: True)
    plan = accel_plan.resolve_train_plan(True, True, acc=acc)
    assert (plan.bf16, plan.fp16, plan.optim, plan.use_unsloth) == (bf16, fp16, optim, unsloth)
    assert plan.use_cpu is False


def test_cpu_plan_is_fp32_no_bnb_no_unsloth(pin) -> None:
    pin(CPU)
    plan = accel_plan.resolve_train_plan(True, True, want_8bit_optim=True)
    assert (plan.bf16, plan.fp16, plan.use_cpu) == (False, False, True)
    assert plan.optim == "adamw_torch" and not plan.load_4bit and not plan.use_unsloth
    assert accel.torch_dtype(CPU) is torch.float32
    assert accel.device_map(CPU) == "cpu"


def test_8bit_optimizer_only_where_bnb_works(pin) -> None:
    pin(CUDA, bnb=True)
    assert accel_plan.pick_optim(CUDA, want_8bit=True) == "adamw_bnb_8bit"
    pin(CUDA, bnb=False)
    assert accel_plan.pick_optim(CUDA, want_8bit=True) == "adamw_torch_fused"
    pin(MPS, bnb=True)
    assert accel_plan.pick_optim(MPS, want_8bit=True) == "adamw_torch"


def test_bf16_request_off_means_fp16_on_capable_gpu() -> None:
    plan = accel_plan.resolve_train_plan(False, False, acc=CUDA)
    assert (plan.bf16, plan.fp16) == (False, True)


@pytest.mark.parametrize("vendors,ver,expected", [
    (["NVIDIA"], "2.8.0", {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    (["AMD"], "2.7.1+rocm6.3", {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}),
    (["AMD"], "2.4.0+rocm6.1", {}),
    (["Intel"], "2.8.0", {}),
    (["Apple"], "2.8.0", {}),
    ([], "2.8.0", {}),
])
def test_alloc_conf_only_where_supported(vendors, ver, expected) -> None:
    assert accel_plan.alloc_conf_env(vendors, ver) == expected


# --------------------------------------------------------------------------- HF loading

class _FakeTransformers:
    """Records every from_pretrained call; ``plan`` is a list of exceptions/None per call."""

    def __init__(self, plan: list[BaseException | None]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._plan = list(plan)
        outer = self

        class Auto:
            @staticmethod
            def from_pretrained(_path: str, **kw: Any) -> str:
                outer.calls.append(kw)
                err = outer._plan.pop(0) if outer._plan else None
                if err is not None:
                    raise err
                return "model"

        class Bnb:
            def __init__(self, **kw: Any) -> None:
                self.kw = kw

        self.AutoModelForCausalLM, self.BitsAndBytesConfig = Auto, Bnb


@pytest.fixture
def fake_tf(monkeypatch: pytest.MonkeyPatch):
    def _make(plan: list[BaseException | None]) -> _FakeTransformers:
        ft = _FakeTransformers(plan)
        monkeypatch.setitem(sys.modules, "transformers", ft)  # type: ignore[arg-type]
        return ft
    return _make


@pytest.mark.parametrize("acc,dmap,dtype", [
    (CUDA, {"": 0}, torch.bfloat16),
    (ROCM, {"": 0}, torch.bfloat16),
    (XPU, {"": 1}, torch.bfloat16),
    (MPS, {"": "mps"}, torch.float16),
    (PASCAL, {"": 0}, torch.float16),
    (CPU, "cpu", torch.float32),
])
def test_load_places_full_model_on_chosen_device(acc, dmap, dtype, pin, fake_tf) -> None:
    pin(acc)
    ft = fake_tf([None])
    assert hf_loader.load_causal_lm("m") == "model"
    assert len(ft.calls) == 1
    assert ft.calls[0]["device_map"] == dmap and ft.calls[0]["torch_dtype"] is dtype
    assert "quantization_config" not in ft.calls[0]


def test_oom_ladder_full_then_4bit_then_auto(pin, fake_tf, monkeypatch) -> None:
    pin(CUDA)
    monkeypatch.setattr(accel, "empty_cache", lambda: None)
    oom = RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    ft = fake_tf([oom, oom, None])
    status: list[str] = []
    assert hf_loader.load_causal_lm("m", on_status=status.append) == "model"
    full, four, auto = ft.calls
    assert full["device_map"] == {"": 0} and "quantization_config" not in full
    assert four["quantization_config"].kw["load_in_4bit"] is True
    assert four["quantization_config"].kw["bnb_4bit_compute_dtype"] is torch.bfloat16
    assert four["device_map"] == {"": 0}
    assert auto["device_map"] == "auto" and "quantization_config" not in auto
    assert any("4-bit" in m for m in status) and any("offload" in m for m in status)


def test_oom_skips_4bit_when_backend_lacks_bnb(pin, fake_tf, monkeypatch) -> None:
    pin(MPS)
    monkeypatch.setattr(accel, "empty_cache", lambda: None)
    ft = fake_tf([RuntimeError("MPS backend out of memory"), None])
    hf_loader.load_causal_lm("m")
    assert [c["device_map"] for c in ft.calls] == [{"": "mps"}, "auto"]
    assert all("quantization_config" not in c for c in ft.calls)


def test_non_oom_error_is_not_retried(pin, fake_tf) -> None:
    pin(CUDA)
    ft = fake_tf([RuntimeError("shape mismatch in layer 3")])
    with pytest.raises(RuntimeError, match="shape mismatch"):
        hf_loader.load_causal_lm("m")
    assert len(ft.calls) == 1


def test_error_mentioning_bloom_is_not_mistaken_for_oom(pin, fake_tf) -> None:
    # D9: a bare "oom" substring matched BloomForCausalLM and walked the 4-bit/RAM-spill ladder.
    pin(CUDA)
    ft = fake_tf([RuntimeError("Error(s) in loading state_dict for BloomForCausalLM")])
    with pytest.raises(RuntimeError, match="BloomForCausalLM"):
        hf_loader.load_causal_lm("m")
    assert len(ft.calls) == 1


def test_cpu_only_never_touches_bnb_even_when_forced(pin, fake_tf) -> None:
    pin(CPU)
    ft = fake_tf([None])
    hf_loader.load_causal_lm("m", force_4bit=True)
    assert ft.calls[0]["device_map"] == "cpu" and ft.calls[0]["torch_dtype"] is torch.float32
    assert "quantization_config" not in ft.calls[0]


def test_cpu_only_oom_is_raised_not_retried(pin, fake_tf) -> None:
    pin(CPU)
    ft = fake_tf([RuntimeError("out of memory")])
    with pytest.raises(RuntimeError):
        hf_loader.load_causal_lm("m")
    assert len(ft.calls) == 1


def test_force_4bit_on_gpu_loads_quantized_first(pin, fake_tf) -> None:
    pin(XPU)
    ft = fake_tf([None])
    hf_loader.load_causal_lm("m", force_4bit=True)
    assert ft.calls[0]["quantization_config"].kw["load_in_4bit"] is True
    assert ft.calls[0]["device_map"] == {"": 1}


def test_merge_base_on_gpu_then_cpu_on_oom(pin, fake_tf, monkeypatch) -> None:
    pin(CUDA)
    monkeypatch.setattr(accel, "empty_cache", lambda: None)
    ft = fake_tf([RuntimeError("CUDA out of memory"), None])
    hf_loader.load_merge_base("base")
    assert [c["device_map"] for c in ft.calls] == [{"": 0}, "cpu"]
    assert all(c["torch_dtype"] is torch.bfloat16 for c in ft.calls)


def test_merge_base_cpu_host_goes_straight_to_cpu_bf16(pin, fake_tf) -> None:
    pin(CPU)
    ft = fake_tf([None])
    hf_loader.load_merge_base("base")
    assert ft.calls == [{"torch_dtype": torch.bfloat16, "device_map": "cpu", "trust_remote_code": True}]


def test_merge_base_fp16_on_gpu_without_bf16(pin, fake_tf) -> None:
    pin(PASCAL)
    ft = fake_tf([None])
    hf_loader.load_merge_base("base")
    assert ft.calls[0]["torch_dtype"] is torch.float16


# --------------------------------------------------------------------------- training args

def test_engine_sft_args_follow_plan(pin, monkeypatch, tmp_path) -> None:
    from finetune_studio.training import engine as eng_mod
    from finetune_studio.training import sft_args

    captured: dict[str, Any] = {}
    monkeypatch.setattr(sft_args, "build_sft_training_args", lambda **kw: captured.update(kw) or kw)

    def args_for(acc: Accelerator) -> dict[str, Any]:
        pin(acc)
        captured.clear()
        e = eng_mod.TrainingEngine()
        e.config = eng_mod.TrainingConfig(output_dir=str(tmp_path))
        e._build_sft_args(False)
        return dict(captured)

    cpu = args_for(CPU)
    assert (cpu["bf16"], cpu["fp16"], cpu["use_cpu"], cpu["optim"]) == (False, False, True, "adamw_torch")
    cuda = args_for(CUDA)
    assert (cuda["bf16"], cuda["fp16"], cuda["use_cpu"], cuda["optim"]) == (True, False, False, "adamw_torch_fused")
    pascal = args_for(PASCAL)
    assert (pascal["bf16"], pascal["fp16"]) == (False, True)


# --------------------------------------------------------------------------- vram.gpu

@pytest.mark.parametrize("acc", [CUDA, ROCM, XPU, MPS])
def test_vram_detect_fields_for_each_vendor(acc, monkeypatch) -> None:
    from finetune_studio.training.vram.gpu import detect
    monkeypatch.setattr(accel_device, "_detect", lambda: acc)
    accel_device.reset_cache()
    info = detect()
    assert info.name == acc.name and info.total_vram_gb == 24.0 and info.free_vram_gb == 20.0
    assert info.supports_bf16 is acc.supports_bf16
    assert info.supports_flash_attention is acc.supports_flash_attention
    assert "CUDA not available" not in info.name
    accel_device.reset_cache()


def test_vram_detect_cpu_surfaces_degraded_reason(monkeypatch) -> None:
    from finetune_studio.training.vram.gpu import detect
    reason = "GPU hardware detected (NVIDIA) but this PyTorch is a CPU-only build"
    monkeypatch.setattr(accel_device, "_detect", lambda: _acc("cpu", reason=reason))
    accel_device.reset_cache()
    info = detect()
    assert info.total_vram_gb == 0 and reason in info.name
    accel_device.reset_cache()


# --------------------------------------------------------------------------- abliteration

def test_abliteration_device_defaults_to_accelerator(pin) -> None:
    from finetune_studio.training import abliteration
    pin(CUDA)
    assert abliteration._resolve_device(None) == "cuda:0"
    pin(XPU)
    assert abliteration._resolve_device(None) == "xpu:1"
    pin(CPU)
    assert abliteration._resolve_device(None) == "cpu"
    assert abliteration._resolve_device("cuda:1") == "cuda:1"


# --------------------------------------------------------------------------- GGUF loader

@pytest.fixture
def llama_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from finetune_studio.models import llama_loader as mod

    calls: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    class Llama:
        def __init__(self, **kw: Any) -> None:
            calls.append(kw)
            if errors:
                raise errors.pop(0)

    stub = types.ModuleType("llama_cpp")
    stub.Llama = Llama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", stub)
    mod._LOGGED_WARNINGS.clear()
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    return mod, calls, errors, str(gguf)


def test_gguf_gets_main_gpu_extras_and_stays_full_offload(llama_env, monkeypatch) -> None:
    mod, calls, _errors, path = llama_env
    monkeypatch.setattr(mod, "llama_gpu_kwargs", lambda: ({"main_gpu": 1}, []))
    res = mod.load_llama_gguf(path, detect_mmproj=False)
    assert calls[0]["main_gpu"] == 1 and calls[0]["n_gpu_layers"] == -1
    assert res.warnings == []


def test_cpu_wheel_on_gpu_host_warning_is_surfaced_and_logged_once(llama_env, monkeypatch, caplog) -> None:
    mod, calls, _errors, path = llama_env
    warn = "NVIDIA GeForce is available but llama-cpp-python was built without GPU offload"
    monkeypatch.setattr(mod, "llama_gpu_kwargs", lambda: ({}, [warn]))
    with caplog.at_level("WARNING", logger=mod.log.name):
        r1 = mod.load_llama_gguf(path, detect_mmproj=False)
        r2 = mod.load_llama_gguf(path, detect_mmproj=False)
    assert r1.warnings == [warn] and r2.warnings == [warn]
    assert sum(warn in rec.getMessage() for rec in caplog.records) == 1
    assert "main_gpu" not in calls[0]


def test_cpu_only_host_passes_no_gpu_kwargs(llama_env, monkeypatch) -> None:
    mod, calls, _errors, path = llama_env
    from finetune_studio.accel import llama as accel_llama
    monkeypatch.setattr(mod, "llama_gpu_kwargs",
                        lambda: accel_llama.llama_gpu_kwargs(CPU, accel_llama.LlamaSupport(installed=True)))
    res = mod.load_llama_gguf(path, detect_mmproj=False)
    assert "main_gpu" not in calls[0] and res.warnings == []


@pytest.mark.parametrize("msg", [
    "ggml_vulkan: Failed to allocate buffer", "hipMalloc failed: out of memory",
    "Metal: failed to allocate buffer of size 2147483648", "SYCL error: out of device memory",
    "CUDA error: out of memory", "not enough VRAM",
])
def test_gguf_oom_retry_matches_every_backend(llama_env, monkeypatch, msg) -> None:
    mod, calls, errors, path = llama_env
    monkeypatch.setattr(mod, "llama_gpu_kwargs", lambda: ({}, []))
    errors.append(ValueError(msg))
    res = mod.load_llama_gguf(path, n_ctx=4096, detect_mmproj=False)
    # every backend's OOM wording triggers the same recovery: layers step down, the context never shrinks
    assert [c["n_ctx"] for c in calls] == [4096, 4096] and res.final_n_ctx == 4096
    assert [c["n_gpu_layers"] for c in calls] == [-1, 0]


def test_gguf_non_oom_error_raises_immediately(llama_env, monkeypatch) -> None:
    mod, calls, errors, path = llama_env
    monkeypatch.setattr(mod, "llama_gpu_kwargs", lambda: ({}, []))
    errors.append(ValueError("invalid magic number"))
    with pytest.raises(ValueError, match="magic"):
        mod.load_llama_gguf(path, detect_mmproj=False)
    assert len(calls) == 1


# --------------------------------------------------------------------------- misc

def test_worker_never_sets_allocator_env_on_import_for_non_nvidia(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_HIP_ALLOC_CONF", raising=False)
    monkeypatch.setattr("finetune_studio.accel.hardware_vendors", lambda: ["Intel"])
    assert accel_plan.apply_alloc_conf() == {}
    import os
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_is_oom_error_covers_typed_and_message_forms() -> None:
    assert accel.is_oom_error(RuntimeError("HIP out of memory"))
    assert accel.is_oom_error(RuntimeError("Failed to allocate 4 GiB"))
    assert not accel.is_oom_error(RuntimeError("size mismatch"))
    assert not accel.is_oom_error(ValueError("out of memory"))
