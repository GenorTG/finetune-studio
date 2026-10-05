"""Accelerator layer: GPU-first detection across vendors, CPU only without a GPU.

Torch is faked (``device._import_torch``) so every vendor path runs on any host.
"""
from __future__ import annotations

import json
import types
from typing import Any

import pytest

from finetune_studio import accel
from finetune_studio.accel import device, env, llama

GIB = 1024 ** 3


def _props(name: str, total_gb: float, major: int = 8, minor: int = 6) -> Any:
    return types.SimpleNamespace(name=name, total_memory=int(total_gb * GIB), major=major, minor=minor)


def _ns(devs: list[Any], bf16: bool = True) -> Any:
    return types.SimpleNamespace(
        is_available=lambda: bool(devs), device_count=lambda: len(devs),
        get_device_properties=lambda i: devs[i],
        mem_get_info=lambda i=0: (int(devs[i].total_memory * 0.9), devs[i].total_memory),
        is_bf16_supported=lambda: bf16,
    )


def _torch(cuda: list[Any] | None = None, hip: str | None = None, xpu: list[Any] | None = None,
           mps: bool = False, cuda_build: str | None = "13.0") -> Any:
    return types.SimpleNamespace(
        __version__="2.14.1", cuda=_ns(cuda or []), xpu=_ns(xpu or []) if xpu is not None else None,
        version=types.SimpleNamespace(cuda=None if hip else cuda_build, hip=hip, xpu=None),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("FTS_DEVICE", raising=False)
    monkeypatch.setattr(device, "hardware_vendors", list)
    device.reset_cache()
    yield
    device.reset_cache()


def _use(monkeypatch: pytest.MonkeyPatch, torch: Any) -> device.Accelerator:
    monkeypatch.setattr(device, "_import_torch", lambda: torch)
    device.reset_cache()
    return device.get_accelerator()


def test_nvidia_picks_largest_vram_not_index_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    acc = _use(monkeypatch, _torch(cuda=[_props("GTX 1070", 8, 6, 1), _props("RTX 3090", 24)]))
    assert (acc.kind, acc.index, acc.name) == ("cuda", 1, "RTX 3090")
    assert acc.torch_device == "cuda:1" and acc.is_gpu and acc.supports_bf16
    assert acc.supports_flash_attention and acc.runtime == "CUDA 13.0" and acc.device_count == 2
    assert accel.device_map(acc) == {"": 1}


def test_rocm_is_reported_as_rocm_and_shares_cuda_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    acc = _use(monkeypatch, _torch(cuda=[_props("AMD Radeon RX 7900 XTX", 24, 11, 0)], hip="7.0.1"))
    assert acc.kind == "rocm" and acc.torch_namespace == "cuda" and acc.torch_device == "cuda:0"
    assert acc.runtime == "ROCm 7.0.1" and not acc.supports_flash_attention
    assert accel.device_map(acc) == {"": 0}


def test_intel_xpu_is_used_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    acc = _use(monkeypatch, _torch(xpu=[_props("Intel Arc A770", 16, 0, 0)]))
    assert acc.kind == "xpu" and acc.torch_device == "xpu:0" and acc.total_gb == 16.0
    assert accel.device_map(acc) == {"": 0}


def test_apple_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    acc = _use(monkeypatch, _torch(mps=True))
    assert acc.kind == "mps" and acc.torch_device == "mps" and not acc.supports_bf16
    assert accel.device_map(acc) == {"": "mps"}


def test_cuda_beats_xpu_when_both_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    acc = _use(monkeypatch, _torch(cuda=[_props("RTX 4090", 24, 8, 9)], xpu=[_props("Arc", 16, 0, 0)]))
    assert acc.kind == "cuda"


def test_cpu_only_when_no_gpu_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    acc = _use(monkeypatch, _torch())
    assert acc.kind == "cpu" and not acc.is_gpu and acc.degraded_reason == ""
    assert accel.device_map(acc) == "cpu" and accel.auto_device_map(acc) == "cpu"
    assert acc.torch_device == "cpu" and acc.torch_namespace is None


def test_gpu_hardware_with_cpu_torch_is_loud_not_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "hardware_vendors", lambda: ["NVIDIA"])
    acc = _use(monkeypatch, _torch(cuda_build=None))
    assert acc.kind == "cpu"
    assert "CPU-only build" in acc.degraded_reason and "install.sh --repair" in acc.degraded_reason


def test_gpu_hardware_but_no_device_names_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "hardware_vendors", lambda: ["AMD"])
    acc = _use(monkeypatch, _torch(hip="7.0"))
    assert acc.kind == "cpu" and "no usable device" in acc.degraded_reason


def test_intel_igpu_with_cpu_wheel_is_not_nagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "hardware_vendors", lambda: ["Intel"])
    assert _use(monkeypatch, _torch(cuda_build=None)).degraded_reason == ""


def test_torch_missing_degrades_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "hardware_vendors", lambda: ["NVIDIA"])
    acc = _use(monkeypatch, None)
    assert acc.kind == "cpu" and "not importable" in acc.degraded_reason


def test_broken_backend_probe_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _torch(cuda=[_props("RTX", 24)], xpu=[_props("Arc", 16, 0, 0)])
    t.cuda.get_device_properties = lambda i: (_ for _ in ()).throw(RuntimeError("driver boom"))
    assert _use(monkeypatch, t).kind == "xpu"


def test_fts_device_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _torch(cuda=[_props("A", 8), _props("B", 24)])
    monkeypatch.setenv("FTS_DEVICE", "cuda:0")
    assert _use(monkeypatch, t).index == 0
    monkeypatch.setenv("FTS_DEVICE", "cpu")
    cpu = _use(monkeypatch, _torch(cuda=[_props("A", 8)]))
    assert cpu.kind == "cpu" and cpu.degraded_reason == ""


def test_dtype_follows_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    assert accel.torch_dtype(_use(monkeypatch, _torch(cuda=[_props("A", 24)]))) is torch.bfloat16
    t = _torch(cuda=[_props("old", 8, 6, 1)])
    t.cuda.is_bf16_supported = lambda: False
    assert accel.torch_dtype(_use(monkeypatch, t)) is torch.float16
    assert accel.torch_dtype(_use(monkeypatch, _torch())) is torch.float32


def test_to_dict_is_json_serialisable(monkeypatch: pytest.MonkeyPatch) -> None:
    acc = _use(monkeypatch, _torch(cuda=[_props("RTX 3090", 24)]))
    assert json.loads(json.dumps(acc.to_dict()))["torch_device"] == "cuda:0"


# ── visibility policy ─────────────────────────────────────────────────────────

NVIDIA_SMI = ("0, GPU-297ed2f9-b4d6, NVIDIA GeForce RTX 3090\n"
              "1, GPU-b6f0e3aa-5f4b, NVIDIA GeForce GTX 1070\n")


def _policy(environ: dict[str, str], smi: str = NVIDIA_SMI, rocm: str = "") -> dict[str, str]:
    return env.apply_device_policy(environ, nvidia=lambda _c: smi, amd=lambda _c: rocm)


def test_exclude_by_name_masks_other_gpu() -> None:
    e = {"FTS_GPU_EXCLUDE": "gtx 1070"}
    assert _policy(e) == {"CUDA_VISIBLE_DEVICES": "GPU-297ed2f9-b4d6"}
    assert e["CUDA_VISIBLE_DEVICES"] == "GPU-297ed2f9-b4d6"


def test_allowlist_by_index_and_uuid_prefix() -> None:
    assert _policy({"FTS_GPU_DEVICES": "1"}) == {"CUDA_VISIBLE_DEVICES": "GPU-b6f0e3aa-5f4b"}
    assert _policy({"FTS_GPU_DEVICES": "GPU-297e"}) == {"CUDA_VISIBLE_DEVICES": "GPU-297ed2f9-b4d6"}


def test_policy_that_hides_everything_is_ignored() -> None:
    e = {"FTS_GPU_EXCLUDE": "nvidia"}
    assert _policy(e) == {} and "CUDA_VISIBLE_DEVICES" not in e


def test_explicit_visible_devices_wins() -> None:
    e = {"FTS_GPU_EXCLUDE": "1070", "CUDA_VISIBLE_DEVICES": "1"}
    assert _policy(e) == {} and e["CUDA_VISIBLE_DEVICES"] == "1"


def test_no_policy_is_a_noop_and_never_shells_out() -> None:
    def boom(_c: list[str]) -> str:
        raise AssertionError("must not run nvidia-smi without a policy")
    assert env.apply_device_policy({}, nvidia=boom, amd=boom) == {}


def test_policy_noop_when_nothing_would_be_masked() -> None:
    assert _policy({"FTS_GPU_EXCLUDE": "radeon"}) == {}


def test_amd_policy_uses_hip_visible_devices() -> None:
    rocm = json.dumps({"card0": {"Card Series": "Radeon RX 7900 XTX"},
                       "card1": {"Card Series": "Radeon 780M"}, "system": {}})
    e = {"FTS_GPU_EXCLUDE": "780m"}
    assert env.apply_device_policy(e, nvidia=lambda _c: "", amd=lambda _c: rocm) == {"HIP_VISIBLE_DEVICES": "0"}


# ── llama.cpp backend capability ──────────────────────────────────────────────

SYSINFO = ("CUDA : ARCHS = 860 | USE_GRAPHS = 1 | FA_QUANTS = q4_0 | "
           "CPU : SSE3 = 1 | AVX2 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | ")


def test_parse_backends() -> None:
    assert llama.parse_backends(SYSINFO) == ("CUDA",)
    assert llama.parse_backends("Vulkan : x = 1 | CPU : AVX2 = 1 | Metal : y = 1") == ("VULKAN", "METAL")
    assert llama.parse_backends("CPU : AVX2 = 1 | ") == ()


def _acc(kind: str = "cuda", index: int = 1, count: int = 2) -> device.Accelerator:
    return device.Accelerator(kind=kind, index=index, name="RTX", total_gb=24, free_gb=20,
                              compute_capability=(8, 6), supports_bf16=True,
                              supports_flash_attention=True, supports_4bit=True,
                              runtime="CUDA 13", device_count=count)


def test_llama_kwargs_set_main_gpu_for_cuda_multi_gpu() -> None:
    sup = llama.LlamaSupport(True, "0.3.36", True, ("CUDA",))
    assert llama.llama_gpu_kwargs(_acc(), sup) == ({"main_gpu": 1}, [])


def test_llama_kwargs_no_main_gpu_for_vulkan_or_single_gpu() -> None:
    assert llama.llama_gpu_kwargs(_acc(), llama.LlamaSupport(True, "x", True, ("VULKAN",))) == ({}, [])
    assert llama.llama_gpu_kwargs(_acc(count=1), llama.LlamaSupport(True, "x", True, ("CUDA",))) == ({}, [])


def test_llama_cpu_wheel_on_gpu_host_warns() -> None:
    _kw, warns = llama.llama_gpu_kwargs(_acc(), llama.LlamaSupport(True, "0.3.36", False, ()))
    assert warns and "without GPU offload" in warns[0] and "--repair" in warns[0]


def test_llama_missing_warns_and_cpu_host_is_silent() -> None:
    _kw, warns = llama.llama_gpu_kwargs(_acc(), llama.LlamaSupport(False, error="ImportError: x"))
    assert warns and "not importable" in warns[0]
    cpu = device.Accelerator("cpu", 0, "CPU", 0, 0, (0, 0), False, False, False, "CPU")
    assert llama.llama_gpu_kwargs(cpu, llama.LlamaSupport(True, "x", False, ())) == ({}, [])
