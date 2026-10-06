"""Vendor-neutral accelerator layer: detect once, route everything through it.

Policy (Genor 2026-10-05): always try the GPU first — NVIDIA CUDA, AMD ROCm,
Intel XPU or Apple Metal — and use the CPU only when the host has no usable
GPU at all. Model loading, inference, training, merge/export and GGUF
offload all ask this package instead of hard-coding ``cuda`` / ``{"": 0}``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.accel.device import (
    Accelerator,
    get_accelerator,
    hardware_vendors,
    reset_cache,
)
from finetune_studio.accel.env import apply_device_policy, isolated_env
from finetune_studio.accel.llama import LlamaSupport, llama_gpu_kwargs, llama_support
from finetune_studio.accel.ops import (
    activate,
    auto_device_map,
    device_map,
    empty_cache,
    enable_fast_matmul,
    is_oom_error,
    is_oom_message,
    mem_info_gb,
    oom_errors,
    peak_memory_gb,
    pin_trainer_args,
    reset_peak_memory,
    synchronize,
    torch_dtype,
)


def describe() -> dict[str, Any]:
    """One JSON-able snapshot for /api/system/accelerator and diagnostics."""
    import os
    acc = get_accelerator()
    llama = llama_support()
    try:
        import torch
        torch_info: dict[str, Any] = {
            "version": torch.__version__, "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None),
            "xpu": getattr(torch.version, "xpu", None),
        }
    except Exception as exc:  # noqa: BLE001
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "accelerator": acc.to_dict(),
        "torch": torch_info,
        "llama_cpp": llama.to_dict(),
        "hardware": hardware_vendors(),
        "policy": {k: os.environ[k] for k in
                   ("FTS_DEVICE", "FTS_GPU_DEVICES", "FTS_GPU_EXCLUDE", "CUDA_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES") if k in os.environ},
    }


__all__ = [
    "Accelerator", "LlamaSupport", "activate", "apply_device_policy", "auto_device_map", "describe",
    "device_map", "empty_cache", "enable_fast_matmul", "get_accelerator", "hardware_vendors",
    "is_oom_error", "is_oom_message", "isolated_env",
    "llama_gpu_kwargs", "llama_support", "mem_info_gb", "oom_errors", "peak_memory_gb",
    "pin_trainer_args",
    "reset_cache", "reset_peak_memory", "synchronize", "torch_dtype",
]
