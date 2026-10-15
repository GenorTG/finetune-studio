"""Training precision / optimizer / quantization decisions, derived from the accelerator.

Single responsibility: turn one ``accel.Accelerator`` into the flags the
trainer needs, so no training code hard-codes ``bf16=True``, ``adamw_8bit``,
``load_in_4bit`` or "unsloth is always available". GPU first on every vendor;
the CPU gets fp32, plain AdamW, no bitsandbytes and no unsloth.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any

from finetune_studio.accel import Accelerator, get_accelerator


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def bnb_usable(acc: Accelerator) -> bool:
    """bitsandbytes is importable AND this backend can run its kernels."""
    return acc.is_gpu and acc.supports_4bit and _has_module("bitsandbytes")


def unsloth_supported(acc: Accelerator) -> bool:
    """unsloth is NVIDIA-centric (triton + CUDA kernels): CUDA only, and only if installed."""
    return acc.kind == "cuda" and _has_module("unsloth")


@dataclass(frozen=True)
class TrainPlan:
    bf16: bool           # TrainingArguments.bf16
    fp16: bool           # TrainingArguments.fp16 (AMP loss-scaled)
    use_cpu: bool        # TrainingArguments.use_cpu
    optim: str           # HF optimizer name
    load_4bit: bool      # QLoRA-style 4-bit base is allowed on this backend
    use_unsloth: bool    # unsloth fast path allowed
    kind: str            # accelerator kind, for logs


def pick_optim(acc: Accelerator, *, want_8bit: bool = False) -> str:
    """Optimizer name: 8-bit bnb only where bitsandbytes works, else (fused) AdamW."""
    if want_8bit and bnb_usable(acc):
        return "adamw_bnb_8bit"
    if acc.kind in ("cuda", "rocm"):
        return "adamw_torch_fused"
    return "adamw_torch"


def resolve_train_plan(
    want_bf16: bool = True, want_unsloth: bool = True, *,
    acc: Accelerator | None = None, want_8bit_optim: bool = False,
) -> TrainPlan:
    """Precision follows the hardware: bf16 only if supported, fp16 AMP on other GPUs, fp32 on CPU."""
    acc = acc or get_accelerator()
    if not acc.is_gpu:
        bf16 = fp16 = False
    elif acc.supports_bf16:
        bf16, fp16 = bool(want_bf16), not want_bf16
    else:
        bf16, fp16 = False, True
    return TrainPlan(
        bf16=bf16, fp16=fp16, use_cpu=not acc.is_gpu,
        optim=pick_optim(acc, want_8bit=want_8bit_optim),
        load_4bit=bnb_usable(acc),
        use_unsloth=bool(want_unsloth) and unsloth_supported(acc),
        kind=acc.kind,
    )


def bnb_4bit_config(acc: Accelerator | None = None) -> Any:
    """``BitsAndBytesConfig`` for a 4-bit NF4 load with the accelerator's compute dtype."""
    from transformers import BitsAndBytesConfig

    from finetune_studio.accel import torch_dtype
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype(acc),
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )


def alloc_conf_env(vendors: list[str], torch_version: str = "") -> dict[str, str]:
    """Env that enables expandable allocator segments where the backend honours it.

    Called before torch is imported (worker start), so it reads sysfs vendors
    rather than the accelerator. NVIDIA: ``PYTORCH_CUDA_ALLOC_CONF``. AMD: the
    HIP variant, only on torch >= 2.6 (earlier ROCm builds ignore/choke on it).
    Never for Intel/Apple/CPU hosts.
    """
    val = "expandable_segments:True"
    if "NVIDIA" in vendors:
        return {"PYTORCH_CUDA_ALLOC_CONF": val}
    if "AMD" in vendors:
        try:
            major, minor = (int(x) for x in torch_version.split("+")[0].split(".")[:2])
        except ValueError:
            return {}
        if (major, minor) >= (2, 6):
            return {"PYTORCH_HIP_ALLOC_CONF": val}
    return {}


def apply_alloc_conf() -> dict[str, str]:
    """``setdefault`` the allocator env for this host; returns what was applied."""
    from importlib import metadata

    from finetune_studio.accel import hardware_vendors
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        version = ""
    applied = alloc_conf_env(hardware_vendors(), version)
    for key, val in applied.items():
        os.environ.setdefault(key, val)
    return applied
