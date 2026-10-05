"""GPU detection — one GPUInfo from the vendor-neutral accelerator layer.

Single responsibility: turn ``finetune_studio.accel`` (NVIDIA CUDA, AMD ROCm,
Intel XPU, Apple MPS) into a GPUInfo dataclass. A CPU-only host gets a clearly
labelled "no GPU" record, with the accelerator's reason when a GPU is present
but unusable (CPU torch wheel, missing driver).
"""
from __future__ import annotations

from finetune_studio.training.vram.schema import GPUInfo


def detect() -> GPUInfo:
    """Auto-detect accelerator properties; zeroed ``GPUInfo`` when there is no usable GPU."""
    from finetune_studio import accel
    acc = accel.get_accelerator(refresh=True)
    if acc.is_gpu:
        return GPUInfo(
            name=acc.name,
            total_vram_gb=round(acc.total_gb, 1),
            free_vram_gb=round(acc.free_gb, 1),
            compute_capability=acc.compute_capability,
            supports_flash_attention=acc.supports_flash_attention,
            supports_bf16=acc.supports_bf16,
        )
    reason = acc.degraded_reason or "no GPU detected"
    return GPUInfo(
        name=f"No GPU ({reason})",
        total_vram_gb=0,
        free_vram_gb=0,
        compute_capability=(0, 0),
        supports_flash_attention=False,
        supports_bf16=False,
    )
