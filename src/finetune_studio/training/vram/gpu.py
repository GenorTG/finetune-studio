"""GPU detection — auto-discover NVIDIA GPU properties.

Single responsibility: turn `nvidia-smi` / PyTorch into a GPUInfo dataclass.
"""
from __future__ import annotations

from finetune_studio.training.vram.schema import GPUInfo


def detect() -> GPUInfo:
    """Auto-detect GPU properties. Falls back to a 'Unknown' GPUInfo if no CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected")
        props = torch.cuda.get_device_properties(0)
        total_bytes = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        free_bytes = torch.cuda.mem_get_info(0)[0]
        cc = (props.major, props.minor)
        # Flash Attention 2 requires Ampere+ (compute capability 8.0+)
        # RTX 3090 = 8.6, RTX 4090 = 8.9
        supports_fa2 = cc[0] >= 8
        supports_bf16 = cc[0] >= 8  # Ampere+ supports BF16
        return GPUInfo(
            name=props.name,
            total_vram_gb=round(total_bytes / (1024**3), 1),
            free_vram_gb=round(free_bytes / (1024**3), 1),
            compute_capability=cc,
            supports_flash_attention=supports_fa2,
            supports_bf16=supports_bf16,
        )
    except Exception:
        return GPUInfo(
            name="Unknown (CUDA not available)",
            total_vram_gb=0,
            free_vram_gb=0,
            compute_capability=(0, 0),
            supports_flash_attention=False,
            supports_bf16=False,
        )
