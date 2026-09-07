"""Core VRAM estimation formula.

Single responsibility: given a (model_size, method, batch, seq, lora_rank),
produce a VRAMEstimate with the 4-component breakdown.

Method-specific notes:
  QLoRA   — 4-bit NF4 weights; gradients/optimizer only on adapter params
  LoRA    — 16-bit BF16 weights; gradients/optimizer only on adapter params
  Full FT — 16-bit BF16 weights; gradients/optimizer on EVERY param

Activation estimate: ~6 bytes * batch * seq * hidden * layers, halved if
gradient checkpointing is on.
"""
from __future__ import annotations

from finetune_studio.training.vram.constants import (
    ACTIVATION_SAFETY_MARGIN, CUDA_OVERHEAD_GB, DTYPE_BYTES,
)
from finetune_studio.training.vram.gpu import detect as detect_gpu
from finetune_studio.training.vram.schema import VRAMEstimate


def estimate_vram(
    model_size_b: float,
    method: str = "qlora",
    batch_size: int = 2,
    seq_length: int = 2048,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    num_layers: int | None = None,
    hidden_size: int | None = None,
    gradient_checkpointing: bool = True,
    available_vram_gb: float | None = None,
) -> VRAMEstimate:
    """Estimate VRAM usage for a training configuration.

    Args:
        model_size_b: Model size in billions of parameters.
        method: "qlora" (4-bit), "lora" (16-bit), or "full_ft" (16-bit full).
        batch_size: Per-device batch size.
        seq_length: Maximum sequence length in tokens.
        lora_rank: LoRA rank (r). Only affects adapter size.
        lora_alpha: LoRA alpha. Typically 2 * rank.
        num_layers: Number of transformer layers (for activation estimation).
        hidden_size: Hidden dimension (for activation estimation).
        gradient_checkpointing: Whether to use gradient checkpointing.
        available_vram_gb: Available VRAM. Auto-detected if None.

    Returns:
        VRAMEstimate with full breakdown.
    """
    # Auto-detect VRAM if not provided
    if available_vram_gb is None:
        gpu = detect_gpu()
        available_vram_gb = gpu.free_vram_gb

    num_params = model_size_b * 1e9

    # --- Model weights ---
    if method == "qlora":
        # 4-bit NF4 quantized weights
        weights_bytes = num_params * DTYPE_BYTES["nf4"]
    elif method == "lora":
        # 16-bit BF16 weights
        weights_bytes = num_params * DTYPE_BYTES["bfloat16"]
    else:  # full_ft
        weights_bytes = num_params * DTYPE_BYTES["bfloat16"]

    model_weights_gb = weights_bytes / (1024**3)

    # --- Gradients ---
    if method in ("qlora", "lora"):
        # Only adapter parameters have gradients
        # LoRA params per layer: 2 * (hidden * rank) for Q/K/V/O + gate/up/down
        # Rough estimate: ~0.1% of total params for rank=64
        adapter_params = num_params * min(lora_rank / (hidden_size or 4096), 0.05)
        gradients_bytes = adapter_params * DTYPE_BYTES["bfloat16"]
    else:
        # Full FT: gradients match weights
        gradients_bytes = num_params * DTYPE_BYTES["bfloat16"]

    gradients_gb = gradients_bytes / (1024**3)

    # --- Optimizer states ---
    # AdamW: 2 moments (m, v) in FP32 = 8 bytes per trainable param
    if method in ("qlora", "lora"):
        trainable_params = adapter_params if 'adapter_params' in dir() else num_params * 0.001
        optimizer_bytes = trainable_params * 8
    else:
        optimizer_bytes = num_params * 8

    optimizer_states_gb = optimizer_bytes / (1024**3)

    # --- Activations ---
    # Rough estimate: ~6 bytes * batch * seq_len * hidden * num_layers
    # With gradient checkpointing: ~40-60% reduction
    _hidden = hidden_size or 4096
    _layers = num_layers or 32
    activation_bytes = (
        6 * batch_size * seq_length * _hidden * _layers
    )
    if gradient_checkpointing:
        activation_bytes = int(activation_bytes * 0.5)  # 50% reduction
    activations_gb = (activation_bytes / (1024**3)) * ACTIVATION_SAFETY_MARGIN

    # --- Total ---
    total_gb = (
        CUDA_OVERHEAD_GB
        + model_weights_gb
        + gradients_gb
        + optimizer_states_gb
        + activations_gb
    )

    headroom_gb = available_vram_gb - total_gb
    fits = headroom_gb > 0.5  # At least 0.5GB headroom

    return VRAMEstimate(
        model_weights_gb=model_weights_gb,
        gradients_gb=gradients_gb,
        optimizer_states_gb=optimizer_states_gb,
        activations_gb=activations_gb,
        total_gb=total_gb,
        available_gb=available_vram_gb,
        headroom_gb=headroom_gb,
        fits=fits,
        method=method,
        model_size_b=model_size_b,
        max_batch_size=batch_size,
        max_seq_length=seq_length,
    )
