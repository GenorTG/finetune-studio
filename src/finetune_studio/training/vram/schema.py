"""Dataclasses for VRAM profiling.

Single responsibility: type definitions for GPU info, estimates, profiles, and recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    name: str
    total_vram_gb: float
    free_vram_gb: float
    compute_capability: tuple[int, int]
    supports_flash_attention: bool
    supports_bf16: bool


@dataclass
class VRAMEstimate:
    """Estimated VRAM breakdown for a training config."""
    model_weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    total_gb: float
    available_gb: float
    headroom_gb: float
    fits: bool
    method: str  # "qlora", "lora", "full_ft"
    model_size_b: float  # billions of params
    max_batch_size: int
    max_seq_length: int

    def to_dict(self) -> dict:
        return {
            "model_weights_gb": round(self.model_weights_gb, 2),
            "gradients_gb": round(self.gradients_gb, 2),
            "optimizer_states_gb": round(self.optimizer_states_gb, 2),
            "activations_gb": round(self.activations_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "headroom_gb": round(self.headroom_gb, 2),
            "fits": self.fits,
            "method": self.method,
            "model_size_b": self.model_size_b,
            "max_batch_size": self.max_batch_size,
            "max_seq_length": self.max_seq_length,
        }


@dataclass
class ProfileResult:
    """Actual measured VRAM from a profiling run."""
    model_name: str
    method: str
    batch_size: int
    seq_length: int
    lora_rank: int
    peak_vram_gb: float
    measured_model_gb: float
    measured_adapters_gb: float
    training_steps: int
    step_time_s: float
    success: bool
    error: str = ""


@dataclass
class RecommendedConfig:
    """A recommended training config for a given model+VRAM combination."""
    model_size_b: float
    method: str
    lora_rank: int
    batch_size: int
    gradient_accumulation: int
    max_seq_length: int
    estimated_vram_gb: float
    fits: bool
    notes: str = ""
