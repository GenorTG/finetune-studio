"""VRAM profiler — estimate and measure GPU memory for training configs.

WHAT THIS DOES
==============
Predicts and measures VRAM usage for different model sizes, training
methods (QLoRA, LoRA, full FT), and hyperparameter combinations.
Helps users find the maximum model size and settings that fit their GPU.

KEY CONCEPTS
============
- VRAM has 4 components: model weights, gradients, optimizer states, activations
- QLoRA reduces weights by 75% (4-bit NF4 vs 16-bit BF16)
- LoRA reduces gradients + optimizer to only adapter parameters
- Gradient checkpointing trades compute for memory (40-60% activation savings)
- Flash Attention 2 makes attention O(N) instead of O(N^2) in memory

LAYOUT
------
training/vram/
  __init__.py        — public API re-exports
  constants.py       — dtype byte sizes, overhead, model presets
  schema.py          — GPUInfo, VRAMEstimate, ProfileResult, RecommendedConfig
  gpu.py             — GPUInfo.detect()
  estimate.py        — estimate_vram() — the core formula
  recommend.py       — recommend_config(), recommend_for_model()
  profile.py         — profile_training(), profile_all_sizes()
  report.py          — generate_vram_report(), _parse_size()
"""

from finetune_studio.training.vram.constants import (
    ACTIVATION_SAFETY_MARGIN, CUDA_OVERHEAD_GB, DTYPE_BYTES, MODEL_PRESETS,
)
from finetune_studio.training.vram.gpu import GPUInfo, detect
from finetune_studio.training.vram.schema import ProfileResult, RecommendedConfig, VRAMEstimate
from finetune_studio.training.vram.estimate import estimate_vram
from finetune_studio.training.vram.recommend import recommend_config, recommend_for_model
from finetune_studio.training.vram.profile import profile_all_sizes, profile_training
from finetune_studio.training.vram.report import _parse_size, generate_vram_report
