"""Constants for VRAM estimation — dtype sizes, overhead, model presets.

Single responsibility: numbers and lookup tables used everywhere else in this package.
"""
from __future__ import annotations


# Bytes per parameter by dtype
DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
    "nf4": 0.5,
}

# VRAM overhead constants (GB) -- empirical from RTX 3090 measurements
CUDA_OVERHEAD_GB = 0.5  # CUDA context, cuDNN, etc.
ACTIVATION_SAFETY_MARGIN = 1.15  # 15% buffer for activation estimation

# Model presets -- known params/hidden/layers for popular models
MODEL_PRESETS: dict[str, dict] = {
    "qwen2.5-0.5b": {"params_b": 0.5, "hidden": 896, "layers": 24},
    "qwen2.5-1.5b": {"params_b": 1.5, "hidden": 1536, "layers": 28},
    "qwen2.5-3b": {"params_b": 3, "hidden": 2048, "layers": 36},
    "qwen2.5-7b": {"params_b": 7, "hidden": 3584, "layers": 28},
    "qwen2.5-14b": {"params_b": 14, "hidden": 5120, "layers": 40},
    "qwen2.5-27b": {"params_b": 27, "hidden": 5120, "layers": 64},
    "gemma-2-2b": {"params_b": 2, "hidden": 2304, "layers": 26},
    "gemma-2-9b": {"params_b": 9, "hidden": 3584, "layers": 42},
    "gemma-2-27b": {"params_b": 27, "hidden": 4608, "layers": 46},
    "phi-4-mini": {"params_b": 3.8, "hidden": 3072, "layers": 32},
    "phi-4": {"params_b": 14, "hidden": 5120, "layers": 40},
    "llama-3.1-8b": {"params_b": 8, "hidden": 4096, "layers": 32},
    "llama-3.1-70b": {"params_b": 70, "hidden": 8192, "layers": 80},
}
