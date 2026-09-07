"""Back-compat shim — the real code lives in finetune_studio.training.vram/.

Older code that did `from finetune_studio.training.vram_profiler import estimate_vram`
keeps working unchanged.
"""

from finetune_studio.training.vram import (  # noqa: F401
    ACTIVATION_SAFETY_MARGIN,
    CUDA_OVERHEAD_GB,
    DTYPE_BYTES,
    GPUInfo,
    MODEL_PRESETS,
    ProfileResult,
    RecommendedConfig,
    VRAMEstimate,
    estimate_vram,
    generate_vram_report,
    profile_all_sizes,
    profile_training,
    recommend_config,
    recommend_for_model,
)
