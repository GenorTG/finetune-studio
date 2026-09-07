"""Config recommendations for a model + VRAM combination.

Single responsibility: brute-force search over (method, batch, seq) combos
that fit in the available VRAM, then sort by quality.
"""
from __future__ import annotations

from finetune_studio.training.vram.constants import MODEL_PRESETS
from finetune_studio.training.vram.estimate import estimate_vram
from finetune_studio.training.vram.gpu import detect as detect_gpu
from finetune_studio.training.vram.schema import RecommendedConfig


def recommend_config(
    model_size_b: float,
    available_vram_gb: float | None = None,
    target_seq_length: int = 2048,
    lora_rank: int = 64,
) -> list[RecommendedConfig]:
    """Generate recommended training configs for a model size + VRAM budget.

    Returns configs sorted by quality (best first), filtering out OOM configs.
    """
    if available_vram_gb is None:
        gpu = detect_gpu()
        available_vram_gb = gpu.free_vram_gb

    configs = []

    # Try different method/batch/seq combinations
    for method in ("qlora", "lora", "full_ft"):
        for bs in (1, 2, 4, 8):
            for seq in (512, 1024, 2048, 4096):
                if seq > target_seq_length:
                    continue
                est = estimate_vram(
                    model_size_b=model_size_b,
                    method=method,
                    batch_size=bs,
                    seq_length=seq,
                    lora_rank=lora_rank,
                    available_vram_gb=available_vram_gb,
                )
                if est.fits:
                    # Quality score: prefer longer seq, bigger batch, higher method quality
                    method_quality = {"full_ft": 3, "lora": 2, "qlora": 1}[method]
                    score = method_quality * 1000 + seq * 10 + bs * 100

                    notes_parts = []
                    if method == "qlora":
                        notes_parts.append("4-bit quantized — slight quality tradeoff")
                    if method == "full_ft":
                        notes_parts.append("Full fine-tuning — best quality")
                    if bs == 1:
                        notes_parts.append("Small batch — use gradient accumulation")
                    if seq >= 4096:
                        notes_parts.append("Long context — slower training")

                    configs.append(RecommendedConfig(
                        model_size_b=model_size_b,
                        method=method,
                        lora_rank=lora_rank,
                        batch_size=bs,
                        gradient_accumulation=max(1, 8 // bs),  # Target effective batch ~8
                        max_seq_length=seq,
                        estimated_vram_gb=round(est.total_gb, 2),
                        fits=True,
                        notes="; ".join(notes_parts),
                    ))

    # Sort by quality (best first)
    configs.sort(key=lambda c: (
        {"full_ft": 3, "lora": 2, "qlora": 1}[c.method],
        c.max_seq_length,
        c.batch_size,
    ), reverse=True)

    # Return top 5
    return configs[:5]


def recommend_for_model(
    model_name: str,
    available_vram_gb: float | None = None,
) -> list[RecommendedConfig]:
    """Get recommendations for a known model name."""
    key = model_name.lower().replace(" ", "-")
    if key not in MODEL_PRESETS:
        # Try partial match
        for preset_key in MODEL_PRESETS:
            if key in preset_key or preset_key in key:
                key = preset_key
                break
        else:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Known: {', '.join(sorted(MODEL_PRESETS.keys()))}"
            )

    preset = MODEL_PRESETS[key]
    return recommend_config(
        model_size_b=preset["params_b"],
        available_vram_gb=available_vram_gb,
        lora_rank=64,
    )
