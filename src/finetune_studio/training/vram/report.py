"""Markdown report generator + size-string parser.

Single responsibility: produce the human-readable VRAM report and parse
strings like "7B" -> 7.0.
"""
from __future__ import annotations

from finetune_studio.training.vram.constants import MODEL_PRESETS
from finetune_studio.training.vram.gpu import detect as detect_gpu
from finetune_studio.training.vram.recommend import recommend_config


def _parse_size(label: str) -> float:
    """Parse '7B', '1.5B', '27b' etc. to float."""
    label = label.lower().replace(",", "")
    for suffix in ("b", "bn"):
        if label.endswith(suffix):
            return float(label[:-len(suffix)])
    return float(label)


def generate_vram_report(
    available_vram_gb: float | None = None,
    output_path: str | None = None,
) -> str:
    """Generate a full VRAM report for the current GPU.

    Includes:
    - GPU info
    - Recommendations for all known model sizes
    - Comparison table

    Returns:
        Markdown-formatted report.
    """
    gpu = detect_gpu()
    if available_vram_gb is None:
        available_vram_gb = gpu.free_vram_gb

    lines = [
        "# VRAM Training Report",
        "",
        f"**GPU:** {gpu.name}",
        f"**Total VRAM:** {gpu.total_vram_gb} GB",
        f"**Available VRAM:** {gpu.free_vram_gb} GB",
        f"**Compute Capability:** {gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
        f"**Flash Attention 2:** {'✅' if gpu.supports_flash_attention else '❌'}",
        f"**BF16:** {'✅' if gpu.supports_bf16 else '❌'}",
        "",
        "## Recommendations",
        "",
        "| Model | Method | Est. VRAM | Headroom | Fits? | Max Batch | Max Seq | Notes |",
        "|-------|--------|-----------|----------|-------|-----------|---------|-------|",
    ]

    for preset_name, preset in sorted(MODEL_PRESETS.items(), key=lambda x: x[1]["params_b"]):
        configs = recommend_config(
            model_size_b=preset["params_b"],
            available_vram_gb=available_vram_gb,
        )
        if configs:
            best = configs[0]
            fit_icon = "✅" if best.fits else "❌"
            lines.append(
                f"| {preset_name} ({preset['params_b']}B) "
                f"| {best.method} "
                f"| {best.estimated_vram_gb:.1f} GB "
                f"| {available_vram_gb - best.estimated_vram_gb:.1f} GB "
                f"| {fit_icon} "
                f"| {best.batch_size} "
                f"| {best.max_seq_length} "
                f"| {best.notes} |"
            )
        else:
            lines.append(
                f"| {preset_name} ({preset['params_b']}B) "
                f"| — "
                f"| — "
                f"| — "
                f"| ❌ "
                f"| — "
                f"| — "
                f"| Doesn't fit in {available_vram_gb}GB |"
            )

    lines.extend([
        "",
        "## VRAM Breakdown Formula",
        "",
        "```",
        "Total = CUDA_overhead + Weights + Gradients + Optimizer + Activations",
        "",
        "QLoRA:",
        "  Weights = params × 0.5 bytes (NF4)",
        "  Gradients = adapter_params × 2 bytes (BF16)",
        "  Optimizer = adapter_params × 8 bytes (AdamW FP32)",
        "  Activations ≈ 6 × batch × seq × hidden × layers × 0.5 (grad checkpoint)",
        "",
        "LoRA (16-bit):",
        "  Weights = params × 2 bytes (BF16)",
        "  Gradients/Optimizer = same as QLoRA (only adapter)",
        "",
        "Full FT:",
        "  Weights = params × 2 bytes (BF16)",
        "  Gradients = params × 2 bytes (BF16)",
        "  Optimizer = params × 8 bytes (AdamW FP32)",
        "```",
        "",
        "## Optimization Tips",
        "",
        "- **Gradient checkpointing**: 40-60% activation savings, 25-30% more compute",
        "- **Flash Attention 2**: O(N) attention memory (Ampere+ GPU required)",
        "- **8-bit optimizer (AdamW)**: ~50% optimizer memory savings",
        "- **Lower LoRA rank**: rank=16 uses 4x less adapter memory than rank=64",
        "- **Shorter sequences**: Halving seq_length roughly halves activation memory",
        "- **Unsloth**: 2-5x memory reduction + speed for QLoRA training",
    ])

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report
