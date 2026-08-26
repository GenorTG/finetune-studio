"""VRAM profiler — estimate and measure GPU memory for training configs.

WHAT THIS FILE DOES
==================
Predicts and measures VRAM usage for different model sizes, training
methods (QLoRA, LoRA, full FT), and hyperparameter combinations.
Helps users find the maximum model size and settings that fit their GPU.

KEY CONCEPTS
============
- VRAM has 4 components: model weights, gradients, optimizer states, activations
- QLoRA reduces weights by 75% (4-bit NF4 vs 16-bit BF16)
- LoRA reduces gradients + optimizer to only adapter parameters
- Gradient checkpointing trades compute for memory (40-60% activation savings)
- Flash Attention 2 makes attention O(N) instead of O(N²) in memory
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Bytes per parameter by dtype
DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
    "nf4": 0.5,
}

# VRAM overhead constants (GB) — empirical from RTX 3090 measurements
CUDA_OVERHEAD_GB = 0.5  # CUDA context, cuDNN, etc.
ACTIVATION_SAFETY_MARGIN = 1.15  # 15% buffer for activation estimation


@dataclass
class GPUInfo:
    name: str
    total_vram_gb: float
    free_vram_gb: float
    compute_capability: tuple[int, int]
    supports_flash_attention: bool
    supports_bf16: bool

    @classmethod
    def detect(cls) -> GPUInfo:
        """Auto-detect GPU properties."""
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
            return cls(
                name=props.name,
                total_vram_gb=round(total_bytes / (1024**3), 1),
                free_vram_gb=round(free_bytes / (1024**3), 1),
                compute_capability=cc,
                supports_flash_attention=supports_fa2,
                supports_bf16=supports_bf16,
            )
        except Exception as e:
            return cls(
                name="Unknown (CUDA not available)",
                total_vram_gb=0,
                free_vram_gb=0,
                compute_capability=(0, 0),
                supports_flash_attention=False,
                supports_bf16=False,
            )


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
        gpu = GPUInfo.detect()
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
        gpu = GPUInfo.detect()
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


# Model presets — known params/hidden/layers for popular models
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


def profile_training(
    model_path: str,
    method: str = "qlora",
    batch_size: int = 1,
    seq_length: int = 512,
    lora_rank: int = 16,
    num_steps: int = 5,
    output_dir: str = "/tmp/vram_profile",
) -> ProfileResult:
    """Run a short training job and measure actual peak VRAM.

    This does actual training (not estimation) to get real numbers.
    Uses a tiny synthetic dataset to minimize time.

    Args:
        model_path: HuggingFace model ID or local path.
        method: "qlora" or "lora".
        batch_size: Per-device batch size.
        seq_length: Sequence length.
        lora_rank: LoRA rank (keep small for profiling).
        num_steps: How many steps to train (5 is enough for measurement).
        output_dir: Where to save (temporary).

    Returns:
        ProfileResult with actual measurements.
    """
    import torch

    # Create synthetic training data
    synthetic_data = [
        {"messages": [{"role": "user", "content": f"What is {i}?"}, {"role": "assistant", "content": f"The answer to {i} is {i * 2}."}]}
        for i in range(max(batch_size * 2, 10))
    ]

    peak_vram_before = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    success = True
    error = ""

    try:
        if method == "qlora":
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model, r=lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_rank * 2,
                lora_dropout=0, bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
        else:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
            )
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        # Measure model loading VRAM
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            measured_model_gb = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            measured_model_gb = 0

        # Format synthetic data
        from datasets import Dataset
        def format_chat(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        dataset = Dataset.from_list(synthetic_data).map(
            format_chat, remove_columns=list(synthetic_data[0].keys())
        )

        from transformers import TrainingArguments

        # Fix PicklingError: unsloth monkey-patches SFTTrainer but pickle
        # looks up the original class. Re-patch sys.modules.
        import sys as _sys
        try:
            import trl.trainer.sft_trainer as _sft_tm
            import trl.trainer.sft_config as _sft_cm
            _sys.modules["trl.trainer.sft_trainer"].SFTTrainer = _sft_tm.SFTTrainer
            _sys.modules["trl.trainer.sft_config"].SFTConfig = _sft_cm.SFTConfig
        except (ImportError, AttributeError):
            pass
        from trl import SFTTrainer

        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=num_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=8e-5,
            warmup_steps=2,
            logging_steps=1,
            save_steps=999999,  # Don't save during profiling
            fp16=False, bf16=True,
            optim="adamw_8bit",
            seed=3407,
            report_to="none",
            gradient_checkpointing=True,
        )

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=dataset, args=args,
        )

        trainer.train()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            measured_adapters_gb = peak_vram - measured_model_gb
        else:
            peak_vram = 0
            measured_adapters_gb = 0

        step_time = (time.time() - start_time) / max(num_steps, 1)

    except Exception as e:
        success = False
        error = str(e)
        peak_vram = 0
        measured_model_gb = 0
        measured_adapters_gb = 0
        step_time = 0

    # Clean up
    try:
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
    except Exception:
        pass

    return ProfileResult(
        model_name=model_path,
        method=method,
        batch_size=batch_size,
        seq_length=seq_length,
        lora_rank=lora_rank,
        peak_vram_gb=round(peak_vram, 2),
        measured_model_gb=round(measured_model_gb, 2),
        measured_adapters_gb=round(measured_adapters_gb, 2),
        training_steps=num_steps,
        step_time_s=round(step_time, 2),
        success=success,
        error=error,
    )


def profile_all_sizes(
    model_paths: dict[str, str],
    available_vram_gb: float | None = None,
    methods: list[str] | None = None,
) -> list[dict]:
    """Profile multiple model sizes and produce a comparison table.

    Args:
        model_paths: Dict of {size_label: huggingface_path}
            e.g. {"0.5B": "Qwen/Qwen2.5-0.5B", "1.5B": "Qwen/Qwen2.5-1.5B"}
        available_vram_gb: VRAM budget. Auto-detected if None.
        methods: Which methods to test. Default: ["qlora"].

    Returns:
        List of dicts with model size, method, fits, estimated vram, etc.
    """
    if methods is None:
        methods = ["qlora"]

    if available_vram_gb is None:
        gpu = GPUInfo.detect()
        available_vram_gb = gpu.free_vram_gb

    results = []
    for label, path in sorted(model_paths.items()):
        for method in methods:
            est = estimate_vram(
                model_size_b=_parse_size(label),
                method=method,
                batch_size=2,
                seq_length=2048,
                available_vram_gb=available_vram_gb,
            )
            results.append({
                "model": label,
                "path": path,
                "method": method,
                "fits": est.fits,
                "estimated_vram_gb": round(est.total_gb, 2),
                "headroom_gb": round(est.headroom_gb, 2),
                "breakdown": est.to_dict(),
            })

    return results


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
    gpu = GPUInfo.detect()
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
        Path(output_path).write_text(report, encoding="utf-8")

    return report
