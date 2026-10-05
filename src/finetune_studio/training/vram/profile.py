"""Actual VRAM profiling — run a tiny training job and measure real peak memory.

Single responsibility: take a model + config, run a few steps, report real numbers.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

from finetune_studio import accel
from finetune_studio.training.vram.estimate import estimate_vram
from finetune_studio.training.vram.gpu import detect as detect_gpu
from finetune_studio.training.vram.report import _parse_size
from finetune_studio.training.vram.schema import ProfileResult

log = logging.getLogger(__name__)


def profile_training(
    model_path: str,
    method: str = "qlora",
    batch_size: int = 1,
    seq_length: int = 512,
    lora_rank: int = 16,
    num_steps: int = 5,
    output_dir: str | None = None,
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
        output_dir: Parent directory for the profiler's scratch checkpoint dir
            (default ``~/.cache/fts-vram-profile``). The profiler creates its own
            fresh ``mkdtemp`` child inside it and deletes only that child; the
            directory you pass in (and anything in it) is never removed.

    Returns:
        ProfileResult with actual measurements.
    """
    # Create synthetic training data
    synthetic_data = [
        {"messages": [{"role": "user", "content": f"What is {i}?"}, {"role": "assistant", "content": f"The answer to {i} is {i * 2}."}]}
        for i in range(max(batch_size * 2, 10))
    ]

    start_time = time.time()
    success = True
    error = ""
    peak_vram = 0.0
    measured_model_gb = 0.0
    measured_adapters_gb = 0.0
    step_time = 0.0
    scratch_dir: Path | None = None

    try:
        # Own a private scratch dir; never hand a caller-supplied path to rmtree.
        scratch_base = Path(output_dir) if output_dir else Path.home() / ".cache" / "fts-vram-profile"
        scratch_base.mkdir(parents=True, exist_ok=True)
        scratch_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(scratch_base)))

        accel.reset_peak_memory()

        from peft import LoraConfig, get_peft_model

        from finetune_studio.hf_env import load_tokenizer
        from finetune_studio.models.hf_loader import load_causal_lm
        tokenizer = load_tokenizer(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # E2E-40: never import unsloth in this process — bitsandbytes (where the
        # backend supports it) + PEFT; plain 16-bit/fp32 otherwise.
        model = load_causal_lm(model_path, force_4bit=(method == "qlora"))
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Measure model loading VRAM
        accel.synchronize()
        measured_model_gb = accel.peak_memory_gb()

        # Format synthetic data
        from datasets import Dataset
        from finetune_studio.training.formatting import render_chat_text
        def format_chat(example):
            text = render_chat_text(tokenizer, example["messages"])
            return {"text": text}

        dataset = Dataset.from_list(synthetic_data).map(
            format_chat, remove_columns=list(synthetic_data[0].keys())
        )

        # Fix PicklingError: unsloth monkey-patches SFTTrainer but pickle
        # looks up the original class. Re-patch sys.modules.
        import sys as _sys
        try:
            import trl.trainer.sft_config as _sft_cm
            import trl.trainer.sft_trainer as _sft_tm
            _sys.modules["trl.trainer.sft_trainer"].SFTTrainer = _sft_tm.SFTTrainer
            _sys.modules["trl.trainer.sft_config"].SFTConfig = _sft_cm.SFTConfig
        except (ImportError, AttributeError):
            pass
        from trl import SFTTrainer

        from finetune_studio.training.accel_plan import resolve_train_plan
        from finetune_studio.training.sft_args import build_sft_training_args

        plan = resolve_train_plan(want_unsloth=False, want_8bit_optim=True)

        # SFTConfig (not TrainingArguments): avoids TRL KeyError push_to_hub_token.
        args = build_sft_training_args(
            output_dir=str(scratch_dir),
            max_steps=num_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=8e-5,
            warmup_steps=2,
            logging_steps=1,
            save_steps=999999,  # Don't save during profiling
            bf16=plan.bf16,
            optim=plan.optim,
            seed=3407,
            gradient_checkpointing=True,
            fp16=plan.fp16,
            use_cpu=plan.use_cpu,
        )

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=dataset, args=args,
        )

        trainer.train()

        accel.synchronize()
        peak_vram = accel.peak_memory_gb()
        measured_adapters_gb = max(peak_vram - measured_model_gb, 0.0)

        step_time = (time.time() - start_time) / max(num_steps, 1)

    except Exception as e:  # noqa: BLE001 — profiling must always return a structured result
        success = False
        error = str(e)
        peak_vram = 0
        measured_model_gb = 0
        measured_adapters_gb = 0
        step_time = 0

    # Clean up only the scratch dir this call created.
    if scratch_dir is not None:
        shutil.rmtree(scratch_dir, ignore_errors=True)

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
        gpu = detect_gpu()
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
