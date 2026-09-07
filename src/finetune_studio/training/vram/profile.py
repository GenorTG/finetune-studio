"""Actual VRAM profiling — run a tiny training job and measure real peak memory.

Single responsibility: take a model + config, run a few steps, report real numbers.
"""
from __future__ import annotations

import logging
import os
import shutil
import time

from finetune_studio.training.vram.constants import MODEL_PRESETS
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
