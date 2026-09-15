"""Unsloth integration — optimized LoRA training.

Provides a drop-in replacement for the standard training loop that uses
Unsloth for 2-5x faster training and lower VRAM usage.
"""

from __future__ import annotations

import os
import sys
import time


def is_unsloth_available() -> bool:
    """Check if unsloth is installed and usable."""
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def train_with_unsloth(
    model_path: str,
    output_dir: str,
    train_data: list[dict],
    config,
    state,
    system_prompt: str = "",
) -> dict:
    """Train using Unsloth's optimized FastLanguageModel.

    Args:
        model_path: Path to the base model
        output_dir: Where to save the adapter
        train_data: List of {"messages": [...]} dicts
        config: TrainingConfig instance
        state: TrainingState instance (for progress updates)
        system_prompt: Optional system prompt

    Returns:
        {adapter_dir, size_bytes, size_human}
    """
    from datasets import Dataset
    from unsloth import FastLanguageModel

    from finetune_studio.training.sft_args import build_sft_training_args

    state.message = "Loading model with Unsloth (4-bit quantized)..."
    _notify_state(state)

    # Load model in 4-bit — much less VRAM, faster training
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Fix PicklingError: unsloth monkey-patches SFTTrainer/SFTConfig
    import trl.trainer.sft_trainer as _sft_trainer_mod
    import trl.trainer.sft_config as _sft_config_mod
    sys.modules["trl.trainer.sft_trainer"].SFTTrainer = _sft_trainer_mod.SFTTrainer
    sys.modules["trl.trainer.sft_config"].SFTConfig = _sft_config_mod.SFTConfig

    from trl import SFTTrainer

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(train_data).map(
        format_chat, remove_columns=list(train_data[0].keys())
    )

    steps_per_epoch = max(len(dataset) // (config.batch_size * config.gradient_accumulation_steps), 1)
    total_steps = steps_per_epoch * config.num_epochs
    state.total_steps = total_steps

    # SFTConfig (not TrainingArguments): avoids TRL KeyError push_to_hub_token.
    args = build_sft_training_args(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
    )

    start_time = time.time()

    class ProgressCallback:
        def on_log(self2, args, state_cb, control, logs=None, **kwargs):
            if logs:
                state.current_step = state_cb.global_step
                state.loss = round(logs.get("loss", 0), 4)
                state.learning_rate = round(logs.get("learning_rate", 0), 8)
                state.epoch = round(state_cb.epoch or 0, 2)
                state.elapsed = round(time.time() - start_time, 1)
                if state_cb.global_step > 0:
                    rate = state.elapsed / state_cb.global_step
                    state.eta = round(rate * (total_steps - state_cb.global_step), 1)
                _notify_state(state)

    # Monkey-patch Trainer._save to avoid PicklingError
    import json as _json
    def _patched_save(self_trainer, output_dir, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self_trainer.model, 'save_pretrained'):
            self_trainer.model.save_pretrained(output_dir)
        if hasattr(self_trainer, 'tokenizer') and self_trainer.tokenizer is not None:
            self_trainer.tokenizer.save_pretrained(output_dir)
        args_path = os.path.join(output_dir, "training_args.json")
        with open(args_path, "w") as f:
            _json.dump(self_trainer.args.to_dict(), f, indent=2, default=str)
    SFTTrainer._save = _patched_save

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=args,
        callbacks=[ProgressCallback()],
    )

    state.status = "training"
    state.message = "Training with Unsloth..."
    _notify_state(state)

    trainer.train()

    # Save adapter
    state.status = "saving"
    state.message = "Saving adapter..."
    _notify_state(state)

    os.makedirs(output_dir, exist_ok=True)
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            with open(os.path.join(adapter_dir, "chat_template.jinja"), "w") as f:
                f.write(tokenizer.chat_template)
        except Exception:
            pass

    size = _dir_size(adapter_dir)
    return {
        "adapter_dir": adapter_dir,
        "size_bytes": size,
        "size_human": _human_size(size),
    }


def _notify_state(state):
    """Call all registered callbacks on the state object."""
    for cb in getattr(state, '_callbacks', []):
        try:
            cb(state)
        except Exception:
            pass


def _dir_size(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"
