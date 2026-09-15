"""Build TRL SFT training args for local runs (no Hub push required).

TRL 0.24 ``SFTTrainer`` converts a plain ``TrainingArguments`` by doing
``dict_args.pop("push_to_hub_token")``. Transformers 5.x ``to_dict()`` only
emits ``hub_token``, so that pop raises ``KeyError`` and training fails in ~8s
before any steps run. Passing ``SFTConfig`` skips the conversion path entirely.
"""

from __future__ import annotations

from typing import Any


def build_sft_training_args(
    *,
    output_dir: str,
    num_train_epochs: float | None = None,
    max_steps: int | None = None,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 8e-5,
    warmup_steps: int = 20,
    weight_decay: float = 0.005,
    logging_steps: int = 10,
    save_steps: int = 100,
    bf16: bool = True,
    optim: str = "adamw_torch",
    seed: int = 3407,
    gradient_checkpointing: bool = False,
    **extra: Any,
) -> Any:
    """Return an ``SFTConfig`` with Hub push disabled (local training default).

    ``hub_token`` / ``push_to_hub`` stay unset/false so a Hugging Face token is
    never required for standard local fine-tunes.
    """
    from trl import SFTConfig

    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "fp16": not bf16,
        "bf16": bf16,
        "optim": optim,
        "seed": seed,
        "report_to": "none",
        "push_to_hub": False,
        "hub_token": None,
        "gradient_checkpointing": gradient_checkpointing,
    }
    if num_train_epochs is not None:
        kwargs["num_train_epochs"] = num_train_epochs
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    kwargs.update(extra)
    return SFTConfig(**kwargs)


def build_sft_args_from_config(cfg: Any) -> Any:
    """Map a ``TrainingConfig`` onto local ``SFTConfig`` defaults."""
    return build_sft_training_args(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        bf16=cfg.bf16,
    )
