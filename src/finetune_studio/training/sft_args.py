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
    never required for standard local fine-tunes. The config is pinned to the
    accelerator's device (``accel.pin_trainer_args``), so every caller trains on
    the chosen card even when it is not index 0.
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
    from finetune_studio import accel
    # transformers pins the Trainer to cuda:0 / n_gpu=device_count; follow the accelerator instead.
    return accel.pin_trainer_args(SFTConfig(**kwargs))


def checkpoint_eval_kwargs(cfg: Any, *, has_eval: bool) -> dict[str, Any]:
    """Checkpoint / eval / early-stopping ``SFTConfig`` kwargs for ``cfg``.

    Eval runs only when ``eval_steps > 0`` and there is held-out data. Early
    stopping additionally keeps the best checkpoint, which HF requires to save
    on the eval schedule, so it forces saving even if periodic checkpoints are
    off. Without eval data it degrades to a plain run.
    """
    eval_on = has_eval and cfg.eval_steps > 0
    early = cfg.early_stopping and eval_on
    kw: dict[str, Any] = {"eval_strategy": "no"}
    if eval_on:
        kw.update(eval_strategy="steps", eval_steps=cfg.eval_steps,
                  per_device_eval_batch_size=cfg.batch_size)
    if not (cfg.save_checkpoints or early):
        kw["save_strategy"] = "no"
        return kw
    kw.update(save_strategy="steps", save_total_limit=cfg.save_total_limit)
    if early:
        kw.update(save_steps=cfg.eval_steps, load_best_model_at_end=True,
                  metric_for_best_model="eval_loss", greater_is_better=False)
    return kw


def build_sft_args_from_config(cfg: Any, *, has_eval: bool = False) -> Any:
    """Map a ``TrainingConfig`` onto local ``SFTConfig`` defaults."""
    extra = checkpoint_eval_kwargs(cfg, has_eval=has_eval)
    return build_sft_training_args(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=extra.pop("save_steps", cfg.save_steps),
        bf16=cfg.bf16,
        **extra,
    )
