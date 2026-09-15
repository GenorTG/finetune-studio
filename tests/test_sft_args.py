"""Regression: local SFT args must not require a Hub token.

Fan-dragon (TRL 0.24 + transformers 5.x) failed POST /api/training/start with:
  KeyError: 'push_to_hub_token'
when SFTTrainer converted plain TrainingArguments via dict_args.pop(...).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import TrainingArguments
from trl import SFTConfig

from finetune_studio.training.engine import TrainingConfig
from finetune_studio.training.sft_args import (
    build_sft_args_from_config,
    build_sft_training_args,
)


def test_build_sft_args_returns_sft_config(tmp_path: Path) -> None:
    """SFTConfig skips TRL's TrainingArguments→dict conversion that KeyErrors."""
    # bf16=False: genorbox1 has no usable CUDA; GPU hosts still pass bf16=True.
    args = build_sft_training_args(
        output_dir=str(tmp_path / "out"),
        num_train_epochs=1,
        bf16=False,
    )
    assert isinstance(args, SFTConfig)
    assert isinstance(args, TrainingArguments)
    # TRL converts only when TrainingArguments and NOT SFTConfig:
    assert not (
        isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig)
    )


def test_build_sft_args_local_defaults_need_no_hub_token(
    tmp_path: Path,
) -> None:
    args = build_sft_training_args(
        output_dir=str(tmp_path / "out"),
        num_train_epochs=1,
        bf16=False,
    )
    assert args.push_to_hub is False
    assert args.hub_token is None


def test_build_sft_args_from_config_maps_training_config(
    tmp_path: Path,
) -> None:
    cfg = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        num_epochs=2,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=5,
        weight_decay=0.02,
        logging_steps=3,
        save_steps=50,
        bf16=False,
        unsloth=False,
        merge_on_save=False,
    )
    args = build_sft_args_from_config(cfg)
    assert isinstance(args, SFTConfig)
    assert args.output_dir == cfg.output_dir
    assert args.num_train_epochs == cfg.num_epochs
    assert args.per_device_train_batch_size == cfg.batch_size
    assert args.gradient_accumulation_steps == cfg.gradient_accumulation_steps
    assert args.push_to_hub is False
    assert args.hub_token is None


def test_plain_training_args_lack_push_to_hub_token(
    tmp_path: Path,
) -> None:
    """Documents TRL 0.24 / transformers 5.x mismatch behind the live failure."""
    plain = TrainingArguments(
        output_dir=str(tmp_path / "plain"),
        report_to="none",
    )
    dict_args = plain.to_dict()
    dict_args["hub_token"] = plain.hub_token
    assert "push_to_hub_token" not in dict_args
    with pytest.raises(KeyError, match="push_to_hub_token"):
        dict_args.pop("push_to_hub_token")
