"""Tests for training/preset_advisor.py — tier recommendations by size + data."""
from __future__ import annotations

import json

import pytest

from finetune_studio.training.preset_advisor import (
    dataset_stats,
    guess_base_params_b,
    propose,
)


def test_parse_base_size_from_names() -> None:
    assert guess_base_params_b("Qwen3-4B")[0] == 4.0
    assert guess_base_params_b("Qwen__Qwen3-0.6B")[0] == 0.6
    # The GGUF quant suffix must not be mistaken for a size.
    assert guess_base_params_b("Qwen3.8-27B-abliterated-Q4_K_M.gguf")[0] == 27.0
    assert guess_base_params_b("8.3B-instruct")[0] == 8.3
    assert guess_base_params_b("completely-unknown")[0] is None


def test_precision_reference_dataset_hits_evidence_scale() -> None:
    a = propose(tier="precision", base_model_ref="Qwen3-4B",
                pair_count_hint=515, batch_size=2, gradient_accumulation_steps=4)
    # Evidence anchors: 12 ep from the reference anchor, r128, ~772 optimizer steps.
    assert a.num_epochs == 12
    assert a.lora_rank == 128
    assert a.learning_rate == "2e-4"
    assert 700 <= a.optimizer_steps <= 2500


def test_small_dataset_raises_epochs_to_hit_steps_floor() -> None:
    tiny = propose(tier="precision", base_model_ref="Qwen3-4B",
                   pair_count_hint=60)
    assert tiny.num_epochs > 12  # floor math must push epochs up
    assert any("step" in n.lower() for n in tiny.notes)


def test_big_dataset_does_not_explode_epochs() -> None:
    big = propose(tier="precision", base_model_ref="Qwen3-4B",
                  pair_count_hint=20000)
    assert big.num_epochs <= 12  # sub-linear dataset scaling keeps epochs sane


def test_lr_drops_for_larger_base() -> None:
    a = propose(tier="precision", base_model_ref="Qwen3-27B", pair_count_hint=500)
    assert a.learning_rate == "8e-5"
    small = propose(tier="precision", base_model_ref="Qwen3-4B", pair_count_hint=500)
    assert small.learning_rate == "2e-4"


def test_unknown_base_warns_and_assumes() -> None:
    a = propose(tier="balanced", base_model_ref="mystery-model", pair_count_hint=500)
    assert a.size_source == "assumed"
    assert a.base_params_b == 4.0
    assert a.warnings  # user gets told about the assumption


def test_smoke_tier_has_no_step_floor_warning() -> None:
    a = propose(tier="smoke", base_model_ref="Qwen3-4B", pair_count_hint=515)
    assert not [w for w in a.warnings if "optimizer steps" in w]


def test_dataset_stats_reads_sharegpt_lines(tmp_path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"messages": [{"role": "user", "content": "hello"},
                                     {"role": "assistant", "content": "world"}]}),
            json.dumps({"messages": [{"role": "user", "content": "x" * 30}]}),
        ]),
        encoding="utf-8",
    )
    n, avg = dataset_stats(str(p))
    assert n == 2
    assert avg > 10


def test_missing_dataset_falls_back_to_reference(tmp_path) -> None:
    a = propose(tier="balanced", base_model_ref="Qwen3-4B",
                dataset_path=str(tmp_path / "nope.jsonl"))
    assert a.pair_count == 500
    assert any("unreadable" in w or "500" in w for w in a.warnings)


def test_invalid_tier_raises() -> None:
    with pytest.raises(ValueError):
        propose(tier="bogus", base_model_ref="Qwen3-4B")
