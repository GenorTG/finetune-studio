"""Use-case templates (plan §7.6): four templates with data needs, a metric, and model-agnostic training settings."""
from __future__ import annotations

import json

import pytest

from finetune_studio.usecases import templates as t


def _rows(n: int, chars: int = 40) -> list[list[dict[str, str]]]:
    return [[{"role": "system", "content": "Do it."}, {"role": "user", "content": "x" * chars},
             {"role": "assistant", "content": "y" * chars}] for _ in range(n)]


def test_the_four_templates_each_have_data_needs_a_metric_and_a_scoring_route() -> None:
    assert set(t.TEMPLATES) == {"extraction", "classification", "style", "distillation"}
    scoring = {tid: tpl.scoring for tid, tpl in t.TEMPLATES.items()}
    assert scoring == {"extraction": "exact", "classification": "exact", "style": "ab", "distillation": "judge"}
    for tpl in t.TEMPLATES.values():
        d = tpl.as_dict()
        assert tpl.needs.minimum >= 50 and tpl.needs.recommended[0] <= tpl.needs.recommended[1] <= 1000
        assert d["minimum_rows"] == tpl.needs.minimum + t.MIN_HELDOUT
        assert tpl.metric and tpl.needs.evidence and tpl.row_shape and tpl.example_row
    assert t.TEMPLATES["classification"].needs.per_label_minimum == t.MIN_PER_LABEL


def test_nothing_in_a_template_names_a_model_family() -> None:
    """Any safetensors chat model: no Qwen / Llama / Gemma special case anywhere in the specs or the settings."""
    text = json.dumps([tpl.as_dict() for tpl in t.TEMPLATES.values()]).lower()
    text += json.dumps(t.recommend_settings(t.TEMPLATES["style"], _rows(100)).as_dict()).lower()
    for family in ("qwen", "llama", "gemma", "mistral", "phi-", "deepseek"):
        assert family not in text


def test_unknown_template_names_the_valid_ids() -> None:
    with pytest.raises(KeyError, match="extraction, classification, style, distillation"):
        t.get_template("summary")


@pytest.mark.parametrize(("n", "held"), [(50, 20), (100, 20), (300, 60), (5000, 200)])
def test_heldout_is_a_fifth_clamped_to_20_and_200(n: int, held: int) -> None:
    assert t.heldout_size(n) == held


@pytest.mark.parametrize(("n", "epochs"), [(40, 5), (160, 3), (240, 2), (2000, 2)])
def test_epochs_give_a_small_set_enough_steps_within_bounds(n: int, epochs: int) -> None:
    plan = t.recommend_settings(t.TEMPLATES["extraction"], _rows(n))
    assert plan.overrides["num_epochs"] == epochs
    assert plan.overrides["batch_size"] * plan.overrides["gradient_accumulation_steps"] == 8
    assert plan.overrides["learning_rate"] == 1e-4 and plan.overrides["lora_rank"] == 64
    assert plan.training_mode == "sft" and plan.system_prompt_mode == "none" and plan.eval_temperature == 0.0


def test_sequence_length_and_answer_budget_follow_the_longest_example() -> None:
    short = t.recommend_settings(t.TEMPLATES["distillation"], _rows(100, 40))
    long = t.recommend_settings(t.TEMPLATES["distillation"], _rows(100, 6000))
    assert short.overrides["max_seq_length"] == 512 and long.overrides["max_seq_length"] == 4096
    assert short.eval_max_tokens == 256 and long.eval_max_tokens == 2048
    assert t.recommend_settings(t.TEMPLATES["classification"], _rows(100, 6000)).eval_max_tokens == 32
