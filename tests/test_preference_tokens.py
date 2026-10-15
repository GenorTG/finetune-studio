"""Prompt-prefix guard for preference training on thinking-by-default chat templates."""
from __future__ import annotations

import pytest

from finetune_studio.training.preference_tokens import (
    describe_mismatch,
    preference_template_kwargs,
    prefix_mismatches,
)

# Condensed Qwen3.5 template: an open ``<think>\n`` generation prompt unless
# ``enable_thinking`` is false, and ``<think>\n\n</think>\n\n`` on a finished turn.
THINKING_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'assistant' %}"
    "{{ '<|im_start|>assistant\n<think>\n\n</think>\n\n' + m.content + '<|im_end|>\n' }}"
    "{% else %}{{ '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}"
    "{% else %}{{ '<think>\n' }}{% endif %}{% endif %}"
)
PLAIN_TEMPLATE = (
    "{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
ROWS = [
    {
        "prompt": [{"role": "user", "content": "What is the capital of Tidewarden?"}],
        "chosen": [{"role": "assistant", "content": "Vael Harbor."}],
        "rejected": [{"role": "assistant", "content": "Paris."}],
    },
    {
        "prompt": [
            {"role": "system", "content": "Answer from the documents."},
            {"role": "user", "content": "Who founded the guild?"},
        ],
        "chosen": [{"role": "assistant", "content": "Maren Voss."}],
        "rejected": [{"role": "assistant", "content": "Nobody knows."}],
    },
]


def _tokenizer(template: str):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    # Qwen's own split regex: it keeps ``\n\n`` in one pre-token, unlike GPT-2's.
    qwen_split = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(qwen_split, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    corpus = ["\n\n</think>\n\n<think>\n\n\n"] * 200 + ["Vael Harbor Maren Voss guild Paris capital"] * 20
    special = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.chat_template = template
    return fast


def test_thinking_template_breaks_the_prefix_without_the_fix() -> None:
    tok = _tokenizer(THINKING_TEMPLATE)
    bad = prefix_mismatches(tok, ROWS)
    assert {index for index, _ in bad} == {0, 1}
    assert {key for _, key in bad} == {"chosen", "rejected"}


def test_enable_thinking_false_restores_the_prefix() -> None:
    tok = _tokenizer(THINKING_TEMPLATE)
    kwargs = preference_template_kwargs(tok)
    assert kwargs == {"enable_thinking": False}
    assert prefix_mismatches(tok, ROWS, kwargs) == []


def test_plain_template_needs_no_kwargs_and_has_no_mismatch() -> None:
    tok = _tokenizer(PLAIN_TEMPLATE)
    assert preference_template_kwargs(tok) == {}
    assert prefix_mismatches(tok, ROWS) == []


@pytest.mark.parametrize("template", [None, "", {"default": "no flag here"}])
def test_missing_or_flagless_templates_get_no_kwargs(template) -> None:
    class Tok:
        chat_template = template

    assert preference_template_kwargs(Tok()) == {}


def test_describe_mismatch_counts_rows_not_pairs() -> None:
    line = describe_mismatch([(0, "chosen"), (0, "rejected"), (3, "chosen")], 10)
    assert "2 of 10 rows" in line
