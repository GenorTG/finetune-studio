"""Tokenisation guards for preference training (DPO / KTO).

TRL tokenises ``prompt`` with ``add_generation_prompt=True`` and the full
``prompt + completion`` conversation separately, then slices the completion as
``full_ids[len(prompt_ids):]``. That only isolates the completion when
``prompt_ids`` is a token prefix of ``full_ids``.

Thinking-by-default templates (Qwen3.5) break that: the generation prompt ends in
an *open* ``<think>\\n`` while a finished assistant turn renders
``<think>\\n\\n</think>\\n\\n``. BPE merges the trailing ``\\n`` of the prompt with
the next ``\\n`` into one ``\\n\\n`` token, so the prompt's last token is not in the
full sequence, TRL logs "Mismatch between tokenized prompt and the start of
tokenized prompt+chosen" for **every row**, drops one token from each completion
and feeds the model ``<think>\\n</think>`` instead of the rendered ``<think>\\n\\n</think>``.
Rendering the prompt with ``enable_thinking=False`` makes the generation prompt
end in the closed empty think block, which *is* a prefix of the rendered turn.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

log = logging.getLogger(__name__)


def preference_template_kwargs(tokenizer: Any) -> dict[str, Any]:
    """``chat_template_kwargs`` that keep prompt tokens a prefix of prompt+completion tokens.

    Only templates that declare ``enable_thinking`` need it (Qwen3 / Qwen3.5); every
    other template renders identically with no kwargs.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(template, Mapping):  # named templates (default / tool_use)
        template = " ".join(str(t) for t in template.values())
    return {"enable_thinking": False} if "enable_thinking" in str(template) else {}


def _ids(tokenizer: Any, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[int]:
    out = tokenizer.apply_chat_template(
        list(messages), tokenize=True, return_dict=True, **kwargs,
    )
    return list(out["input_ids"])


def prefix_mismatches(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    template_kwargs: Mapping[str, Any] | None = None,
    *,
    completion_keys: Sequence[str] = ("chosen", "rejected"),
) -> list[tuple[int, str]]:
    """``(row_index, key)`` pairs whose prompt tokens are not a prefix of prompt+completion tokens.

    ``rows`` are conversational (``prompt`` message list + one list per key in
    ``completion_keys``). Empty result = TRL's chosen/rejected split is exact.
    """
    kwargs = dict(template_kwargs or {})
    bad: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        prompt_ids = _ids(tokenizer, row["prompt"], add_generation_prompt=True, **kwargs)
        for key in completion_keys:
            full = _ids(tokenizer, [*row["prompt"], *row[key]], **kwargs)
            if full[: len(prompt_ids)] != prompt_ids:
                bad.append((index, key))
    return bad


def describe_mismatch(bad: Sequence[tuple[int, str]], total_rows: int) -> str:
    """One honest line for the run log when rows still mis-split after the template fix."""
    rows = len({index for index, _ in bad})
    return (
        f"Tokenizer check: {rows} of {total_rows} rows have prompt tokens that are not a prefix "
        "of prompt+answer tokens; TRL will cut those answers one or more tokens late. "
        "Check this model's chat template."
    )
