"""Single source of truth for rendering chat messages to model text.

Training (SFT) and inference/eval must go through this so role tags, system
prompt placement and template kwargs cannot drift apart.
"""

from __future__ import annotations


def with_system_prompt(messages: list, system_prompt: str = "") -> list:
    """Prepend ``system_prompt`` unless empty or the first turn is already a system turn.

    Mirrors the baking rule in ``training.data.format_for_sft``.
    """
    msgs = list(messages)
    if system_prompt and (not msgs or msgs[0].get("role") != "system"):
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return msgs


def render_chat_text(tokenizer, messages: list, *, generation: bool = False,
                     enable_thinking: bool | None = None) -> str:
    """Render ``messages`` with the tokenizer's chat template.

    ``generation=False`` is the training form (full conversation);
    ``generation=True`` appends the assistant generation prompt (inference form).
    ``enable_thinking`` is only forwarded when given and the template accepts it.
    """
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)
