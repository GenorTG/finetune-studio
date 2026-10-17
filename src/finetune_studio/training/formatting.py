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
                     enable_thinking: bool | None = None,
                     tools: list | None = None) -> str:
    """Render ``messages`` with the tokenizer's chat template.

    ``generation=False`` is the training form (full conversation);
    ``generation=True`` appends the assistant generation prompt (inference form).
    ``enable_thinking`` is only forwarded when given and the template accepts it.
    """
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    if tools:
        kwargs["tools"] = tools
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


# End-of-turn markers of the common chat templates (ChatML, Llama 3, Gemma, Phi).
_CHAT_END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>")


def chat_stop_token_ids(tokenizer, generation_config=None) -> list[int]:
    """Token ids that end an assistant turn, for ``generate(eos_token_id=...)`` and saved generation configs.

    A base without ``generation_config.json`` (the local Qwen3.5-9B) derives its EOS from ``config.json``
    (``<|endoftext|>``) while chat turns end with ``<|im_end|>``: transformers then generated up to
    ``max_new_tokens`` on every answer (28 s for a one-line reply) and the extra text was cut afterwards.
    """
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)   # a multimodal processor wraps the tokenizer
    ids: list[int] = []

    def add(value) -> None:
        for v in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(v, int) and v >= 0 and v not in ids:
                ids.append(v)

    add(getattr(generation_config, "eos_token_id", None))
    add(getattr(tokenizer, "eos_token_id", None))
    unk = getattr(tokenizer, "unk_token_id", None)
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for token in _CHAT_END_TOKENS:
        tid = vocab.get(token)
        if tid is not None and tid != unk:
            add(tid)
    return ids


def _end_at_eos(text: str, tokenizer) -> str:
    """Drop the newline a ChatML template writes after the final end token: TRL appends EOS to any text or completion
    that does not end with it, which made every row end in ``<|im_end|>\\n<|im_end|>``."""
    eos = getattr(getattr(tokenizer, "tokenizer", tokenizer), "eos_token", None)
    stripped = text.rstrip("\n")
    return stripped if isinstance(eos, str) and eos and stripped.endswith(eos) else text


def split_prompt_completion(tokenizer, messages: list, *, tools: list | None = None) -> tuple[str, str] | None:
    """``(prompt, completion)`` for completion-only SFT loss, or None when the template cannot be split cleanly.

    The prompt is the conversation up to the last assistant turn rendered exactly as inference renders it (generation
    prompt, Qwen's empty think block with thinking off); the completion is that turn. Training on the full text spent
    the loss on the baked system prompt, the template and the question: a run reported 0.046 while its answer tokens
    were still at 0.65 and it recalled 16/56 of its own training answers.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    full = render_chat_text(tokenizer, messages, tools=tools)
    for think in (False, None):
        prompt = render_chat_text(tokenizer, messages[:-1], generation=True, enable_thinking=think, tools=tools)
        if full.startswith(prompt) and len(full) > len(prompt):
            completion = _end_at_eos(full[len(prompt):], tokenizer)
            if completion.strip():
                return prompt, completion
    return None
