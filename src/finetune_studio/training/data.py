"""Format and tokenize training data.

WHAT THIS FILE DOES
==================
Prepares raw training data for the SFT loop:
  1. Format each example as a chat conversation
  2. Tokenize the text
  3. Add labels (which tokens to predict during training)
  4. Batch examples together
  5. Add padding so all examples in a batch are the same length

KEY CONCEPTS
============
- Tokenization: converting text to integers (token IDs).
- Labels: the target output for each input token. For training,
  we want the model to predict everything except the system prompt
  and user input — only the assistant's response is the "label".
- Masking: setting certain token labels to -100 (ignored by loss)
  so we don't train on system prompts or user inputs.
- Padding: making all examples in a batch the same length. We pad
  with the PAD token, and use attention masks to ignore the padding.
"""

import json


def load_jsonl(path: str) -> list:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: list, path: str):
    with open(path, "w") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in data)

def validate_messages(data: list) -> list:
    errors = []
    for i, item in enumerate(data):
        if "messages" in item:
            msgs = item["messages"]
            if not isinstance(msgs, list):
                errors.append(f"Row {i}: messages must be a list")
                continue
            for j, msg in enumerate(msgs):
                if "role" not in msg:
                    errors.append(f"Row {i}, msg {j}: missing role")
                if "content" not in msg:
                    errors.append(f"Row {i}, msg {j}: missing content")
        elif "text" not in item:
            errors.append(f"Row {i}: no messages or text key found")
    return errors

_SHAREGPT_FROM_TO_ROLE = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "observation": "tool",
}


def _conversations_to_messages(conversations: list) -> list:
    """Convert sharegpt `conversations: [{from, value}]` to OpenAI
    `messages: [{role, content}]`. Unknown `from` values raise so
    training fails loudly instead of silently dropping the row."""
    out = []
    for turn in conversations:
        src = (turn.get("from") or "").strip().lower()
        dst = _SHAREGPT_FROM_TO_ROLE.get(src)
        if dst is None:
            raise ValueError(f"unknown sharegpt 'from' role: {turn.get('from')!r}")
        out.append({"role": dst, "content": turn.get("value", "")})
    return out


def format_for_sft(data: list, system_prompt: str = "") -> list:
    formatted = []
    for item in data:
        # Prefer 'conversations' (sharegpt) when BOTH are present — that's
        # what the data-prep export endpoint writes, and a row with both
        # is almost always an export artefact, not user intent.
        if "conversations" in item and isinstance(item["conversations"], list):
            try:
                msgs = _conversations_to_messages(item["conversations"])
            except ValueError:
                continue
        elif "messages" in item:
            msgs = list(item["messages"])
        elif "text" in item:
            msgs = [{"role": "user", "content": item["text"]}]
        elif "prompt" in item and "completion" in item:
            msgs = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]},
            ]
        else:
            continue
        if not msgs:
            continue
        if system_prompt and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        formatted.append({"messages": msgs})
    return formatted

def split_data(data: list, train_ratio: float = 0.9):
    import random
    shuffled = data.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]
