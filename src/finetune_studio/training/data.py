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
import re

_PROVENANCE_LINE = re.compile(
    r"(?im)^\s*(?:source|filename|file)\s*:\s*[^\n]+$"
)
_PROVENANCE_PAREN = re.compile(
    r"(?i)\s*[\(\[]\s*(?:source|filename|file)\s*:\s*[^\)\]]+\s*[\)\]]\s*$"
)


def clean_answer_for_training(answer: str) -> str:
    """Remove display-only provenance from the learned assistant response.

    Source IDs remain in the exported JSONL row metadata. Teaching the model
    filenames makes answers noisy and can turn digits in filenames into false
    numeric answers; provenance belongs in the audit trail, not the target.
    """
    text = _PROVENANCE_LINE.sub("", answer or "")
    text = _PROVENANCE_PAREN.sub("", text)
    return text.strip()


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
        for msg in msgs:
            if msg.get("role") == "assistant":
                msg["content"] = clean_answer_for_training(str(msg.get("content", "")))
        if system_prompt and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        row = {"messages": msgs}
        if "tools" in item:
            row["tools"] = item["tools"]
        formatted.append(row)
    return formatted


def format_for_continued_pretraining(data: list) -> list[dict[str, str]]:
    """Normalize raw-domain language-modeling rows without chat-role wrapping."""
    formatted: list[dict[str, str]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Row {index}: expected a JSON object")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Row {index}: continued pretraining requires non-empty 'text'")
        formatted.append({"text": text.strip()})
    if not formatted:
        raise ValueError("Continued-pretraining dataset is empty")
    return formatted


def format_for_preference(data: list, system_prompt: str = "") -> list[dict[str, object]]:
    """Normalize DPO examples to TRL's conversational preference format.

    Each result has a shared ``prompt`` and assistant ``chosen``/``rejected``
    completions. Standard string rows are converted to chat turns so an
    optional system prompt is represented without ambiguous string joining.
    """
    formatted: list[dict[str, object]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Row {index}: expected a JSON object")
        prompt = item.get("prompt")
        chosen = item.get("chosen")
        rejected = item.get("rejected")
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        if isinstance(chosen, str):
            chosen = [{"role": "assistant", "content": chosen}]
        if isinstance(rejected, str):
            rejected = [{"role": "assistant", "content": rejected}]
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Row {index}: prompt must be non-empty text or message list")
        if not isinstance(chosen, list) or not chosen:
            raise ValueError(f"Row {index}: chosen must be non-empty text or message list")
        if not isinstance(rejected, list) or not rejected:
            raise ValueError(f"Row {index}: rejected must be non-empty text or message list")

        for field, messages in (("prompt", prompt), ("chosen", chosen), ("rejected", rejected)):
            for message_index, message in enumerate(messages, start=1):
                if (not isinstance(message, dict)
                        or not isinstance(message.get("role"), str)
                        or not isinstance(message.get("content"), str)
                        or not message["content"].strip()):
                    raise ValueError(
                        f"Row {index}: {field} message {message_index} needs role and non-empty content"
                    )
        prompt_messages = [dict(message) for message in prompt]
        chosen_messages = [dict(message) for message in chosen]
        rejected_messages = [dict(message) for message in rejected]
        if prompt_messages[-1]["role"] != "user":
            raise ValueError(f"Row {index}: prompt must end with a user message")
        if chosen_messages[-1]["role"] != "assistant" or rejected_messages[-1]["role"] != "assistant":
            raise ValueError(f"Row {index}: chosen and rejected must end with an assistant response")
        if chosen_messages == rejected_messages:
            raise ValueError(f"Row {index}: chosen and rejected responses must differ")
        if system_prompt and (not prompt_messages or prompt_messages[0]["role"] != "system"):
            prompt_messages.insert(0, {"role": "system", "content": system_prompt})
        formatted.append({
            "prompt": prompt_messages,
            "chosen": chosen_messages,
            "rejected": rejected_messages,
        })
    if not formatted:
        raise ValueError("Preference dataset is empty")
    return formatted

def split_data(data: list, train_ratio: float = 0.9, seed: int = 42):
    import random
    shuffled = data.copy()
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    split = int(len(shuffled) * train_ratio)
    split = min(max(1, split), len(shuffled) - 1)
    return shuffled[:split], shuffled[split:]
