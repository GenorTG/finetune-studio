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

def format_for_sft(data: list, system_prompt: str = "") -> list:
    formatted = []
    for item in data:
        if "messages" in item:
            msgs = list(item["messages"])
            if system_prompt and (not msgs or msgs[0].get("role") != "system"):
                msgs = [{"role": "system", "content": system_prompt}] + msgs
            formatted.append({"messages": msgs})
        elif "text" in item:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": item["text"]})
            formatted.append({"messages": msgs})
    return formatted

def split_data(data: list, train_ratio: float = 0.9):
    import random
    shuffled = data.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]
