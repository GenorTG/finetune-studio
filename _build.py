#!/usr/bin/env python3
"""Build script to create all source files for Finetune Studio."""
import os, sys

BASE = "/home/genorbox1/.openclaw/workspace/finetune-studio/src/finetune_studio"
count = 0

def w(rel_path, content):
    global count
    full = os.path.join(BASE, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    count += 1
    print(f"  wrote {rel_path}")

print("Building Finetune Studio source files...")

# training/data.py
w("training/data.py", """\"\"\"Training data - dataset loading, chatml formatting, tokenization.\"\"\"

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".jsonl", ".json", ".txt", ".csv"}


def load_dataset(file_path: str) -> list[dict]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_dataset(file_path)
    elif suffix == ".json":
        return load_json_dataset(file_path)
    elif suffix == ".txt":
        return load_text_dataset(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_jsonl_dataset(file_path: str) -> list[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {i}: {e}")
    return data


def load_json_dataset(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def load_text_dataset(file_path: str, separator: str = "\\n\\n") -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = [c.strip() for c in content.split(separator) if c.strip()]
    return [{"messages": [{"role": "user", "content": c}]} for c in chunks]


def format_chatml(messages: list[dict]) -> str:
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\\n{content}