"""Convert various formats to JSONL training data.

WHAT THIS FILE DOES
==================
Converts CSV, plain text, or JSON files into the JSONL format
expected by the training pipeline. Each JSON line is a training
example with a "messages" field.

KEY CONCEPTS
============
- JSONL (JSON Lines): each line is a separate JSON object. Unlike
  regular JSON, it's not wrapped in [...]. Easy to stream and process.
- Training example format: {"messages": [{"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}]}
- CSV column mapping: if your CSV has columns like "question" and
  "answer", we map them to user/assistant messages.
"""

import csv
import json


def jsonl_to_json(jsonl_path, json_path):
    data = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def json_to_jsonl(json_path, jsonl_path):
    with open(json_path) as f:
        data = json.load(f)
    with open(jsonl_path, "w") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in data)

def csv_to_jsonl(csv_path, jsonl_path, text_column="text", system_prompt=""):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        with open(jsonl_path, "w") as out:
            for row in reader:
                text = row.get(text_column, "")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": text})
                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

def simple_to_chat(text_path, jsonl_path, system_prompt=""):
    with open(text_path) as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    with open(jsonl_path, "w") as out:
        for block in blocks:
            lines = block.strip().split("\n")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for line in lines:
                if line.startswith(("Q:", "q:")):
                    messages.append({"role": "user", "content": line[2:].strip()})
                elif line.startswith(("A:", "a:")):
                    messages.append({"role": "assistant", "content": line[2:].strip()})
            if len(messages) > 1:
                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
