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
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def json_to_jsonl(json_path, jsonl_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in data)

def csv_to_jsonl(csv_path, jsonl_path, text_column="text", system_prompt=""):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        with open(jsonl_path, "w", encoding="utf-8") as out:
            for row in reader:
                text = row.get(text_column, "")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": text})
                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
