"""Validate JSONL training data format.

WHAT THIS FILE DOES
==================
Checks training data for common errors before training:
  - Valid JSON syntax (each JSONL row must be an object)
  - Required fields (messages, role, content)
  - Rows with neither "messages" nor "text" produce a warning

KEY CONCEPTS
============
- Fail-fast: catch errors BEFORE training, not after 4 hours of training.
- Detailed error messages: not just "invalid" but "line 42: empty
  content in user message".
- Validation report: a summary of issues found, with line numbers.
"""

import json
from pathlib import Path


def validate_file(path):
    p = Path(path)
    report = {"path": str(p), "name": p.name, "valid": True, "errors": [], "warnings": [], "stats": {}}
    if not p.exists():
        report["valid"] = False
        report["errors"].append("File not found")
        return report
    try:
        if p.suffix == ".jsonl":
            return validate_jsonl(p, report)
        elif p.suffix == ".json":
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            report["stats"] = {"rows": len(data) if isinstance(data, list) else "dict"}
            return report
        elif p.suffix == ".txt":
            with open(p, encoding="utf-8") as f:
                lines = f.readlines()
            report["stats"] = {"lines": len(lines)}
            return report
        else:
            report["warnings"].append(f"Unknown format: {p.suffix}")
            return report
    except Exception as e:  # noqa: BLE001
        report["valid"] = False
        report["errors"].append(str(e))
        return report

def validate_jsonl(p, report):
    rows = 0
    msg_count = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                report["errors"].append(f"Row {rows}: invalid JSON - {e}")
                report["valid"] = False
                continue
            if not isinstance(item, dict):
                report["errors"].append(f"Row {rows}: expected an object, got {type(item).__name__}")
                report["valid"] = False
                continue
            if "messages" in item:
                msgs = item["messages"]
                if not isinstance(msgs, list):
                    report["errors"].append(f"Row {rows}: messages must be a list")
                    report["valid"] = False
                else:
                    for j, msg in enumerate(msgs):
                        if not isinstance(msg, dict):
                            report["errors"].append(f"Row {rows}, msg {j}: message must be an object")
                            report["valid"] = False
                            msg_count += 1
                            continue
                        if "role" not in msg:
                            report["errors"].append(f"Row {rows}, msg {j}: missing role")
                            report["valid"] = False
                        if "content" not in msg:
                            report["errors"].append(f"Row {rows}, msg {j}: missing content")
                            report["valid"] = False
                        msg_count += 1
            elif "text" not in item:
                report["warnings"].append(f"Row {rows}: no messages or text key")
    if rows == 0:
        report["errors"].append("File has no rows (empty JSONL)")
        report["valid"] = False
    report["stats"] = {"rows": rows, "messages": msg_count}
    return report
