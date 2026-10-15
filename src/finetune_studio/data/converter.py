"""Convert various formats to JSONL training data.

WHAT THIS FILE DOES
==================
Converts CSV and JSON files into the JSONL format expected by the
training pipeline, for both the CLI (`fts convert`) and WebUI dataset
uploads (`records_from_upload`).

KEY CONCEPTS
============
- JSONL (JSON Lines): each line is a separate JSON object. Unlike
  regular JSON, it's not wrapped in [...]. Easy to stream and process.
- Training example format: {"messages": [{"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}]}. Records already in a shape
  ``training.data.format_for_sft`` reads (messages / conversations / text /
  prompt+completion) pass through unchanged.
- Column mapping: tabular rows (CSV, or flat JSON objects) map a question-like
  column (question, prompt, instruction, input, query, user) to the user turn
  and an answer-like column (answer, response, output, completion, assistant)
  to the assistant turn; an optional "system" column becomes the system turn.
  Alpaca rows (instruction + input + output) join instruction and input.
  A lone "text" column is kept as plain text.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

# Shapes supported by the SFT formatter (or the preference trainer) pass through.
_TRAINABLE_KEYS = ("messages", "conversations", "text")
_USER_COLS = ("question", "prompt", "instruction", "query", "user", "input")
_ASSISTANT_COLS = ("answer", "response", "output", "completion", "assistant")
_WRAPPER_KEYS = ("data", "examples", "rows", "records", "items")
UPLOAD_SUFFIXES = (".jsonl", ".json", ".csv")


def _is_trainable(rec: dict) -> bool:
    return (
        any(k in rec for k in _TRAINABLE_KEYS)
        or ("prompt" in rec and "completion" in rec)
        or {"prompt", "chosen", "rejected"}.issubset(rec)
    )


def _pick(row: dict[str, Any], names: tuple[str, ...]) -> str | None:
    """First of ``names`` present in ``row`` (keys already lower-cased)."""
    return next((n for n in names if n in row), None)


def row_to_record(row: dict[str, Any], system_prompt: str = "") -> dict:
    """Map one flat row (CSV row or JSON object) to a trainable record.

    Raises ``ValueError`` naming the columns when no mapping applies.
    """
    if _is_trainable(row):
        return row
    low = {str(k).strip().lower(): ("" if v is None else str(v)) for k, v in row.items()}
    user_col = _pick(low, _USER_COLS)
    asst_col = _pick(low, _ASSISTANT_COLS)
    if user_col and asst_col and user_col != asst_col:
        user = low[user_col].strip()
        # Alpaca: instruction + input -> one user turn.
        if user_col == "instruction" and low.get("input", "").strip():
            user = f"{user}\n\n{low['input'].strip()}"
        messages = []
        system = low.get("system", "").strip() or system_prompt
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": low[asst_col].strip()})
        return {"messages": messages}
    if "text" in low:
        return {"text": low["text"]}
    cols = ", ".join(str(k) for k in row)
    raise ValueError(
        f"cannot map columns ({cols}) to training examples; expected a question/prompt "
        "column plus an answer/response column, a 'text' column, or 'messages'"
    )


def _rows_to_records(rows: list, system_prompt: str) -> list[dict]:
    out = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            # Bad *file content*, not a programming error: callers report ValueError as a 400.
            raise ValueError(f"example {i} is not a JSON object")  # noqa: TRY004
        out.append(row_to_record(row, system_prompt))
    return out


def _parse_jsonl(text: str) -> list:
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON on line {n}: {e.msg}") from e
    return rows


def _parse_json(text: str) -> list:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e.msg} (line {e.lineno})") from e
    if isinstance(data, dict):
        wrapped = next((data[k] for k in _WRAPPER_KEYS if isinstance(data.get(k), list)), None)
        return wrapped if wrapped is not None else [data]
    if isinstance(data, list):
        return data
    raise ValueError("JSON must be an array of examples or an object")


def records_from_upload(raw: bytes, filename: str, system_prompt: str = "") -> list[dict]:
    """Parse an uploaded dataset file into trainable records.

    Raises ``ValueError`` with a user-readable reason for unsupported formats,
    malformed content, unmappable columns, or an empty result.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise ValueError(f"unsupported file type '{suffix or filename}'; upload .jsonl, .json or .csv")
    text = raw.decode("utf-8-sig", errors="replace")
    if suffix == ".jsonl":
        rows = _parse_jsonl(text)
    elif suffix == ".json":
        rows = _parse_json(text)
    else:
        rows = list(csv.DictReader(io.StringIO(text)))
    records = _rows_to_records(rows, system_prompt)
    if not records:
        raise ValueError("file contains no training examples")
    return records


def write_jsonl(records: list[dict], jsonl_path: str | Path) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def jsonl_to_json(jsonl_path: str, json_path: str) -> None:
    data = _parse_jsonl(Path(jsonl_path).read_text(encoding="utf-8"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def json_to_jsonl(json_path: str, jsonl_path: str) -> None:
    rows = _parse_json(Path(json_path).read_text(encoding="utf-8"))
    write_jsonl(_rows_to_records(rows, ""), jsonl_path)


def csv_to_jsonl(csv_path: str, jsonl_path: str, system_prompt: str = "") -> None:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    write_jsonl(_rows_to_records(rows, system_prompt), jsonl_path)
