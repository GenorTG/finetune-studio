"""JSONL exporters for curated Q&A pairs.

Supported formats:
  sharegpt — {"conversations": [{"from":"human", ...}, {"from":"gpt", ...}], ...}
  alpaca   — {"instruction", "input", "output"}
  openai   — {"messages": [{"role":"user", ...}, {"role":"assistant", ...}]}

`only` filters by status: "approved" | "pending" | "rejected" | "all" (default approved).
"""
from __future__ import annotations

import json
import re
from typing import Any

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.training.data import clean_answer_for_training


def deduplicate_qa_pairs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one deterministic training target per normalized question.

    Curated answers outrank ordinary source-grounded answers, which outrank
    generated augmentation. Within the same class the shorter answer wins so
    a concise target is preferred over a whole serialized source record.
    """
    selected: dict[str, tuple[tuple[int, int], int, dict[str, Any]]] = {}
    for index, item in enumerate(items):
        question = re.sub(r"\s+", " ", str(item.get("question") or "")).strip()
        answer = clean_answer_for_training(str(item.get("answer") or "")).strip()
        if not question or not answer:
            continue
        category = str(item.get("category") or "source-grounded")
        priority = 3 if category == "source-grounded-curated" else (
            1 if category == "source-grounded-augmented" else 2
        )
        key = question.casefold()
        rank = (priority, -len(answer))
        candidate = {**item, "question": question, "answer": answer}
        current = selected.get(key)
        if current is None or rank > current[0]:
            selected[key] = (rank, index, candidate)
    return [entry[2] for entry in sorted(selected.values(), key=lambda entry: entry[1])]


def export_qa_jsonl(pid: str, fmt: str = "sharegpt", only: str = "approved") -> str:
    items = pfs.list_qa_pairs(pid, status=only if only != "all" else None)
    if only == "all":
        items = [q for q in items if q.get("status") != "rejected"]
    items = deduplicate_qa_pairs(items)
    if fmt == "sharegpt":
        out = [{"conversations": [
            {"from": "human", "value": it["question"]},
            {"from": "gpt", "value": clean_answer_for_training(it["answer"])},
        ], "source_id": it.get("source_id", ""),
           "chunk_idx": it.get("chunk_idx", 0),
           "score": it.get("score", 0.0)} for it in items]
    elif fmt == "alpaca":
        out = [{"instruction": it["question"], "input": "", "output": clean_answer_for_training(it["answer"]),
                "source_id": it.get("source_id", ""), "chunk_idx": it.get("chunk_idx", 0)} for it in items]
    elif fmt == "openai":
        out = [{"messages": [
            {"role": "user", "content": it["question"]},
            {"role": "assistant", "content": clean_answer_for_training(it["answer"])},
        ], "source_id": it.get("source_id", ""),
           "chunk_idx": it.get("chunk_idx", 0)} for it in items]
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in out) + ("\n" if out else "")


def export_qa_jsonl_from_sources(
    pid: str,
    source_ids: list[str],
    fmt: str = "sharegpt",
    only: str = "approved",
) -> str:
    """Export a dataset built ONLY from the hand-picked sources (subset build).

    Same contract as `export_qa_jsonl` but filtered to the given source ids —
    the base for specialized custom versions trained on selected old+new
    files. Unknown source ids are skipped silently at this layer; the caller
    (route) validates them and reports the real counts.
    """
    wanted = set(source_ids)
    items = [q for q in pfs.list_qa_pairs(pid, status=only if only != "all" else None)
             if q.get("source_id") in wanted]
    if only == "all":
        items = [q for q in items if q.get("status") != "rejected"]
    items = deduplicate_qa_pairs(items)
    if fmt == "sharegpt":
        out = [{"conversations": [
            {"from": "human", "value": it["question"]},
            {"from": "gpt", "value": clean_answer_for_training(it["answer"])},
        ], "source_id": it.get("source_id", ""),
           "chunk_idx": it.get("chunk_idx", 0),
           "score": it.get("score", 0.0)} for it in items]
    elif fmt == "alpaca":
        out = [{"instruction": it["question"], "input": "", "output": clean_answer_for_training(it["answer"]),
                "source_id": it.get("source_id", ""), "chunk_idx": it.get("chunk_idx", 0)} for it in items]
    elif fmt == "openai":
        out = [{"messages": [
            {"role": "user", "content": it["question"]},
            {"role": "assistant", "content": clean_answer_for_training(it["answer"])},
        ], "source_id": it.get("source_id", ""),
           "chunk_idx": it.get("chunk_idx", 0)} for it in items]
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in out) + ("\n" if out else "")
