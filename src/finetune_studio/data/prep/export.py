"""JSONL exporters for curated Q&A pairs.

Supported formats:
  sharegpt — {"conversations": [{"from":"human", ...}, {"from":"gpt", ...}], ...}
  alpaca   — {"instruction", "input", "output"}
  openai   — {"messages": [{"role":"user", ...}, {"role":"assistant", ...}]}

`only` filters by status: "approved" | "pending" | "rejected" | "all" (default approved).
"""
from __future__ import annotations

import json

from finetune_studio.data import project_filesystem as pfs


def export_qa_jsonl(pid: str, fmt: str = "sharegpt", only: str = "approved") -> str:
    items = pfs.list_qa_pairs(pid, status=only if only != "all" else None)
    if only == "all":
        items = [q for q in items if q.get("status") != "rejected"]
    if fmt == "sharegpt":
        out = [{"conversations": [
            {"from": "human", "value": it["question"]},
            {"from": "gpt", "value": it["answer"]},
        ], "source_id": it.get("source_id", ""),
           "chunk_idx": it.get("chunk_idx", 0),
           "score": it.get("score", 0.0)} for it in items]
    elif fmt == "alpaca":
        out = [{"instruction": it["question"], "input": "", "output": it["answer"]} for it in items]
    elif fmt == "openai":
        out = [{"messages": [
            {"role": "user", "content": it["question"]},
            {"role": "assistant", "content": it["answer"]},
        ]} for it in items]
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in out) + ("\n" if out else "")
