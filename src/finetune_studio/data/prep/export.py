"""JSONL exporters for curated Q&A pairs.

Supported formats:
  sharegpt — {"conversations": [{"from":"human", ...}, {"from":"gpt", ...}], ...}
  alpaca   — {"instruction", "input", "output"}
  openai   — {"messages": [{"role":"user", ...}, {"role":"assistant", ...}]}

`only` filters by status: "approved" | "pending" | "rejected" | "all" (default approved).

Optional ``grounding`` (``data.prep.grounding``) rewrites a seeded share of rows to carry the
RAG-chat system prompt + a CONTEXT block from the pair's own source chunk; the answer is
unchanged. Rows then gain a system turn (sharegpt ``from: system``, openai ``role: system``,
alpaca ``system`` key) and ``"grounded": true``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.prep.grounding import (
    GroundingOptions,
    GroundingStats,
    apply_grounding,
)
from finetune_studio.training.data import clean_answer_for_training


def deduplicate_qa_pairs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one deterministic training target per (source, chunk, normalized question).

    Curated answers outrank ordinary source-grounded answers, which outrank
    generated augmentation. Within the same class the shorter answer wins so
    a concise target is preferred over a whole serialized source record.

    The key is scoped to (source_id, chunk_idx) as well as the question text.
    Two DIFFERENT sources/chunks can legitimately produce the same auto-
    generated question — e.g. coverage_fill's extractive templates reuse a
    chunk's own leading text as the question "subject", so two CSVs sharing
    an identical header row collide on question text despite holding
    completely different row data. A question-only key silently dropped the
    losing source's entire answer, breaking the per-chunk coverage guarantee
    the export gate had just certified. Scoping by source+chunk keeps the
    original intent (collapse a chunk re-asked the same question twice)
    without ever discarding a different chunk's unique content.
    """
    selected: dict[tuple[str, str, str], tuple[tuple[int, int], int, dict[str, Any]]] = {}
    for index, item in enumerate(items):
        question = re.sub(r"\s+", " ", str(item.get("question") or "")).strip()
        answer = clean_answer_for_training(str(item.get("answer") or "")).strip()
        if not question or not answer:
            continue
        category = str(item.get("category") or "source-grounded")
        priority = 3 if category == "source-grounded-curated" else (
            1 if category == "source-grounded-augmented" else 2
        )
        key = (str(item.get("source_id") or ""), str(item.get("chunk_idx") or ""), question.casefold())
        rank = (priority, -len(answer))
        candidate = {**item, "question": question, "answer": answer}
        current = selected.get(key)
        if current is None or rank > current[0]:
            selected[key] = (rank, index, candidate)
    return [entry[2] for entry in sorted(selected.values(), key=lambda entry: entry[1])]


def _to_jsonl(items: list[dict[str, Any]], fmt: str) -> str:
    # items are already cleaned by deduplicate_qa_pairs
    if fmt == "sharegpt":
        out = []
        for it in items:
            convo = [{"from": "human", "value": it["question"]},
                     {"from": "gpt", "value": it["answer"]}]
            if it.get("system"):
                convo.insert(0, {"from": "system", "value": it["system"]})
            out.append({"conversations": convo, "source_id": it.get("source_id", ""),
                        "chunk_idx": it.get("chunk_idx", 0), "score": it.get("score", 0.0),
                        **_grounded_flag(it)})
    elif fmt == "alpaca":
        out = [{"instruction": it["question"], "input": "", "output": it["answer"],
                **({"system": it["system"]} if it.get("system") else {}),
                "source_id": it.get("source_id", ""), "chunk_idx": it.get("chunk_idx", 0),
                **_grounded_flag(it)} for it in items]
    elif fmt == "openai":
        out = []
        for it in items:
            msgs = [{"role": "user", "content": it["question"]},
                    {"role": "assistant", "content": it["answer"]}]
            if it.get("system"):
                msgs.insert(0, {"role": "system", "content": it["system"]})
            out.append({"messages": msgs, "source_id": it.get("source_id", ""),
                        "chunk_idx": it.get("chunk_idx", 0), **_grounded_flag(it)})
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in out) + ("\n" if out else "")


def _grounded_flag(item: dict[str, Any]) -> dict[str, bool]:
    return {"grounded": True} if item.get("grounded") else {}


@dataclass(frozen=True)
class ExportResult:
    body: str
    rows: int
    grounding: GroundingStats | None = None  # None = plain export


def build_qa_export(
    pid: str,
    fmt: str = "sharegpt",
    only: str = "approved",
    *,
    source_ids: list[str] | None = None,
    grounding: GroundingOptions | None = None,
) -> ExportResult:
    """Dedupe → (optionally ground) → serialise. ``source_ids`` restricts to a subset build."""
    items = pfs.list_qa_pairs(pid, status=only if only != "all" else None)
    if source_ids is not None:
        wanted = set(source_ids)
        items = [q for q in items if q.get("source_id") in wanted]
    if only == "all":
        items = [q for q in items if q.get("status") != "rejected"]
    items = deduplicate_qa_pairs(items)
    stats: GroundingStats | None = None
    if grounding is not None and grounding.share > 0:
        names = {str(s.get("id") or ""): str(s.get("filename") or s.get("name") or "")
                 for s in pfs.list_qa_sources(pid)}
        items, stats = apply_grounding(items, grounding, names)
    return ExportResult(body=_to_jsonl(items, fmt), rows=len(items), grounding=stats)


def export_qa_jsonl(pid: str, fmt: str = "sharegpt", only: str = "approved",
                    grounding: GroundingOptions | None = None) -> str:
    return build_qa_export(pid, fmt, only, grounding=grounding).body


def export_qa_jsonl_from_sources(
    pid: str,
    source_ids: list[str],
    fmt: str = "sharegpt",
    only: str = "approved",
    grounding: GroundingOptions | None = None,
) -> str:
    """Export a dataset built ONLY from the hand-picked sources (subset build).

    Same contract as `export_qa_jsonl` but filtered to the given source ids —
    the base for specialized custom versions trained on selected old+new
    files. Unknown source ids are skipped silently at this layer; the caller
    (route) validates them and reports the real counts.
    """
    return build_qa_export(pid, fmt, only, source_ids=source_ids, grounding=grounding).body
