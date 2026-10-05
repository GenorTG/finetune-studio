"""Context-grounded training rows for RAG-chat use.

Evidence (``.tmp/rag-grounding/RESULTS.md``, 2026-10-05): a small model tuned on
plain QA rows ignores the CONTEXT that ``rag/chat`` hands it (Qwen3-0.6B: 2/5
grounded); mixing in rows whose prompt carries the pair's own source passage
lifted it to 4/5. This module turns a seeded share of approved pairs into such
rows — same system-prompt layout ``rag_chat`` builds (``rag_portable.prompt``),
answer unchanged. The remaining pairs stay plain, so the model still answers
without context.

Context text comes from the pair's own stored ``chunk_text`` (no corpus load);
distractors are chunks of *other* source files, so the model also learns to
ignore irrelevant retrieved passages. Pure + deterministic: same pairs, same
options, same bytes.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from finetune_studio.data.fs.paths import rag_corpus_dir
from finetune_studio.data.rag_portable.prompt import (
    build_system_prompt,
    format_context_blocks,
    synthetic_hit,
)

DEFAULT_SHARE = 0.4
MAX_DISTRACTORS = 2
# rag_chat caps CONTEXT at 4000 chars; keep every grounded row under that so
# its gold block is never dropped by format_context_blocks.
_CONTEXT_BUDGET = 3600


@dataclass(frozen=True)
class GroundingOptions:
    """share = fraction of pairs rewritten; distractors = other-source chunks added per row."""

    share: float = DEFAULT_SHARE
    distractors: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.share <= 1.0:
            raise ValueError(f"grounded share must be between 0 and 1, got {self.share}")
        if not 0 <= self.distractors <= MAX_DISTRACTORS:
            raise ValueError(f"distractors must be 0-{MAX_DISTRACTORS}, got {self.distractors}")


@dataclass(frozen=True)
class GroundingStats:
    rows: int = 0
    grounded: int = 0
    with_distractors: int = 0
    no_chunk_text: int = 0  # eligible by share but the pair stored no source text → stayed plain

    def as_dict(self) -> dict[str, int]:
        return {"rows": self.rows, "grounded": self.grounded,
                "with_distractors": self.with_distractors, "no_chunk_text": self.no_chunk_text}


def corpus_built(pid: str) -> bool:
    """True when the project has a built RAG corpus (what rag/chat retrieves from)."""
    from finetune_studio.data.rag_portable import PortableRAG
    return PortableRAG(rag_corpus_dir(pid)).exists()


def resolve_grounding(pid: str, share: float | None = None, distractors: int = 0,
                      seed: int = 42) -> GroundingOptions | None:
    """Turn request parameters into options; ``None`` = plain export.

    ``share=None`` means "auto": on at ``DEFAULT_SHARE`` when the project has a
    built RAG corpus (its tuned model will be asked questions with retrieved
    context), off otherwise. An explicit ``0`` turns it off; an explicit value
    is honoured even without a corpus. Raises ``ValueError`` on bad values.
    """
    if share is None:
        share = DEFAULT_SHARE if corpus_built(pid) else 0.0
    opts = GroundingOptions(share=float(share), distractors=int(distractors), seed=int(seed))
    return opts if opts.share > 0 else None


def _stable_key(seed: int, item: dict[str, Any]) -> str:
    raw = f"{seed}|{item.get('source_id', '')}|{item.get('chunk_idx', '')}|{item['question']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip()


def apply_grounding(items: list[dict[str, Any]], opts: GroundingOptions,
                    filenames: dict[str, str]) -> tuple[list[dict[str, Any]], GroundingStats]:
    """Return ``items`` with a seeded share carrying ``system`` (the rag_chat prompt) + ``grounded``.

    ``items`` are deduplicated pairs (question/answer already cleaned);
    ``filenames`` maps ``source_id`` → the filename cited in the context block.
    Order and plain rows are untouched.
    """
    target = round(opts.share * len(items))
    ranked = sorted(range(len(items)), key=lambda i: _stable_key(opts.seed, items[i]))

    def label(item: dict[str, Any]) -> str:
        return filenames.get(str(item.get("source_id") or "")) or "source"

    # Distractor pool: one chunk per (source, chunk), sorted so the draw is deterministic.
    pool: dict[tuple[str, str], tuple[str, str]] = {}
    if opts.distractors:
        for it in items:
            text = str(it.get("chunk_text") or "").strip()
            if text:
                pool.setdefault((str(it.get("source_id") or ""), str(it.get("chunk_idx") or "")),
                                (label(it), text))
    pool_keys = sorted(pool)

    clip = _CONTEXT_BUDGET // (1 + opts.distractors)
    out = list(items)
    grounded = with_dist = no_text = 0
    for idx in ranked[:target]:
        item = items[idx]
        gold_text = _clip(str(item.get("chunk_text") or ""), clip)
        if not gold_text:
            no_text += 1
            continue
        rng = random.Random(_stable_key(opts.seed, item))
        passages = [(label(item), gold_text)]
        own = str(item.get("source_id") or "")
        others = [k for k in pool_keys if k[0] != own and pool[k][1] != str(item.get("chunk_text") or "").strip()]
        for k in rng.sample(others, min(opts.distractors, len(others))):
            passages.append((pool[k][0], _clip(pool[k][1], clip)))
        if len(passages) > 1:
            with_dist += 1
        rng.shuffle(passages)
        hits = [synthetic_hit(rank, fname, text) for rank, (fname, text) in enumerate(passages, 1)]
        out[idx] = {**item, "grounded": True,
                    "system": build_system_prompt(format_context_blocks(hits))}
        grounded += 1
    return out, GroundingStats(rows=len(items), grounded=grounded,
                               with_distractors=with_dist, no_chunk_text=no_text)
