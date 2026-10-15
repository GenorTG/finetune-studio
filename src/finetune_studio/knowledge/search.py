"""Knowledge search: find facts (mined Q&A pairs) and passages (parsed text) before anything is trained or indexed.

Genor 2026-10-14: "search our files/knowledge for facts and info before we train, to be able to edit things too".
Keyword ranking (Okapi BM25, ``smart_brain.bm25``) over two kinds of hit at once:

- **fact**    a mined Q&A pair (question + answer); editable in place on the pairs page / via the pair routes;
- **passage** a passage of a file's parsed text; opens in the parsed-text editor of its file.

No embedder, no GPU and no built index needed: it works the moment files are parsed. The Smart Brain's own search
(semantic + keyword, rank fusion, reranking) answers questions; this one is for checking and fixing the knowledge.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from finetune_studio.knowledge import sources

KINDS = ("fact", "passage")
_SNIPPET = 320


@dataclass(frozen=True)
class Hit:
    kind: str                   # fact | passage
    score: float
    source_id: str
    file_id: str
    filename: str
    chunk_idx: int              # 1-based passage index (a fact's passage)
    text: str                   # passage snippet, or "question\nanswer"
    pair_id: str = ""
    question: str = ""
    answer: str = ""
    status: str = ""            # a fact's review status: pending | approved | rejected

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def search(pid: str, query: str, *, kinds: tuple[str, ...] = KINDS, status: str | None = None,
           limit: int = 25) -> list[Hit]:
    """Rank the project's facts and passages for ``query`` (best first, only hits that share a word with it)."""
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.smart_brain.bm25 import BM25Index

    query = (query or "").strip()
    if not query:
        return []
    files = {s.id: s for s in sources.list_sources(pid)}
    docs: list[str] = []
    meta: list[Hit] = []
    if "fact" in kinds:
        for p in qa_fs.list_qa_pairs(pid, status=status):
            s = files.get(str(p.get("source_id") or ""))
            q, a = str(p.get("question") or ""), str(p.get("answer") or "")
            docs.append(f"{q}\n{a}")
            meta.append(Hit("fact", 0.0, str(p.get("source_id") or ""), s.file_id if s else "", s.filename if s else "",
                            int(p.get("chunk_idx") or 0), f"{q}\n{a}", pair_id=str(p.get("id") or ""), question=q,
                            answer=a, status=str(p.get("status") or "pending")))
    if "passage" in kinds:
        for s in files.values():
            if not s.ready:
                continue
            for i, text in enumerate(sources.passages(pid, s.sha256), 1):
                if text.strip():
                    docs.append(text)
                    meta.append(Hit("passage", 0.0, s.id, s.file_id, s.filename, i, text))
    if not docs:
        return []
    scores = BM25Index.build(docs).score(query)
    # A hit must share an informative query word: one in at most half the documents ("the", "of" and the project's
    # own name match everything). A query made only of such words falls back to plain ranking.
    from finetune_studio.smart_brain.tokenize import tokenize

    words = [set(tokenize(d)) for d in docs]
    terms = set(tokenize(query))
    informative = {t for t in terms if sum(t in w for w in words) <= max(1, len(docs) // 2)}
    keep = [i for i in range(len(docs)) if scores[i] > 0 and (not informative or informative & words[i])]
    ranked = sorted(keep, key=lambda i: -float(scores[i]))[:max(1, limit)]
    out = []
    for i in ranked:
        h = meta[i]
        text = h.text if h.kind == "fact" else _snippet(h.text, query)
        out.append(Hit(**{**asdict(h), "score": round(float(scores[i]), 4), "text": text}))
    return out


def _snippet(text: str, query: str) -> str:
    """The passage around the first query word it contains (whole passage when short)."""
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= _SNIPPET:
        return flat
    from finetune_studio.smart_brain.tokenize import tokenize

    low = flat.lower()
    pos = min((p for p in (low.find(w) for w in tokenize(query)) if p >= 0), default=0)
    start = max(0, pos - _SNIPPET // 3)
    end = min(len(flat), start + _SNIPPET)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")
