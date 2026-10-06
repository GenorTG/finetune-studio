"""``app_help`` retrieval: BM25 over KB sections (no embedder, no GPU).

Reuses the RAG package's ``BM25Index``. Each KB section is one document whose
text is the entry title + keywords + section name + body, so a question matches
either the feature's vocabulary or the section's content.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from finetune_studio.data.rag_portable.bm25 import BM25Index
from finetune_studio.data.rag_portable.tokenize import tokenize
from finetune_studio.guide.kb import KbEntry, load_kb

SNIPPET_CHARS = 1200
_STOP_TEXT = (
    "a an and are as at be by can could do does for from give has have how i if in into is it its me my "
    "of on or our should so tell that the their then there these this to us use using was we what when "
    "where which who why will with would you your want need get make please explain show help about"
)
_STOPWORDS = frozenset(_STOP_TEXT.split())
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s")


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def _normalize(text: str, *, query: bool = False) -> str:
    """Lowercase, drop filler words, strip plural/-ing endings: 'training' matches 'train'.

    In a question the word "model" is on nearly every page, so it is dropped when other
    content words remain ("how do I train my model" is about training, not the model library).
    """
    tokens = [t for t in tokenize(text) if t not in _STOPWORDS]
    if query and len(tokens) > 1:
        tokens = [t for t in tokens if t not in ("model", "models")] or tokens
    return " ".join(_stem(t) for t in tokens)


@lru_cache(maxsize=1)
def _index() -> tuple[BM25Index, list[tuple[KbEntry, str]]]:
    chunks: list[tuple[KbEntry, str]] = []
    docs: list[str] = []
    for entry in load_kb().values():
        for name, body in entry.sections.items():
            chunks.append((entry, name))
            head = f"{entry.title} {' '.join(entry.keywords)} {entry.id}"
            docs.append(_normalize(f"{head} {head} {name} {body}"))  # head twice: feature vocabulary outweighs body prose
    return BM25Index.build(docs), chunks


def search_kb(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Top sections for a question, one result per entry (best section wins)."""
    index, chunks = _index()
    scores = index.score(_normalize(query, query=True))
    ranked = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in ranked:
        score = float(scores[i])
        if score <= 0:
            break
        entry, section = chunks[i]
        if entry.id in seen:
            continue
        seen.add(entry.id)
        out.append({
            "entry": entry.id,
            "title": entry.title,
            "page": entry.route,
            "section": section,
            "score": round(score, 2),
            "text": entry.sections[section][:SNIPPET_CHARS],
        })
        if len(out) >= max(1, min(limit, 5)):
            break
    return out
