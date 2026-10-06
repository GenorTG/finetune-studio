"""Paragraph/sentence-aware text splitter.

Single responsibility: take a long string, return a list of chunks ~target_chars long,
preserving natural boundaries (paragraph > sentence > hard wrap) and keeping
`overlap` chars of trailing context between adjacent chunks.
"""
from __future__ import annotations

import re

# A buffer shorter than this (typically a lone heading) carries no standalone
# facts. It is never flushed as its own chunk ahead of an oversized paragraph:
# coverage-fill cannot mint a pair from "# Lore", so such a chunk would be a
# permanent coverage hole that blocks dataset export.
_MIN_STANDALONE_CHARS = 200


def _tail(text: str, overlap: int) -> str:
    """Last ~``overlap`` chars of ``text``, starting on a word boundary.

    A raw ``text[-overlap:]`` starts mid-word ("r support" for "Customer support"); that fragment led the
    next chunk and later became a heading in generated questions."""
    if len(text) <= overlap:
        return text.strip()
    cut = text[-overlap:]
    if not text[-overlap - 1].isspace():          # the slice starts inside a word: drop the partial word
        parts = cut.split(None, 1)
        if len(parts) == 2:
            cut = parts[1]
        # no whitespace at all (one giant token): there is no boundary to snap to, keep the raw slice
    return cut.strip()


def _pack_sentences(prefix: str, para: str, target_chars: int) -> tuple[list[str], str]:
    """Greedy-pack ``para``'s sentences after ``prefix``.

    Returns (full chunks, trailing partial buffer).
    """
    chunks: list[str] = []
    buf = prefix
    for s in re.split(r"(?<=[.!?])\s+", para):
        sep = "\n\n" if buf == prefix and prefix else " "
        cand = (buf + sep + s).strip() if buf else s
        if len(cand) <= target_chars:
            buf = cand
        else:
            if buf:
                chunks.append(buf)
            buf = s
    return chunks, buf


def chunk_text(text: str, target_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Split on paragraph boundaries; fall back to sentence boundaries; then hard wrap.

    Returns list of chunks. Each chunk ends with the last paragraph boundary
    that fits, with `overlap` chars of trailing context prepended to the next chunk.
    """
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        cand = (buf + "\n\n" + p).strip() if buf else p
        if len(cand) <= target_chars:
            buf = cand
            continue
        if buf and len(p) > target_chars:
            # Oversized paragraph after buffered text: it still needs a
            # sentence split. A short buffer (heading) leads the first
            # sentences instead of becoming a fact-free chunk of its own.
            if len(buf) < _MIN_STANDALONE_CHARS:
                prefix = buf
            else:
                chunks.append(buf)
                prefix = _tail(buf, overlap)
            packed, buf = _pack_sentences(prefix, p, target_chars)
            chunks.extend(packed)
        elif buf:
            chunks.append(buf)
            tail = _tail(buf, overlap)
            buf = (tail + "\n\n" + p).strip()
        else:
            # Single para too long — sentence split
            packed, buf = _pack_sentences("", p, target_chars)
            chunks.extend(packed)
    if buf:
        chunks.append(buf)
    return chunks
