"""Paragraph/sentence-aware text splitter.

Single responsibility: take a long string, return a list of chunks ~target_chars long,
preserving natural boundaries (paragraph > sentence > hard wrap) and keeping
`overlap` chars of trailing context between adjacent chunks.
"""
from __future__ import annotations

import re


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
        if buf:
            chunks.append(buf)
            tail = buf[-overlap:] if len(buf) > overlap else buf
            buf = (tail + "\n\n" + p).strip()
        else:
            # Single para too long — sentence split
            sents = re.split(r"(?<=[.!?])\s+", p)
            buf2 = ""
            for s in sents:
                cand = (buf2 + " " + s).strip() if buf2 else s
                if len(cand) <= target_chars:
                    buf2 = cand
                else:
                    if buf2:
                        chunks.append(buf2)
                    buf2 = s
            buf = buf2
    if buf:
        chunks.append(buf)
    return chunks
