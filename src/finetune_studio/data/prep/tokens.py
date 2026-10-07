"""Distinctive-token helpers shared by mining coverage and answer validation (no imports from the rest of ``prep``).

A *distinctive token* is something a reader could be asked about: anything with a digit, a code (``RC-01``, ``hub-gdy1-03``), an
ALLCAPS word, or a capitalised word in mid-sentence (a name). ``canon`` makes two spellings of the same token comparable.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[\w][\w.\-/:@#%+]*", re.UNICODE)
_NUMBERISH = re.compile(r"\d")
_CODE = re.compile(r"[A-Za-z]+[-_/][A-Za-z0-9]+|[A-Z]{2,}\d*")
_TRAILING_ZEROS = re.compile(r"(\d+\.\d*?)0+(?!\d)")
_SENT_START_STOP = frozenset({
    "the", "a", "an", "if", "when", "in", "on", "at", "for", "to", "of", "and", "or", "but", "this", "that", "these", "those",
    "we", "you", "it", "he", "she", "they", "our", "your", "their", "all", "any", "each", "every", "no", "not", "once", "after",
    "before", "during", "where", "while", "as", "by", "from", "with", "without", "please", "note", "see", "per", "under",
})


def canon(token: str) -> str:
    """Comparison form: lowercase, edge punctuation off, trailing zeros of decimals off (38.50 == 38.5)."""
    t = token.strip(".,;:()[]{}\"'`!?").lower()
    for dash in ("\u2011", "\u2212", "\u2013", "\u2014"):
        t = t.replace(dash, "-")
    return _TRAILING_ZEROS.sub(lambda m: m.group(1).rstrip("."), t).rstrip(".")


def distinctive_tokens(text: str) -> frozenset[str]:
    """Tokens a reader could be asked about: anything with a digit, codes, ALLCAPS, and capitalised words mid-sentence."""
    out: set[str] = set()
    words = _WORD.findall(text)
    for i, w in enumerate(words):
        c = canon(w)
        if len(c) < 2:
            continue
        mid_sentence_name = (
            w[0].isupper() and i > 0 and c not in _SENT_START_STOP and not words[i - 1].endswith((".", ":", "!", "?"))
        )
        if _NUMBERISH.search(c) or _CODE.fullmatch(w.strip(".,;:()")) or (w.isupper() and len(w) > 2) or mid_sentence_name:
            out.add(c)
    return frozenset(out)


