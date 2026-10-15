"""Unicode-aware tokenizer (shared by BM25 and any code that wants word tokens).

Single responsibility: turn a string into a list of lowercased word tokens
that work for English, Polish diacritics, CJK, etc.
"""
from __future__ import annotations

import re

# Words (letters+digits, allow internal apostrophes/hyphens) OR
# contiguous high-range unicode (covers CJK and other scripts).
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[\-'][A-Za-z0-9]+)*|[\u00A0-\uFFFF]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercased, unicode-aware word tokens. Works for English, Polish diacritics, etc."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]
