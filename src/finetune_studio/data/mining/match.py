"""Value matching shared by the inventory grounding check, the coverage check and ``scripts/corpus_coverage.py``.

A value matches a text when it appears in it after normalisation: case, whitespace, dashes and quotes do not matter and a
decimal compares by value (``38.50`` == ``38.5``, as a spreadsheet stores it). A second, lenient form drops all punctuation and
spacing (``45,000 - 89,999`` == ``45,000-89,999``). ``loosely_contains`` accepts a re-worded value: every number of it exact
and at least 80 % of its words (an answer says "base pay then continues in full" for "base pay continues in full").
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from functools import lru_cache

_TRAILING_ZEROS = re.compile(r"(\d+\.\d*?)0+(?!\d)")
_DASHES = str.maketrans({"−": "-", "‑": "-", "–": "-", "—": "-", " ": " ",
                         "’": "'", "‘": "'", "“": '"', "”": '"'})


def norm(text: str) -> str:
    """Case/space/dash/quote-insensitive form; numbers compare by value."""
    s = html.unescape(str(text or "")).translate(_DASHES)
    s = _TRAILING_ZEROS.sub(lambda m: m.group(1).rstrip("."), s)
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", s)).strip().lower()


_NUM = re.compile(r"[+\-]?\d[\d.,]*\d|\d")
_WORDS = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def squash(text: str) -> str:
    """Only letters and digits: the lenient form."""
    return re.sub(r"[^0-9a-zа-яÀ-ɏ]+", "", norm(text))


@dataclass(frozen=True)
class Haystack:
    """A text normalised once, for checking many values against it (coverage checks every item against every answer)."""
    normed: str
    squashed: str


def haystack(text: str | Haystack) -> Haystack:
    if isinstance(text, Haystack):
        return text
    n = norm(text)
    return Haystack(n, re.sub(r"[^0-9a-zа-яÀ-ɏ]+", "", n))


@lru_cache(maxsize=65536)
def _needle(value: str) -> tuple[str, str, frozenset[str], frozenset[str]]:
    """``value`` as (normalised, squashed, its numbers, its 4+-letter words); values repeat across thousands of checks."""
    v = norm(value)
    nums = frozenset(norm(n).lstrip("+") for n in _NUM.findall(value))
    return v, re.sub(r"[^0-9a-zа-яÀ-ɏ]+", "", v), nums, frozenset(w.lower() for w in _WORDS.findall(v))


def contains(value: str, text: str | Haystack) -> bool:
    """True when ``value`` appears in ``text`` (normalised, else squashed)."""
    v, sq, _, _ = _needle(value)
    if not v:
        return False
    h = haystack(text)
    return v in h.normed or (bool(sq) and sq in h.squashed)


def loosely_contains(value: str, text: str | Haystack) -> bool:
    """``contains``, or: all numbers of ``value`` in ``text`` and at least 80 % of its words (4+ letters)."""
    h = haystack(text)
    if contains(value, h):
        return True
    nt = h.normed
    _, _, nums, words = _needle(value)
    if nums and not all(n in nt for n in nums):
        return False
    if not words:
        return bool(nums)
    return sum(1 for w in words if w in nt) >= 0.8 * len(words)
