"""Heuristic quality score (0..1) for a single Q&A pair.

Single responsibility: take (question, answer, source) → score.

Signals:
  - length windows (Q 8..200, A 30..1500)
  - ends in '?'  (proper question)
  - lexical overlap between answer and source (grounding)
  - common AI-decline phrases (penalty)
"""
from __future__ import annotations

import re


def heuristic_score(q: str, a: str, source: str) -> float:
    if not q or not a:
        return 0.0
    score = 0.5
    qlen, alen = len(q), len(a)
    if 8 <= qlen <= 200:
        score += 0.05
    if 30 <= alen <= 1500:
        score += 0.1
    if q.strip().endswith("?"):
        score += 0.05
    a_low, s_low = a.lower(), source.lower()
    src_words = set(re.findall(r"\w+", s_low))
    a_words = set(re.findall(r"\w+", a_low))
    if src_words and a_words:
        overlap = len(src_words & a_words) / max(len(a_words), 1)
        score += min(0.25, overlap * 0.5)
    if any(ph in a_low for ph in ["as an ai", "i don't have", "i cannot"]):
        score -= 0.3
    return max(0.0, min(1.0, score))
