#!/usr/bin/env python3
"""Re-score saved RAG-suite reports with a normalising matcher, to separate judge strictness from real misses.

The Testing page's keyword judge is a plain substring test, so ``44.2`` != ``44.20``, ``4`` != ``four`` and
``bastion-gdy1`` != ``bastion-gdy1.korvane.example``.  This tool treats a keyword as present when its normalised
form is in the normalised answer (numbers compared by value, number words 0-20 mapped to digits, thousands commas
dropped, a host name matches its first label).  It only re-scores rows the page marked fail/partial.

    .venv/bin/python scripts/rag_rejudge.py .tmp/ragtrace/gemma12-capauto-k20.json [...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

WORDS = {w: str(i) for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"])}


def norm(text: str) -> str:
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower())
    t = re.sub(r"\b(" + "|".join(WORDS) + r")\b", lambda m: WORDS[m.group(1)], t)
    t = re.sub(r"(\d+)\.0+\b|(\d+\.\d*?)0+\b", lambda m: m.group(1) or m.group(2).rstrip("."), t)   # 44.20 -> 44.2, 185.00 -> 185
    return " ".join(t.replace("*", "").replace("`", "").split())


def present(keyword: str, answer: str) -> bool:
    k, a = norm(keyword), norm(answer)
    if k in a:
        return True
    first_label = k.split(".")[0] if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", k) else ""
    return bool(first_label) and first_label in a


def rejudge(row: dict) -> str:
    kws = row.get("keywords") or []
    if not kws or row["category"] == "unanswerable":
        return row["verdict"]
    hits = sum(present(k, row["model_answer"]) for k in kws)
    return "pass" if hits == len(kws) else ("partial" if hits else "fail")


def main() -> int:
    for path in sys.argv[1:]:
        rows = json.loads(Path(path).read_text())["results"]
        quiz = [r for r in rows if r["category"] != "unanswerable"]
        before = sum(r["verdict"] == "pass" for r in quiz)
        after = sum(rejudge(r) == "pass" for r in quiz)
        moved = [r["name"] for r in quiz if r["verdict"] != "pass" and rejudge(r) == "pass"]
        print(f"{Path(path).name}: page {before}/{len(quiz)} -> normalised {after}/{len(quiz)}  (flipped to pass: {', '.join(moved) or '-'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
