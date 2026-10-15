"""Validate a corpus lane's ground-truth manifest against its authored sources.

    .venv/bin/python scripts/corpus_check.py --lane hr [--stats]

Manifest (``tests/corpus/korvane/manifest/<lane>.jsonl``), one JSON object per atomic fact:
  id        unique, ``<lane>-NNNN``
  file      the ``out:`` name of the source it lives in (e.g. ``employee_handbook_2024.pdf``)
  locator   where a human finds it (section number, sheet+cell range, slide number, email date ...)
  fact      ONE self-contained sentence stating the fact (names the subject; no "it"/"the above")
  values    list of exact strings that must appear in the answer to count as covered (numbers, names, codes, dates);
            each must occur verbatim (case/whitespace-insensitive) in the source body
  kind      text | table | list | legal | casual | email | chart | scan
  status    current | superseded     (superseded = a later document in the corpus replaces it)
Checks: unique ids, known file, values present in that source, fact length, no duplicate (file, values) pairs.
Exit code 1 on any problem; ``--stats`` prints facts per file / kind.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_build import ROOT, parse_source

KINDS = {"text", "table", "list", "legal", "casual", "email", "chart", "scan"}


def norm(s: str) -> str:
    s = html.unescape(str(s))  # manifest values may be entity-encoded; parsed text is decoded
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", s)).strip().lower()


def load_sources(lane: str) -> dict[str, tuple[dict, str]]:
    out = {}
    for src in sorted((ROOT / "src" / lane).glob("*.src")):
        meta, body = parse_source(src)
        out[meta["out"]] = (meta, body)
    return out


def check(lane: str) -> tuple[list[str], list[dict]]:
    problems: list[str] = []
    sources = load_sources(lane)
    path = ROOT / "manifest" / f"{lane}.jsonl"
    if not path.exists():
        return [f"{path} does not exist"], []
    facts, seen, pairs = [], set(), set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            f = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: invalid JSON ({e})")
            continue
        fid = f.get("id", f"line{n}")
        facts.append(f)
        if fid in seen or not re.fullmatch(rf"{re.escape(lane)}-\d{{4}}", str(fid)):
            problems.append(f"{fid}: id missing, malformed or duplicated")
        seen.add(fid)
        for key in ("file", "locator", "fact", "values", "kind", "status"):
            if not f.get(key):
                problems.append(f"{fid}: missing `{key}`")
        if f.get("kind") not in KINDS:
            problems.append(f"{fid}: kind must be one of {sorted(KINDS)}")
        if f.get("status") not in ("current", "superseded"):
            problems.append(f"{fid}: status must be current or superseded")
        if f.get("file") not in sources:
            problems.append(f"{fid}: file {f.get('file')!r} is not an `out:` of any source in lane {lane}")
            continue
        body = norm(sources[f["file"]][1])
        for v in f.get("values") or []:
            if norm(str(v)) not in body:
                problems.append(f"{fid}: value {v!r} not found in {f['file']}")
        fact = str(f.get("fact", ""))
        if len(fact.split()) < 5 or re.match(r"^(it|this|that|they|the above)\b", fact.strip(), re.IGNORECASE):
            problems.append(f"{fid}: `fact` must be a full self-contained sentence")
        key = (f.get("file"), tuple(sorted(norm(str(v)) for v in f.get("values") or [])))
        if key in pairs:
            problems.append(f"{fid}: duplicates another fact's (file, values)")
        pairs.add(key)
    return problems, facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    problems, facts = check(a.lane)
    for p in problems[:200]:
        print("PROBLEM", p)
    if a.stats:
        srcs = load_sources(a.lane)
        per = Counter(f["file"] for f in facts)
        for name, (meta, body) in srcs.items():
            words = len(body.split())
            print(f"{name:48} tier={meta.get('tier','core'):8} words={words:6} facts={per.get(name, 0):4}"
                  f"  words/fact={words / max(per.get(name, 0), 1):5.1f}")
        print("kinds:", dict(Counter(f.get("kind") for f in facts)), "total facts:", len(facts))
    print(f"{len(problems)} problem(s), {len(facts)} fact(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
