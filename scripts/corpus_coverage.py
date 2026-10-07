"""Fact coverage of a project's Q&A pairs against the corpus manifests (the real 'did the miner digest the file' gate).

    .venv/bin/python scripts/corpus_coverage.py --pid <project> [--tier core] [--status approved] [--json out.json]

For every manifest fact of the selected tier, a fact is COVERED when one pair's *answer* contains every value of the fact
(case/whitespace/dash-insensitive; a second 'lenient' pass ignores all punctuation and spacing). Pairs are matched to a
file by the project source's filename. Output: per-file covered/total, every uncovered fact with its locator, and the
share of pairs that carry no manifest fact at all (candidates for the manual review: extra but possibly valid facts, or noise).
Exit code 1 unless coverage is 100 % (the goal Genor set: 100 % or as close as reproducible).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_build import ROOT, parse_source

BASE = "http://127.0.0.1:7860"


_TRAILING_ZEROS = re.compile(r"(\d+\.\d*?)0+(?![\d])")


def norm(s: str) -> str:
    """Case/space/dash/quote-insensitive form; numbers compare by value (38.50 == 38.5, as a spreadsheet stores them)."""
    s = str(s).replace("\u2212", "-")
    s = _TRAILING_ZEROS.sub(lambda m: m.group(1).rstrip("."), html.unescape(str(s)))
    s = html.unescape(s)  # manifest values may be entity-encoded; parsed text is decoded
    s = s.replace(" ", " ").replace("‑", "-").replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", s)).strip().lower()


def squash(s: str) -> str:
    return re.sub(r"[^0-9a-zа-яÀ-ɏ]+", "", norm(s))


_NUM = re.compile(r"[+\-]?\d[\d.,]*\d|\d")
_WORDS = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def lenient_value_in(value: str, blob: str) -> bool:
    """Token-level proxy for a value the answer may have re-worded: all its numbers present, and (for words) 80 % of them."""
    nb = norm(blob)
    nums = {norm(n).lstrip("+") for n in _NUM.findall(value)}
    if nums and not all(n in nb for n in nums):
        return False
    words = {w.lower() for w in _WORDS.findall(norm(value))}
    if not words:
        return bool(nums)
    return sum(1 for w in words if w in nb) >= 0.8 * len(words)


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def manifest_facts(tier: str, lanes: list[str] | None, files: list[str] | None = None) -> list[dict]:
    tiers: dict[str, str] = {}
    for src in (ROOT / "src").glob("*/*.src"):
        meta, _ = parse_source(src)
        tiers[meta["out"]] = meta.get("tier", "core")
    facts = []
    for mf in sorted((ROOT / "manifest").glob("*.jsonl")):
        if lanes and mf.stem not in lanes:
            continue
        for line in mf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                f = json.loads(line)
                if files and f["file"] not in files:
                    continue
                if tier == "all" or tiers.get(f["file"], "core") == tier:
                    facts.append(f)
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--tier", default="core", choices=["core", "extended", "all"])
    ap.add_argument("--lanes", nargs="*")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--status", default="approved", help="pair status to count: approved | pending | any")
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args()

    facts = manifest_facts(a.tier, a.lanes, a.files)
    sources = {s["id"]: s["filename"] for s in get(f"/api/projects/{a.pid}/data-prep/sources")["sources"]}
    pairs = get(f"/api/projects/{a.pid}/data-prep/qa")["items"]
    if a.status != "any":
        pairs = [p for p in pairs if p.get("status") == a.status]
    by_file: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_file[sources.get(p.get("source_id"), "?")].append(p)

    per_file: dict[str, dict] = {}
    used: set[str] = set()
    for f in facts:
        row = per_file.setdefault(f["file"], {"total": 0, "covered": 0, "lenient_only": 0, "missing": []})
        row["total"] += 1
        hit = None
        for p in by_file.get(f["file"], []):
            ans = str(p.get("answer", ""))
            if all(norm(v) in norm(ans) for v in f["values"]):
                hit = p
                break
        if hit is None:
            for p in by_file.get(f["file"], []):
                if all(squash(v) in squash(str(p.get("answer", ""))) for v in f["values"]):
                    hit = p
                    row["lenient_only"] += 1
                    break
        if hit is None:
            # The pairs of the chunk(s) that contain the fact: together, do their answers state every value?
            near = [p for p in by_file.get(f["file"], [])
                    if all(norm(v) in norm(str(p.get("chunk_text", ""))) or squash(v) in squash(str(p.get("chunk_text", "")))
                           for v in f["values"])]
            blob = " ".join(str(p.get("answer", "")) for p in near)
            if near and all(norm(v) in norm(blob) or squash(v) in squash(blob) for v in f["values"]):
                row["union_only"] = row.get("union_only", 0) + 1
                row["covered"] += 1
                used.update(str(p.get("id")) for p in near)
                continue
        if hit is None:
            near = [p for p in by_file.get(f["file"], [])
                    if all(lenient_value_in(v, str(p.get("chunk_text", ""))) for v in f["values"])]
            blob = " ".join(str(p.get("answer", "")) for p in near)
            if near and all(lenient_value_in(v, blob) for v in f["values"]):
                row["lenient_only"] += 1
                row["covered"] += 1
                used.update(str(p.get("id")) for p in near)
                continue
        if hit is not None:
            row["covered"] += 1
            used.add(str(hit.get("id")))
        else:
            row["missing"].append({"id": f["id"], "locator": f["locator"], "fact": f["fact"], "values": f["values"],
                                   "kind": f["kind"]})
    tot = sum(r["total"] for r in per_file.values())
    cov = sum(r["covered"] for r in per_file.values())
    print(f"{'file':58} covered/total   pairs")
    for name, r in sorted(per_file.items()):
        print(f"{name:58} {r['covered']:4}/{r['total']:<4} {100 * r['covered'] / r['total']:5.1f}%  {len(by_file.get(name, [])):5}")
    unmatched = [p for p in pairs if str(p.get("id")) not in used]
    union = sum(r.get("union_only", 0) for r in per_file.values())
    lenient = sum(r["lenient_only"] for r in per_file.values())
    print(f"of the covered facts, {union} are stated across several answers of their chunk and {lenient} only by token-level "
          "match (re-worded values): treat those as 'probably' until the manual review confirms")
    print(f"TOTAL {cov}/{tot} = {100 * cov / max(tot, 1):.1f}%   pairs counted: {len(pairs)}   pairs matching no fact: {len(unmatched)}")
    for name, r in sorted(per_file.items()):
        for m in r["missing"][:60]:
            print(f"  MISSING {m['id']} [{name} · {m['locator']}] {m['values']}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"per_file": per_file, "total": tot, "covered": cov,
                                                "unmatched_pair_ids": [p.get("id") for p in unmatched]}, indent=2))
    return 0 if cov == tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
