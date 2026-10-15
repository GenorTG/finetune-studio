"""Question-paraphrase augmentation of REVIEWED pairs (experiment: does SFT recall rise with more formulations per fact?).

    set -a; . ~/.openclaw/workspace/.env_opcgo; set +a
    .venv/bin/python scripts/corpus_paraphrase.py --pid <project> --variants 3 --out .tmp/augmented-sft.jsonl

The reviewed ANSWER is never touched; only the question is re-worded by an API model. A variant is kept only when it (1) differs from
the original and the other variants, (2) keeps every number/code token of the original question and most of its capitalised names,
(3) leaks no number from the answer that the original question did not already contain. Output = sharegpt rows (original + variants)
for "Upload my own" on the Training page. Never prints the key.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_coverage import BASE

URL = "https://opencode.ai/zen/go/v1/chat/completions"
SYSTEM = ("You rewrite questions for a training set. Given a question and its reviewed answer, write {n} DIFFERENT re-wordings of the "
          "QUESTION only: same meaning, same specific names, numbers and codes, different phrasing (vary the sentence form, the "
          "word order, the level of formality). Never answer the question and never copy facts from the answer into the question. "
          "Reply with a JSON list of {n} strings and nothing else.")
_NUM = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-/:]*\d[A-Za-z0-9.\-/:]*|\d+")
_CAP = re.compile(r"(?<=[a-z,;:] )[A-Z][A-Za-zÀ-ɏ\-]{2,}")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", s.lower())).strip()


def anchors(q: str) -> tuple[set[str], set[str]]:
    return {t.lower().strip(".,:") for t in _NUM.findall(q)}, {t.lower() for t in _CAP.findall(q)}


def valid(orig_q: str, answer: str, variant: str, seen: set[str]) -> bool:
    if not (15 <= len(variant) <= 400) or norm(variant) in seen or norm(variant) == norm(orig_q):
        return False
    nums, caps = anchors(orig_q)
    low = variant.lower()
    if not all(n in low for n in nums):
        return False
    if caps and sum(c in low for c in caps) < 0.7 * len(caps):
        return False
    leaked = {t for t in anchors(answer)[0] - nums if t in low and len(t) > 2}
    return not leaked


def ask(key: str, model: str, q: str, a: str, n: int) -> list[str]:
    body = {"model": model, "temperature": 0.7, "max_tokens": 600, "reasoning_effort": "none",
            "messages": [{"role": "system", "content": SYSTEM.format(n=n)},
                         {"role": "user", "content": json.dumps({"question": q, "answer": a}, ensure_ascii=False)}]}
    headers = {"Authorization": f"Bearer {key}", "x-opencode-session": uuid.uuid4().hex}
    for _ in range(3):
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=120)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\[.*\]", text, re.DOTALL)
            out = json.loads(m.group(0)) if m else []
            return [str(x).strip() for x in out if isinstance(x, str)]
        except (requests.RequestException, ValueError, KeyError, IndexError):
            time.sleep(2)   # transient API/JSON problem: back off, retry, then give up on this pair
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=".tmp/augmented-sft.jsonl")
    a = ap.parse_args()
    key = os.environ.get("OPENCODE_GO_API_KEY", "")
    if not key:
        raise SystemExit("OPENCODE_GO_API_KEY is not set (source ~/.openclaw/workspace/.env_opcgo)")
    with urllib.request.urlopen(f"{BASE}/api/projects/{a.pid}/data-prep/qa", timeout=60) as r:
        pairs = [p for p in json.loads(r.read())["items"] if p["status"] == "approved"]
    # one row per (question, answer): the same de-duplication the export applies
    uniq = {}
    for p in pairs:
        uniq.setdefault((norm(p["question"]), norm(p["answer"])), p)
    pairs = list(uniq.values())[: a.limit or None]
    print(f"{len(pairs)} unique approved pairs", flush=True)

    def work(p: dict) -> tuple[dict, list[str]]:
        got = ask(key, a.model, p["question"], p["answer"], a.variants)
        seen = {norm(p["question"])}
        kept = []
        for v in got:
            if valid(p["question"], p["answer"], v, seen):
                kept.append(v)
                seen.add(norm(v))
        return p, kept

    rows, stats = [], {"pairs": len(pairs), "variants": 0, "pairs_without_variant": 0}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, (p, kept) in enumerate(pool.map(work, pairs), 1):
            for q in [p["question"], *kept]:
                rows.append({"conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": p["answer"]}],
                             "source_id": p.get("source_id"), "paraphrase": q != p["question"], "pair_id": p["id"]})
            stats["variants"] += len(kept)
            stats["pairs_without_variant"] += not kept
            if i % 200 == 0:
                print(f"{i}/{len(pairs)} {stats}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out} {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
