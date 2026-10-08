#!/usr/bin/env python3
"""Compare RAG *readers* (models) and context caps on the same indexed project.

Drives ``POST /api/testing/run-rag-suite`` (the Testing page's RAG-grounded quiz) once per
reader x context-cap, saves each full report, and prints where every miss comes from:

* ``not retrieved``  - the expected values are not in the top-k chunks (retrieval problem),
* ``cut by cap``     - retrieved, but the context cap dropped the chunk before the model saw it,
* ``reader missed``  - the model saw the values and still did not pass (reader problem),
* ``wrongly answered`` - an unanswerable question that got a confident answer.

    .venv/bin/python scripts/rag_reader_compare.py --pid e9f951f8 \
        --reader base-q4=/path/base.gguf --reader gemma12=/path/gemma.gguf --cap 5000 --cap 16000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUIZ = ROOT / "tests" / "corpus" / "korvane" / "eval" / "korvane_quiz_core.jsonl"


def run_suite(base: str, pid: str, model_path: str, *, top_k: int, cap: int, quiz: Path) -> dict:
    """One blocking RAG-suite run through the app (loads ``model_path`` if it is not the loaded one)."""
    body = {"project_id": pid, "suite_path": str(quiz), "model_path": model_path, "top_k": top_k,
            "max_context_chars": cap, "max_tokens": 512, "temperature": 0.0}
    req = urllib.request.Request(f"{base}/api/testing/run-rag-suite", method="POST",
                                 headers={"Content-Type": "application/json"}, data=json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=4 * 3600) as resp:
        return json.load(resp)


def split_misses(results: list[dict]) -> Counter:
    """Classify every non-pass row by where the answer was lost."""
    out: Counter = Counter()
    for r in results:
        if r.get("verdict") == "pass":
            out["pass " + r["category"]] += 1
        elif r["category"] == "unanswerable":
            out["wrongly answered"] += 1
        elif not r.get("gold_in_retrieved"):
            out["not retrieved"] += 1
        elif not r.get("gold_in_context"):
            out["cut by cap"] += 1
        else:
            out["reader missed"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--reader", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--cap", action="append", type=int, default=[])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--base", default="http://127.0.0.1:7860")
    ap.add_argument("--quiz", type=Path, default=QUIZ)
    ap.add_argument("--out", type=Path, default=ROOT / ".tmp" / "ragtrace")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.reader:
        name, _, path = spec.partition("=")
        for cap in args.cap or [5000]:
            report = run_suite(args.base, args.pid, path, top_k=args.top_k, cap=cap, quiz=args.quiz)
            (args.out / f"{name}-cap{cap}-k{args.top_k}.json").write_text(json.dumps(report, indent=1))
            counts = split_misses(report["results"])
            print(f"{name} cap={cap} k={args.top_k}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                  f"| chunks in context avg {report['retrieval']['mean_chunks_in_context']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
