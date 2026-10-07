"""Compare helper backends (local GGUF vs API providers) on the work the helper really does.

    .venv/bin/python scripts/helper_bench.py --label deepseek-v4-flash [--pid <project>] [--skip-pref]

Runs against the LIVE service with whatever currently holds the helper seat (Settings → Helper model), on an
existing throwaway project that has approved Q&A pairs (e.g. after the walkthrough phases create/upload/prep/
review). Measures:
  * Guide: five fixed questions through ``POST /api/guide/chat`` — wall time, tool calls, UI events, whether a
    final answer arrived, and the answer text (read it: speed means nothing if the advice is wrong).
  * Preference authoring: ``POST /data-prep/preference`` for ``--max-pairs`` pairs — wall time, kept counts.
Mining speed/quality comes from ``tests/e2e_user_walkthrough.py --phase prep`` (it prints time and fact coverage).
Results go to ``.tmp/helper-bench/<label>.json``. Never prints keys.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO = Path(__file__).resolve().parents[1]
BASE = os.environ.get("FTS_BASE", "http://127.0.0.1:7860").rstrip("/")
OUT = REPO / ".tmp" / "helper-bench"

QUESTIONS = [
    "What should I do next in this project?",
    "Which training settings should I use for my approved pairs, and why?",
    "How do I export a GGUF of my trained model?",
    "Is my dataset good enough to train on?",
    "Take me to the testing page and tell me what to click there.",
]


def guide_turn(pid: str, question: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    events: list[dict[str, Any]] = []
    first = None
    with requests.post(f"{BASE}/api/guide/chat", stream=True, timeout=600, json={
        "messages": [{"role": "user", "content": question}], "project_id": pid,
        "page_path": f"/projects/{pid}",
    }) as r:
        if r.status_code != 200:
            return {"question": question, "error": f"HTTP {r.status_code}: {r.text[:200]}",
                    "seconds": round(time.perf_counter() - t0, 1)}
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            ev = json.loads(raw[5:].strip())
            if first is None and ev.get("type") in ("tool_call", "final"):
                first = round(time.perf_counter() - t0, 1)
            events.append(ev)
    types = [e.get("type") for e in events]
    final = next((e for e in events if e.get("type") == "final"), {})
    err = next((e for e in events if e.get("type") == "error"), {})
    return {
        "question": question,
        "seconds": round(time.perf_counter() - t0, 1),
        "first_action_seconds": first,
        "tool_calls": [e.get("name") for e in events if e.get("type") == "tool_call"],
        "ui_events": [(e.get("event") or {}).get("type") or (e.get("event") or {}).get("kind") for e in events if e.get("type") == "ui"],
        "has_final": bool(final), "forced_final": bool(final.get("forced_final")), "fallback": bool(final.get("fallback")),
        "error": err.get("error"),
        "answer": str(final.get("reply") or "")[:700],
        "event_types": sorted({str(t) for t in types}),
    }


def preference(pid: str, max_pairs: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/api/projects/{pid}/data-prep/preference", timeout=1800,
                      json={"kinds": ["hallucination", "abstain"], "max_pairs": max_pairs, "seed": 42})
    seconds = round(time.perf_counter() - t0, 1)
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:300]}
    keep = {k: body.get(k) for k in ("kept", "counts", "dropped", "drop_reasons", "length_ratio", "total", "pairs",
                                     "error", "code") if k in body}
    return {"status": r.status_code, "seconds": seconds, **keep, "keys": sorted(body)[:20]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--pid", default="")
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--skip-pref", action="store_true")
    args = ap.parse_args()
    pid = args.pid
    if not pid:
        state = REPO / ".tmp" / "manual-e2e" / "state.json"
        pid = json.loads(state.read_text()).get("project_id", "") if state.exists() else ""
    if not pid:
        print("no project: pass --pid or run the walkthrough create/upload/prep/review phases first", file=sys.stderr)
        return 2
    seat = requests.get(f"{BASE}/api/settings/helper", timeout=10).json()
    result: dict[str, Any] = {"label": args.label, "seat": seat["seat_label"], "project": pid, "guide": []}
    for q in QUESTIONS:
        row = guide_turn(pid, q)
        result["guide"].append(row)
        print(f"[guide] {row['seconds']:>6}s tools={row.get('tool_calls')} final={row.get('has_final')} "
              f"err={row.get('error')} :: {q[:50]}", flush=True)
    done = [g for g in result["guide"] if g.get("has_final") and not g.get("error")]
    result["guide_summary"] = {
        "answered": f"{len(done)}/{len(QUESTIONS)}",
        "median_seconds": sorted(g["seconds"] for g in result["guide"])[len(QUESTIONS) // 2],
        "total_seconds": round(sum(g["seconds"] for g in result["guide"]), 1),
    }
    if not args.skip_pref:
        result["preference"] = preference(pid, args.max_pairs)
        print(f"[pref] {result['preference']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.label}.json").write_text(json.dumps(result, indent=2))
    print("summary:", json.dumps(result["guide_summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
