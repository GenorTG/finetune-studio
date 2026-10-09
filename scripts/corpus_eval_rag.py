"""Track B (RAG) quiz: the SAME questions as ``corpus_eval.py``, answered through the project's RAG chat.

    .venv/bin/python scripts/corpus_eval_rag.py --pid <project> [--top-k 10] [--json out.json]

Uses whatever model is loaded (the base model and the fine-tuned one should both do well: retrieval, not training, carries the
facts). Scores two things separately: RETRIEVAL (does any returned chunk contain every expected value) and the ANSWER (does the
reply contain every expected value / does it abstain on unanswerable questions). Retrieval-only misses are index problems, answer
misses on retrieved facts are generator problems.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_build import ROOT
from corpus_coverage import BASE, norm
from corpus_eval import ABSTAIN, passes, variants


def rag_chat(pid: str, question: str, top_k: int, max_tokens: int = 200) -> tuple[str, list[dict]]:
    req = urllib.request.Request(f"{BASE}/api/projects/{pid}/rag/chat", method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"messages": [{"role": "user", "content": question}], "top_k": top_k,
                                                  "max_tokens": max_tokens, "temperature": 0.0}).encode())
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return str(d.get("reply", "")), list(d.get("hits") or [])


def retrieved(hits: list[dict], expects: list[str]) -> bool:
    blob = norm(" ".join(str(h.get("text", "")) for h in hits))
    blob2 = blob.replace(",", "")
    return all(any(v in blob or v in blob2 for v in variants(x)) for x in expects)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--set", default="paraphrase_core")
    ap.add_argument("--unanswerable", default="unanswerable_core")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args()
    rows = [json.loads(ln) for ln in (ROOT / "eval" / f"{a.set}.jsonl").read_text().splitlines() if ln.strip()]
    unans = [json.loads(ln) for ln in (ROOT / "eval" / f"{a.unanswerable}.jsonl").read_text().splitlines() if ln.strip()]
    if a.limit:
        rows, unans = rows[:a.limit], unans[:max(1, a.limit // 5)]
    results = []
    for r in rows:
        reply, hits = rag_chat(a.pid, r["q"], a.top_k)
        res = {**r, "reply": reply, "ok": passes(reply, r["expect"]), "retrieved": retrieved(hits, r["expect"]),
               "files": sorted({h.get("filename", "") for h in hits})}
        results.append(res)
        print(f"{'PASS' if res['ok'] else 'FAIL'} {'RET ' if res['retrieved'] else 'noRET'} {r['id']} {r['q'][:60]} -> {reply[:80]!r}", flush=True)
    abst = []
    for u in unans:
        reply, hits = rag_chat(a.pid, u["q"], a.top_k)
        ok = any(m in reply.lower() for m in ABSTAIN)
        abst.append({**u, "reply": reply, "ok": ok})
        print(f"{'ABSTAIN' if ok else 'INVENTED'} {u['id']} {u['q'][:60]} -> {reply[:90]!r}", flush=True)
    n = max(len(results), 1)
    rec, ret = sum(r["ok"] for r in results), sum(r["retrieved"] for r in results)
    ab = sum(r["ok"] for r in abst)
    print(f"RETRIEVAL@{a.top_k} {ret}/{len(results)} = {100 * ret / n:.1f}%   ANSWER RECALL {rec}/{len(results)} = {100 * rec / n:.1f}%   "
          f"ABSTAINS ON UNKNOWN {ab}/{len(abst)} = {100 * ab / max(len(abst), 1):.1f}%")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"answerable": results, "unanswerable": abst}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
