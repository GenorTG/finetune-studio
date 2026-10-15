"""Quiz the model that is loaded in the app on facts it can only know from training.

    .venv/bin/python scripts/corpus_eval.py [--set paraphrase_core] [--system "..."] [--json out.json]

Asks every question of ``tests/corpus/korvane/eval/<set>.jsonl`` plain (no retrieved context, temperature 0) through
``POST /api/inference/chat`` and scores: an answerable question passes when every `expect` string appears in the reply
(number-aware, digits and small number words interchangeable); an unanswerable one passes when the model abstains instead of
inventing a value. Reports recall, abstention and a verdict on over-fitting: recall of PARAPHRASED questions is the number
that matters (training-pair recall only proves memorisation of the training wording).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_build import ROOT
from corpus_coverage import BASE, norm

WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ABSTAIN = ("don't know", "do not know", "not know", "no information", "not mentioned", "not specified", "not provided", "cannot",
           "can't", "unable", "no record", "not available", "isn't stated", "is not stated", "not in the", "not covered",
           "i have no", "unknown", "not aware", "does not contain", "doesn't contain", "do not contain", "not contain", "no mention",
           "not found", "isn't something", "is not something", "don't have that", "do not have that", "don't say", "do not say",
           "documents cover", "not say", "doesn't say", "does not say", "not stated", "no data", "not given", "not listed")


def ask(question: str, system: str, max_tokens: int = 160) -> str:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": question})
    req = urllib.request.Request(BASE + "/api/inference/chat", method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
                                                  "top_p": 1.0, "top_k": 1, "repeat_penalty": 1.0}).encode())
    with urllib.request.urlopen(req, timeout=300) as r:
        return str(json.loads(r.read()).get("response", ""))


def variants(expect: str) -> list[str]:
    e = norm(expect)
    out = [e, e.replace(",", "")]
    for w, d in WORDS.items():
        if re.fullmatch(rf"{w}|{d}", e):
            out += [w, d]
    return out


def passes(reply: str, expects: list[str]) -> bool:
    r = norm(reply)
    r2 = r.replace(",", "")
    return all(any(v in r or v in r2 for v in variants(x)) for x in expects)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="paraphrase_core")
    ap.add_argument("--unanswerable", default="unanswerable_core")
    ap.add_argument("--system", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args()
    rows = [json.loads(ln) for ln in (ROOT / "eval" / f"{a.set}.jsonl").read_text().splitlines() if ln.strip()]
    unans = [json.loads(ln) for ln in (ROOT / "eval" / f"{a.unanswerable}.jsonl").read_text().splitlines() if ln.strip()]
    if a.limit:
        rows, unans = rows[:a.limit], unans[:max(1, a.limit // 5)]
    results = []
    for r in rows:
        reply = ask(r["q"], a.system)
        results.append({**r, "reply": reply, "ok": passes(reply, r["expect"])})
        print(f"{'PASS' if results[-1]['ok'] else 'FAIL'} {r['id']} {r['q'][:70]} -> {reply[:100]!r}", flush=True)
    abst = []
    for u in unans:
        reply = ask(u["q"], a.system)
        ok = any(m in reply.lower() for m in ABSTAIN)
        abst.append({**u, "reply": reply, "ok": ok})
        print(f"{'ABSTAIN' if ok else 'INVENTED'} {u['id']} {u['q'][:60]} -> {reply[:90]!r}", flush=True)
    rec = sum(r["ok"] for r in results)
    ab = sum(r["ok"] for r in abst)
    print(f"PARAPHRASE RECALL {rec}/{len(results)} = {100 * rec / max(len(results), 1):.1f}%   "
          f"ABSTAINS ON UNKNOWN {ab}/{len(abst)} = {100 * ab / max(len(abst), 1):.1f}%")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"answerable": results, "unanswerable": abst}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
