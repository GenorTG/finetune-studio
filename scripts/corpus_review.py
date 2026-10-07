"""Pair-by-pair human review helper: show each pair with the chunk it came from, record a verdict per pair.

    .venv/bin/python scripts/corpus_review.py dump  --pid P --file NAME [--offset 0] [--limit 25] [--status pending]
    .venv/bin/python scripts/corpus_review.py apply --pid P --ledger .tmp/review-ledger.jsonl <<'V'
    <pair-id> A                      # approve
    <pair-id> R wrong number         # reject + reason
    V
    .venv/bin/python scripts/corpus_review.py status --pid P --ledger .tmp/review-ledger.jsonl

``apply`` PATCHes each pair through the app API and appends ``{id, verdict, reason, file, ts}`` to the ledger, so a review is
auditable and resumable: ``dump`` skips pairs that already have a verdict in the ledger. A pair never gets approved unless a
verdict line names it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7860"


def call(path: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def ledger_ids(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["dump", "apply", "status"])
    ap.add_argument("--pid", required=True)
    ap.add_argument("--file")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--status", default="pending")
    ap.add_argument("--ledger", default=".tmp/review-ledger.jsonl")
    ap.add_argument("--chunks", action="store_true", help="print the full source chunk once per chunk")
    a = ap.parse_args()
    ledger = Path(a.ledger)
    sources = {s["id"]: s["filename"] for s in call(f"/api/projects/{a.pid}/data-prep/sources")["sources"]}
    pairs = call(f"/api/projects/{a.pid}/data-prep/qa")["items"]
    done = ledger_ids(ledger)
    if a.cmd == "dump":
        mine = [p for p in pairs if sources.get(p.get("source_id")) == a.file and p.get("status") == a.status and p["id"] not in done]
        mine.sort(key=lambda p: (p.get("chunk_idx", 0), p["id"]))
        print(f"# {a.file}: {len(mine)} unreviewed {a.status} pairs; showing {a.offset}..{a.offset + a.limit}")
        last_chunk = None
        for p in mine[a.offset:a.offset + a.limit]:
            if a.chunks and p.get("chunk_idx") != last_chunk:
                last_chunk = p.get("chunk_idx")
                print(f"\n=== CHUNK {last_chunk} ===\n{p.get('chunk_text', '')}\n")
            print(f"[{p['id']}] c{p.get('chunk_idx')}\n  Q: {p.get('question')}\n  A: {p.get('answer')}")
        return 0
    if a.cmd == "apply":
        by_id = {p["id"]: p for p in pairs}
        ledger.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with ledger.open("a") as fh:
            for line in sys.stdin.read().splitlines():
                line = line.split("#")[0].strip()
                if not line:
                    continue
                parts = line.split(None, 2)
                pid_, verdict = parts[0], parts[1].upper()[:1]
                reason = parts[2] if len(parts) > 2 else ""
                if pid_ not in by_id or verdict not in "AR":
                    print(f"SKIP bad line: {line}", file=sys.stderr)
                    continue
                call(f"/api/projects/{a.pid}/data-prep/qa/{pid_}", "PATCH", {"status": "approved" if verdict == "A" else "rejected"})
                fh.write(json.dumps({"id": pid_, "verdict": verdict, "reason": reason, "file": sources.get(by_id[pid_].get("source_id")),
                                     "ts": time.time()}) + "\n")
                n += 1
        print(f"recorded {n} verdicts")
        return 0
    per: dict[str, list[int]] = {}
    for p in pairs:
        f = sources.get(p.get("source_id"), "?")
        row = per.setdefault(f, [0, 0, 0])
        row[0] += 1
        if p["id"] in done:
            row[1 if done[p["id"]]["verdict"] == "A" else 2] += 1
    print(f"{'file':56} pairs  approved  rejected  unreviewed")
    for f, (t, ap_, rj) in sorted(per.items()):
        print(f"{f:56} {t:5} {ap_:8} {rj:9} {t - ap_ - rj:10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
