"""Pair-by-pair human review helper: show each pair with the chunk it came from, record a verdict per pair.

    .venv/bin/python scripts/corpus_review.py dump  --pid P --file NAME [--offset 0] [--limit 25] [--status pending]
    .venv/bin/python scripts/corpus_review.py apply --pid P --ledger .tmp/review-ledger.jsonl <<'V'
    <pair-id> A                      # approve
    <pair-id> R wrong number         # reject + reason
    <pair-id> E new question text || new answer text    # edit wording, then approve
    V
    .venv/bin/python scripts/corpus_review.py add --pid P --file NAME --chunk N --ledger L <<'V'
    reviewer-written question || reviewer-written answer      # a pair the miner never wrote (origin=human_review, approved)
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


def full_chunk(pid: str, pair: dict) -> str:
    """The whole chunk from disk (the pair stores only its first 1,500 characters), falling back to the stored text."""
    try:
        from finetune_studio.data.prep.ingest import load_existing_chunks

        chunks = load_existing_chunks(pid, str(pair.get("sha256") or ""))
        idx = int(pair.get("chunk_idx") or 0)
        if 1 <= idx <= len(chunks) and chunks[idx - 1]:
            return chunks[idx - 1]
    except Exception:  # noqa: BLE001 — a missing chunk file must not stop a review session
        pass
    return str(pair.get("chunk_text", ""))


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
    ap.add_argument("cmd", choices=["dump", "apply", "add", "status"])
    ap.add_argument("--pid", required=True)
    ap.add_argument("--file")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--status", default="pending")
    ap.add_argument("--ledger", default=".tmp/review-ledger.jsonl")
    ap.add_argument("--chunk", type=int, default=1)
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
                print(f"\n=== CHUNK {last_chunk} ===\n{full_chunk(a.pid, p)}\n")
            print(f"[{p['id']}] c{p.get('chunk_idx')}\n  Q: {p.get('question')}\n  A: {p.get('answer')}")
        return 0
    if a.cmd == "apply":
        by_id = {p["id"]: p for p in pairs}
        ledger.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with ledger.open("a") as fh:
            for line in sys.stdin.read().splitlines():
                # a trailing " # note" is a comment, except inside an edit ("question || answer") where # is text
                line = (line if "||" in line else line.split(" #")[0]).strip()
                if not line:
                    continue
                parts = line.split(None, 2)
                pid_, verdict = parts[0], parts[1].upper()[:1]
                reason = parts[2] if len(parts) > 2 else ""
                if pid_ not in by_id or verdict not in "ARE":
                    print(f"SKIP bad line: {line}", file=sys.stderr)
                    continue
                if verdict == "E":
                    q, _, ans = reason.partition("||")
                    if not q.strip() or not ans.strip():
                        print(f"SKIP edit without 'question || answer': {line}", file=sys.stderr)
                        continue
                    call(f"/api/projects/{a.pid}/data-prep/qa/{pid_}", "PATCH",
                         {"question": q.strip(), "answer": ans.strip(), "status": "approved"})
                    reason = f"edited: {reason}"
                else:
                    call(f"/api/projects/{a.pid}/data-prep/qa/{pid_}", "PATCH",
                         {"status": "approved" if verdict == "A" else "rejected"})
                fh.write(json.dumps({"id": pid_, "verdict": "A" if verdict == "E" else verdict, "reason": reason, "file": sources.get(by_id[pid_].get("source_id")),
                                     "ts": time.time()}) + "\n")
                n += 1
        print(f"recorded {n} verdicts")
        return 0
    if a.cmd == "add":
        from finetune_studio.data import project_filesystem as pfs
        src = next((sid for sid, name in sources.items() if name == a.file), None)
        if not src:
            print(f"no source named {a.file}", file=sys.stderr)
            return 2
        meta = pfs.read_qa_source(a.pid, src)
        chunk_text = next((p.get("chunk_text", "") for p in pairs if p.get("source_id") == src and p.get("chunk_idx") == a.chunk
                           and p.get("chunk_text")), "")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with ledger.open("a") as fh:
            for line in sys.stdin.read().splitlines():
                q, sep, ans = line.partition("||")
                if not sep or not q.strip() or not ans.strip():
                    continue
                import uuid
                qa_id = uuid.uuid4().hex[:12]
                now = time.time()
                pfs.write_qa_pair(a.pid, {
                    "id": qa_id, "source_id": src, "sha256": meta.get("sha256", ""), "chunk_idx": a.chunk, "chunk_text": chunk_text,
                    "question": q.strip(), "answer": ans.strip(), "difficulty": "medium", "style": "factual", "score": 1.0,
                    "status": "approved", "origin": "human_review", "created_at": now, "updated_at": now,
                    "provenance": {"source_id": src, "filename": a.file, "chunk_idx": a.chunk, "generator": "human-review"},
                    "validation": {"accepted": True, "version": "human", "reasons": []},
                })
                fh.write(json.dumps({"id": qa_id, "verdict": "A", "reason": "added by reviewer", "file": a.file, "ts": now}) + "\n")
                n += 1
        print(f"added {n} reviewer-written pairs")
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
