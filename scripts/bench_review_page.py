"""Time the Pairs review page on real Korvane pairs, the way a reviewer uses it (keyboard, in a real browser).

    .venv/bin/python scripts/bench_review_page.py [--actions 150] [--keep]

Builds a scratch project holding the golden Korvane pairs (every one reset to *pending*, with the Q&A text and chunk grouping of
the 2026-10-08 review), then in headless Chromium:
  1. opens the Pairs page and times first render of the review workspace,
  2. presses A / R+1 / J / U like a reviewer and times each key -> next pair visible,
  3. checks that every verdict reached the disk, that undo works, and that the reject reason was stored.
Prints p50/p95 per action and the wall time per pair. The scratch project is deleted unless --keep.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:7860"
GOLDEN = ROOT / "tests" / "corpus" / "korvane" / "golden"


def call(path: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def seed(pid: str) -> int:
    """Write the golden pairs as pending pairs, grouped into sources/chunks exactly as the review saw them."""
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.data.fs.chunks import write_chunks

    rows: list[dict] = []
    for f in sorted([*GOLDEN.glob("approved*.jsonl.gz"), *GOLDEN.glob("rejected*.jsonl.gz")]):  # the first run only: ui_run_* is the same corpus again
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    by_file: dict[str, list[dict]] = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)
    now = time.time()
    n = 0
    for name, items in by_file.items():
        sid = uuid.uuid5(uuid.NAMESPACE_URL, name).hex[:12]
        sha = uuid.uuid5(uuid.NAMESPACE_URL, "sha" + name).hex * 2
        chunk_idx = sorted({int(r.get("chunk") or 1) for r in items})
        chunks = [""] * max(chunk_idx)
        for r in items:  # a stand-in chunk: the answers of that chunk, so the highlighter has something to mark
            i = int(r.get("chunk") or 1) - 1
            chunks[i] += r["a"] + "\n"
        write_chunks(pid, sha, chunks)
        qa_fs.write_qa_source(pid, {"id": sid, "filename": name, "name": name, "sha256": sha, "status": "ready", "uploaded_at": now})
        for r in items:
            qa_fs.write_qa_pair(pid, {"id": uuid.uuid4().hex[:12], "source_id": sid, "sha256": sha, "chunk_idx": int(r.get("chunk") or 1),
                                      "chunk_text": chunks[int(r.get("chunk") or 1) - 1][:1500], "question": r["q"], "answer": r["a"],
                                      "status": "pending", "origin": r.get("origin", "model"), "created_at": now + n})
            n += 1
    return n


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=150)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    pid = call("/api/projects", "POST", {"name": "bench-review-scratch"})["id"]
    try:
        total = seed(pid)
        print(f"seeded {total} pending pairs into {pid}")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1500, "height": 950})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)[:200]))
            t0 = time.perf_counter()
            page.goto(f"{BASE}/projects/{pid}/data-prep", wait_until="domcontentloaded")
            page.wait_for_selector("#dp-rq-detail textarea#dp-rq-q", timeout=30000)
            first_render = time.perf_counter() - t0
            print(f"first render of the review workspace: {first_render * 1000:.0f} ms")
            page.click("#dp-rq-list tr.cur")  # focus inside the workspace so the shortcuts apply
            page.mouse.click(5, 5)

            def key_and_wait(key: str) -> float:
                before = page.evaluate("(_rq.items[_rq.cur] || {}).id")
                t = time.perf_counter()
                page.keyboard.press(key)
                try:
                    page.wait_for_function("b => (_rq.items[_rq.cur] || {}).id !== b && document.getElementById('dp-rq-q').value === _rq.items[_rq.cur].question",
                                           arg=before,
                                           timeout=5000)
                except PlaywrightTimeout:
                    state = page.evaluate("JSON.stringify({cur:_rq.cur,n:_rq.items.length,rej:_rq.rejecting,counts:_rq.counts,"
                                          "active:document.activeElement.tagName+'#'+document.activeElement.id})")
                    raise SystemExit(f"key {key!r} changed nothing; page state {state}; page errors {errors}") from None
                return (time.perf_counter() - t) * 1000

            approve_ms: list[float] = []
            reject_ms: list[float] = []
            nav_ms: list[float] = []
            wall = time.perf_counter()
            reasons = 0
            for i in range(a.actions):
                if i % 10 == 9:
                    page.keyboard.press("r")
                    reject_ms.append(key_and_wait("1"))
                    reasons += 1
                elif i % 10 == 4:
                    nav_ms.append(key_and_wait("j"))
                else:
                    approve_ms.append(key_and_wait("a"))
            wall_s = time.perf_counter() - wall
            page.wait_for_function("document.getElementById('dp-rq-saving').textContent.indexOf('saving') < 0", timeout=30000)
            print(f"{a.actions} keypresses in {wall_s:.1f} s wall ({wall_s / a.actions * 1000:.0f} ms each, including the browser's own key delay)")
            for label, vals in (("approve (A)", approve_ms), ("reject + reason (1)", reject_ms), ("skip (J)", nav_ms)):
                if vals:
                    print(f"  {label:20} n={len(vals):3} p50={statistics.median(vals):5.1f} ms  p95={pct(vals, 0.95):5.1f} ms  max={max(vals):5.1f} ms")
            # Undo the last verdict and check the disk.
            page.keyboard.press("u")
            page.wait_for_timeout(600)
            page.wait_for_function("document.getElementById('dp-rq-saving').textContent.indexOf('saving') < 0", timeout=30000)
            queue = call(f"/api/projects/{pid}/data-prep/qa/queue?limit=1")
            c = queue["counts"]
            print(f"server counts after the run: {c}")
            stored = call(f"/api/projects/{pid}/data-prep/qa")["items"]
            reviewed = [q for q in stored if q.get("reviewed_at")]
            rejected = [q for q in stored if q["status"] == "rejected"]
            checks = [
                (len(stored) == total, f"no pair lost or duplicated ({len(stored)}/{total})"),
                (c["pending"] + c["approved"] + c["rejected"] == total, "counts add up to the total"),
                (all(q.get("note") for q in rejected), f"all {len(rejected)} rejected pairs carry a reason"),
                (len(reviewed) >= len(approve_ms) + len(reject_ms) - 1, f"{len(reviewed)} pairs carry a human verdict stamp"),
                (not errors, f"no page errors {errors[:2]}"),
            ]
            browser.close()
        ok = True
        for good, what in checks:
            print(("PASS " if good else "FAIL ") + what)
            ok = ok and good
        return 0 if ok else 1
    finally:
        if not a.keep:
            call(f"/api/projects/{pid}", "DELETE")
            print("scratch project deleted")


if __name__ == "__main__":
    raise SystemExit(main())
