#!/usr/bin/env python3
"""Compare RAG *readers* (models) and context caps on the same indexed project.

Per reader x context-cap this starts a RAG-suite test run (``POST /api/testing/run-rag-suite``, which only records
transcripts), polls it to the end, has an AI judge read the saved answers (``POST .../runs/{id}/judge``), saves the
full case list and prints where every miss comes from:

* ``not retrieved``  - the expected values are not in the top-k chunks (retrieval problem),
* ``cut by cap``     - retrieved, but the context cap dropped the chunk before the model saw it,
* ``reader missed``  - the model saw the values and still did not pass (reader problem),
* ``wrongly answered`` - an unanswerable question that got a confident answer,
* ``awaiting judgement`` - no verdict yet (the judge failed on the case, or ``--no-judge``).

    .venv/bin/python scripts/rag_reader_compare.py --pid e9f951f8 \
        --reader base-q4=/path/base.gguf --reader gemma12=/path/gemma.gguf --cap 5000 --cap 16000 \
        [--judge-provider PROVIDER_ID]

``--judge-provider`` is a provider row id; empty (default) = the judge configured on the Testing page, else the
helper seat. ``--no-judge`` only records transcripts (the miss split then shows everything as awaiting judgement).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
QUIZ = ROOT / "tests" / "corpus" / "korvane" / "eval" / "korvane_quiz_core.jsonl"
RUN_TERMINAL = {"done", "failed", "cancelled"}
JUDGE_TERMINAL = {"done", "failed", "cancelled"}
POLL_SECONDS = 5.0
BUSY_RETRY_SECONDS = 15.0


class ApiError(RuntimeError):
    """The app answered with an error status; ``status`` and the server's reason are kept."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def _call(base: str, method: str, path: str, body: dict | None = None, *, timeout: float = 120) -> Any:
    req = urllib.request.Request(
        f"{base}{path}", method=method, headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            data = json.loads(raw)
            raw = str(data.get("error") or data.get("detail") or raw)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise ApiError(exc.code, raw) from None


def _post_when_free(base: str, path: str, body: dict) -> Any:
    """POST, waiting while the app is busy with another test job (409) instead of failing."""
    while True:
        try:
            return _call(base, "POST", path, body)
        except ApiError as exc:
            if exc.status != 409:
                raise
            print(f"  app busy ({exc}); retrying in {BUSY_RETRY_SECONDS:.0f}s", flush=True)
            time.sleep(BUSY_RETRY_SECONDS)


def _poll(base: str, pid: str, bid: str, *, done_key: str, total_key: str, status_key: str, terminal: set[str], label: str) -> dict:
    """Poll the run row until ``status_key`` is terminal, printing progress whenever it moves."""
    last = None
    while True:
        run = _call(base, "GET", f"/api/testing/projects/{pid}/runs/{bid}")
        progress = (run.get(done_key, 0), run.get(total_key, 0))
        if progress != last:
            print(f"  {label}: {progress[0]}/{progress[1]}", flush=True)
            last = progress
        if run.get(status_key) in terminal:
            return run
        time.sleep(POLL_SECONDS)


def run_suite(base: str, pid: str, model_path: str, *, top_k: int, cap: int | None, quiz: Path) -> dict:
    """One RAG-suite run through the app (loads ``model_path`` if it is not the loaded one); blocks until it ends.

    ``cap`` None = let the app size the context from the loaded model's window. Returns the final run row."""
    body: dict[str, Any] = {"project_id": pid, "suite_path": str(quiz), "model_path": model_path, "top_k": top_k,
                            "max_tokens": 512, "temperature": 0.0, "auto_judge": False}
    if cap:
        body["max_context_chars"] = cap
    started = _post_when_free(base, "/api/testing/run-rag-suite", body)
    run = _poll(base, pid, started["benchmark_id"], done_key="progress_done", total_key="progress_total",
                status_key="status", terminal=RUN_TERMINAL, label="answered")
    if run["status"] != "done":
        raise RuntimeError(f"test run {run['id']} ended {run['status']}: {run.get('error') or 'no reason recorded'}")
    return run


def judge_run(base: str, pid: str, bid: str, provider_id: str) -> dict:
    """Judge every case of a finished run with ``provider_id`` ('' = configured default); blocks until it ends."""
    _post_when_free(base, f"/api/testing/projects/{pid}/runs/{bid}/judge", {"provider_id": provider_id, "only_unjudged": True})
    run = _poll(base, pid, bid, done_key="judge_done", total_key="judge_total", status_key="judge_status",
                terminal=JUDGE_TERMINAL, label="judged")
    if run["judge_status"] != "done":
        print(f"  WARNING judge ended {run['judge_status']}: {run.get('judge_error') or 'no reason recorded'}", flush=True)
    return run


def fetch_cases(base: str, pid: str, bid: str) -> list[dict]:
    return _call(base, "GET", f"/api/testing/projects/{pid}/runs/{bid}/cases")


def split_misses(cases: list[dict]) -> Counter:
    """Classify every case by where the answer was lost; a case without a verdict is awaiting judgement, never a miss."""
    out: Counter = Counter()
    for c in cases:
        info = c.get("judge_input") or {}
        if not c.get("verdict"):
            out["awaiting judgement"] += 1
        elif c["verdict"] == "pass":
            out["pass " + c["category"]] += 1
        elif c["category"] == "unanswerable":
            out["wrongly answered"] += 1
        elif not info.get("gold_in_retrieved"):
            out["not retrieved"] += 1
        elif not info.get("gold_in_context"):
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
    ap.add_argument("--judge-provider", default="", metavar="PROVIDER_ID",
                    help="provider row that judges the saved answers ('' = configured default judge, else helper seat)")
    ap.add_argument("--no-judge", action="store_true", help="only record transcripts; skip the judge step")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.reader:
        name, _, path = spec.partition("=")
        for cap in args.cap or [None]:   # no --cap: the app derives it from the model's window
            run = run_suite(args.base, args.pid, path, top_k=args.top_k, cap=cap, quiz=args.quiz)
            if not args.no_judge:
                run = judge_run(args.base, args.pid, run["id"], args.judge_provider)
            cases = fetch_cases(args.base, args.pid, run["id"])
            (args.out / f"{name}-cap{cap or 'auto'}-k{args.top_k}.json").write_text(
                json.dumps({"run": run, "results": cases}, indent=1, ensure_ascii=False))
            counts = split_misses(cases)
            retrieval = (run.get("scores") or {}).get("retrieval") or {}
            judge_note = f" | judge {run.get('judge_model') or 'none'}"
            print(f"{name} cap={cap or 'auto'} k={args.top_k}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                  f"| chunks in context avg {retrieval.get('mean_chunks_in_context', '?')}" + judge_note, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
