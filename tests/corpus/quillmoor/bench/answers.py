"""Collect model answers for a question set (run-then-judge: this records, judge.py grades).

Modes:
  closed  -- the bare model, no documents (what a fine-tuned adapter has to carry on its own)
  ask     -- a Smart Brain package server's /ask: retrieval + the answer layer (its own prompt, its own LLM endpoint)
Answers append to a JSONL file; a re-run skips ids already answered (resumable).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from llm import CLOSED_SYSTEM, chat


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["id"]
        for line in path.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if not row.get("error") and str(row.get("answer") or "").strip()
    }


def collect(questions: list[dict], out: Path, *, mode: str, base_url: str, model: str = "local", api_key: str = "",
            workers: int = 4, max_tokens: int = 256, label: str = "", extra: dict | None = None) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    have = done_ids(out)
    todo = [q for q in questions if q["id"] not in have]
    todo_ids = {str(q["id"]) for q in todo}
    if todo_ids and out.is_file():
        old_rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        kept_rows = [row for row in old_rows if str(row.get("id")) not in todo_ids]
        if len(kept_rows) != len(old_rows):
            temp = out.with_suffix(out.suffix + ".tmp")
            temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept_rows))
            temp.replace(out)
    lock = threading.Lock()
    stopped = threading.Event()
    connection_errors: list[str] = []

    def one(q: dict) -> None:
        if stopped.is_set():
            return
        t0 = time.time()
        err, ans, sources = "", "", []
        for attempt in range(3):
            try:
                if mode == "closed":
                    ans = chat(base_url, [{"role": "system", "content": CLOSED_SYSTEM}, {"role": "user", "content": q["question"]}],
                               model=model, max_tokens=max_tokens, api_key=api_key, extra=extra)
                else:
                    r = httpx.post(f"{base_url}/ask", json={"question": q["question"]}, timeout=600)
                    r.raise_for_status()
                    j = r.json()
                    ans, sources = j.get("answer") or "", [s.get("filename") for s in j.get("sources") or []]
                if not ans.strip():
                    raise RuntimeError("model returned an empty answer")
                err = ""
                break
            except httpx.ConnectError as e:
                err = f"{type(e).__name__}: {e}"
                with lock:
                    if not connection_errors:
                        connection_errors.append(err)
                stopped.set()
                break
            except Exception as e:  # noqa: BLE001 - recorded; the judge skips errored rows
                err = f"{type(e).__name__}: {e}"
                time.sleep(3 * (attempt + 1))
        rec = {"id": q["id"], "type": q["type"], "question": q["question"], "key": q["answer"], "accept": q["accept"],
               "wrong": q["wrong"], "formats": q["formats"], "etype": q["etype"], "attr": q["attr"], "answer": ans,
               "sources": sources, "error": err, "latency": round(time.time() - t0, 3), "label": label, "mode": mode}
        if "fact" in q:
            rec["fact"] = q["fact"]
        with lock, open(out, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    if connection_errors:
        raise RuntimeError(f"answer endpoint became unreachable: {connection_errors[0]}")
    return len(todo)
