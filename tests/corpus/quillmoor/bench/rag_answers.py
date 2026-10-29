"""RAG answers from many answer models over ONE exported Smart Brain package, retrieving once per question.

    .venv/bin/python tests/corpus/quillmoor/bench/rag_answers.py --corpus .tmp/quillmoor/s --config qwen3emb-hybrid-bge \
        --package .tmp/bench/packages/<emb>__<rr> --flags --reranker \
        --model qwen35-4b-base=~/.finetune-studio/bench-gguf/Qwen__Qwen3.5-4B/model-q8_0.gguf ...

What the package's /ask does is reproduced exactly, but its retrieval is done once: the package's server is started
from its own server.py and asked /search (its defaults: top_k, dense_extra, reranker), the passages are cached, and
each answer model gets the messages /ask would send, built with the same functions (format_context_blocks,
build_system_prompt), temperature 0.2, the same token limit. export_combos.py checks /ask itself end to end.
Answers: <corpus>/results/answers/<label>__rag-<config>.jsonl, judged in place. The local inference context is
configurable per slot so answer prompts plus the package's answer-token allowance fit in the llama-server context.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answers import done_ids
from brain_matrix import start_server, stop
from common import ROOT, gpu_serialized, load_jsonl, write_json
from evaluate import subset
from judge import judge_files, score
from llm import CLOSED_SYSTEM, chat, llama_server

from finetune_studio.smart_brain.standalone_server import (
    DEFAULT_CONFIG as DEFAULTS,
)
from finetune_studio.smart_brain.standalone_server import (
    build_system_prompt,
    format_context_blocks,
    strip_thinking,
)


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], **gen: Any) -> str: ...


def retrieve(corpus: Path, config: str, package: Path, flags: list[str], qs: list[dict]) -> dict[str, list[dict]]:
    cache = corpus / "results" / "rag_cache" / f"{config}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pkg = next(p for p in package.iterdir() if p.is_dir()) if not (package / "server.py").exists() else package
    p, base = start_server(pkg, flags, ROOT / ".tmp" / "bench" / f"server-rag-{config}.log")
    out: dict[str, list[dict]] = {}
    try:
        info = httpx.get(f"{base}/health", timeout=10).json()
        t0 = time.time()
        for q in qs:
            r = httpx.post(f"{base}/search", json={"query": q["question"]}, timeout=300)   # /ask's own search call
            r.raise_for_status()
            out[q["id"]] = r.json()["hits"]
        print(f"[{config}] retrieved {len(qs)} questions in {time.time() - t0:.0f}s ({info.get('mode')})", flush=True)
    finally:
        stop(p)
    write_json(cache, {"_mode": info.get("mode"), **out})
    return out


def answer_all(qs: list[dict], hits: dict[str, list[dict]] | None, out: Path, *, base_url: str, model: str,
               api_key: str, label: str, workers: int, client: ChatClient | None = None,
               server_health_url: str | None = None) -> None:
    have = done_ids(out)
    lock = threading.Lock()
    server_dead = threading.Event()

    def one(q: dict) -> None:
        h = hits.get(q["id"], []) if hits is not None else []
        t0, err, ans = time.time(), "", ""
        if hits is not None and not h:
            ans = "Nothing in this corpus matches the question."
        else:
            msgs = ([{"role": "system", "content": build_system_prompt(format_context_blocks(h))}] if hits is not None
                    else [{"role": "system", "content": CLOSED_SYSTEM}]) + [{"role": "user", "content": q["question"]}]
            for attempt in range(3):
                if server_dead.is_set():
                    err = "local inference server unavailable after an earlier health-check failure"
                    break
                try:
                    if client is not None:   # an app API connection: its key stays in the app's vault
                        ans = strip_thinking(client.chat(msgs, max_tokens=int(DEFAULTS["answer_max_tokens"]), temperature=0.2,
                                                         timeout=180))
                    else:
                        ans = strip_thinking(chat(base_url, msgs, model=model, api_key=api_key,
                                                  max_tokens=int(DEFAULTS["answer_max_tokens"]), extra={"temperature": 0.2}))
                    if not ans.strip():
                        raise RuntimeError("model returned an empty answer")
                    err = ""
                    break
                except Exception as e:  # noqa: BLE001 - recorded; the judge skips errored rows
                    err = f"{type(e).__name__}: {e}"
                    if server_health_url is not None and client is None:
                        try:
                            healthy = httpx.get(server_health_url, timeout=2).status_code == 200
                        except httpx.HTTPError:
                            healthy = False
                        if not healthy:
                            server_dead.set()
                            err += "; local llama-server health check failed"
                            break
                    time.sleep(3 * (attempt + 1))
        rec = {"id": q["id"], "type": q["type"], "question": q["question"], "key": q["answer"], "accept": q["accept"],
               "wrong": q["wrong"], "formats": q["formats"], "etype": q["etype"], "attr": q["attr"], "answer": ans,
               "sources": [x.get("filename") for x in h], "error": err, "latency": round(time.time() - t0, 3),
               "label": label, "mode": "rag" if hits is not None else "closed"}
        with lock, open(out, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    todo = [q for q in qs if q["id"] not in have]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))


@gpu_serialized
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--config", default="", help="short name of the retrieval setup (file names)")
    ap.add_argument("--package", default="", help="an extracted package (its dir or the dir holding it)")
    ap.add_argument("--flags", nargs=argparse.REMAINDER, default=[], help="server flags, last on the line")
    ap.add_argument("--model", action="append", default=[], help="label=path.gguf (repeatable)")
    ap.add_argument("--api", action="append", default=[], help="label=base_url|model|keyfile (repeatable)")
    ap.add_argument("--connection", action="append", default=[],
                    help="label=<Settings -> AI assistant connection id>, e.g. deepseek-flash=api-helper (repeatable)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ctx-per-slot", type=int, default=8192,
                    help="llama-server prompt context per parallel slot (default: 8192)")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke tests)")
    ap.add_argument("--closed", action="store_true", help="no retrieval: the --connection models answer alone")
    a = ap.parse_args()
    corpus = Path(a.corpus)
    qs = subset(load_jsonl(corpus / "eval" / "questions.jsonl"), 140, 60)
    qs = qs[: a.limit] if a.limit else qs
    out_dir = corpus / "results" / "answers"
    if a.closed:   # no Smart Brain: the answer model alone, same questions and judge
        from finetune_studio.models.assistant import connection_client
        for spec in a.connection:
            label, _, conn = spec.partition("=")
            f = out_dir / f"{label}__closed.jsonl"
            answer_all(qs, None, f, base_url="", model="", api_key="", label=label, workers=8, client=connection_client(conn))
            print(judge_files([f]), flush=True)
            print(f.stem, json.dumps(score(load_jsonl(f))), flush=True)
        return 0
    hits = retrieve(corpus, a.config, Path(a.package).expanduser(), a.flags, qs)
    hits.pop("_mode", None)
    for spec in a.model:
        label, _, path = spec.partition("=")
        f = out_dir / f"{label}__rag-{a.config}.jsonl"
        t0 = time.time()
        with llama_server(Path(path).expanduser(), ctx_per_slot=a.ctx_per_slot, parallel=a.workers,
                          log=ROOT / ".tmp" / "bench" / f"llama-{label}-rag.log") as url:
            health_url = f"{url.removesuffix('/v1')}/health"
            answer_all(qs, hits, f, base_url=url, model="local", api_key="", label=label, workers=a.workers,
                       server_health_url=health_url)
        print(f"[{label}] {len(qs)} RAG answers in {time.time() - t0:.0f}s", flush=True)
        print(judge_files([f]), flush=True)
        print(f.stem, json.dumps(score(load_jsonl(f))), flush=True)
    for spec in a.api:
        label, _, rest = spec.partition("=")
        url, model, keyfile = (rest.split("|") + ["", ""])[:3]
        f = out_dir / f"{label}__rag-{a.config}.jsonl"
        key = Path(keyfile).read_text().strip() if keyfile else ""
        answer_all(qs, hits, f, base_url=url, model=model, api_key=key, label=label, workers=8)
        print(judge_files([f]), flush=True)
        print(f.stem, json.dumps(score(load_jsonl(f))), flush=True)
    for spec in a.connection:
        from finetune_studio.models.assistant import connection_client
        label, _, conn = spec.partition("=")
        f = out_dir / f"{label}__rag-{a.config}.jsonl"
        answer_all(qs, hits, f, base_url="", model="", api_key="", label=label, workers=8, client=connection_client(conn))
        print(judge_files([f]), flush=True)
        print(f.stem, json.dumps(score(load_jsonl(f))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
