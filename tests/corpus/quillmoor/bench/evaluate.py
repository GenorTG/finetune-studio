"""Answer + judge one model GGUF on the Quillmoor questions, closed book and/or through a Smart Brain package.

    .venv/bin/python tests/corpus/quillmoor/bench/evaluate.py --corpus .tmp/quillmoor/s --label qwen35-4b-base \
        --gguf ~/.finetune-studio/bench-gguf/Qwen__Qwen3.5-4B/model-q8_0.gguf --modes closed rag --package <dir>

closed: the model alone (llama-server from the app's llama.cpp build).
rag:    the exported package's own server with its answer layer pointed at that llama-server (/ask): retrieval, the
        package's grounded prompt and this model writing the answer.
api:    instead of a GGUF, an OpenAI-compatible endpoint (--api-url/--api-model/--api-key-file) as the answer model.
Answers: <corpus>/results/answers/<label>__<mode>.jsonl, then graded in place by judge.py (the app's AI judge).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answers import collect
from brain_matrix import start_server, stop
from common import ROOT, gpu_serialized, load_jsonl, write_json
from judge import judge_files, score
from llm import llama_server
from sft_holdout import TRAIN_SAMPLE_SEED, sft_fact_questions


def subset(qs: list[dict], n_para: int, n_verb: int, seed: int = 7) -> list[dict]:
    """A fixed mix: every multi-hop / reverse / unanswerable / superseded question + a sample of the rest."""
    r = random.Random(seed)
    keep = [q for q in qs if q["type"] in ("multihop", "reverse", "unanswerable", "superseded")]
    para = [q for q in qs if q["type"] == "paraphrase"]
    verb = [q for q in qs if q["type"] == "verbatim"]
    keep += r.sample(para, min(n_para, len(para))) + r.sample(verb, min(n_verb, len(verb)))
    return sorted(keep, key=lambda q: q["id"])


def run_modes(qs: list[dict], out: Path, label: str, modes: list[str], base_url: str, model: str, api_key: str,
              package: Path | None, workers: int) -> list[Path]:
    files = []
    if "closed" in modes:
        f = out / f"{label}__closed.jsonl"
        t0 = time.time()
        n = collect(qs, f, mode="closed", base_url=base_url, model=model, api_key=api_key, workers=workers, label=label)
        print(f"[{label}] closed: {n} answers in {time.time() - t0:.0f}s", flush=True)
        files.append(f)
    if "rag" in modes:
        if package is None:
            raise SystemExit("--package is required for the rag mode")
        f = out / f"{label}__rag.jsonl"
        flags = ["--answer", "--llm-base-url", base_url, "--llm-model", model]
        env_key = {"RAG_LLM_API_KEY": api_key} if api_key else {}
        os.environ.update(env_key)
        p, pkg_url = start_server(package, flags, ROOT / ".tmp" / "bench" / f"server-ask-{label}.log")
        try:
            info = httpx.get(f"{pkg_url}/health", timeout=10).json()
            t0 = time.time()
            n = collect(qs, f, mode="ask", base_url=pkg_url, workers=workers, label=label)
            print(f"[{label}] rag ({info.get('mode')}): {n} answers in {time.time() - t0:.0f}s", flush=True)
        finally:
            stop(p)
            for k in env_key:
                os.environ.pop(k, None)
        files.append(f)
    return files


@gpu_serialized
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gguf")
    ap.add_argument("--api-url")
    ap.add_argument("--api-model", default="")
    ap.add_argument("--api-key-file")
    ap.add_argument("--modes", nargs="+", default=["closed", "rag"])
    ap.add_argument("--package", help="extracted Smart Brain package dir (for the rag mode)")
    ap.add_argument("--n-para", type=int, default=140)
    ap.add_argument("--n-verb", type=int, default=60)
    ap.add_argument("--all", action="store_true", help="every question instead of the fixed subset")
    ap.add_argument("--sft-facts", help="compare a matched seen/held-out fact sample from a file in <corpus>/train/")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ctx-per-slot", type=int, default=6144,
                    help="llama-server context tokens per slot (default: 6144)")
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    if a.ctx_per_slot < 1:
        raise SystemExit("--ctx-per-slot must be positive")
    corpus = Path(a.corpus)
    if a.sft_facts:
        if a.modes != ["closed"]:
            raise SystemExit("--sft-facts requires --modes closed")
        train_file = Path(a.sft_facts).name
        training_path = corpus / "train" / train_file
        canonical_path = corpus / "train" / "train.jsonl"
        if not training_path.is_file() or not canonical_path.is_file():
            raise SystemExit("--sft-facts needs the selected file and train.jsonl under <corpus>/train/")
        qs = sft_fact_questions(load_jsonl(training_path), load_jsonl(canonical_path))
        out = corpus / "results" / "sft-facts"
    else:
        qs = load_jsonl(corpus / "eval" / "questions.jsonl")
        qs = qs if a.all else subset(qs, a.n_para, a.n_verb)
        out = corpus / "results" / "answers"
    out.mkdir(parents=True, exist_ok=True)
    if a.sft_facts:
        expected = {q["id"]: q for q in qs}
        existing_path = out / f"{a.label}__closed.jsonl"
        if existing_path.exists():
            for row in load_jsonl(existing_path):
                question = expected.get(row.get("id"))
                if (question is None or row.get("question") != question["question"]
                        or row.get("key") != question["answer"] or row.get("fact") != question["fact"]
                        or row.get("type") != question["type"]):
                    raise SystemExit(f"{existing_path} contains a question outside this SFT fact comparison; use a new --label")
    pkg = Path(a.package) if a.package else None
    if a.gguf:
        with llama_server(Path(a.gguf).expanduser(), ctx_per_slot=a.ctx_per_slot, parallel=a.workers,
                          log=ROOT / ".tmp" / "bench" / f"llama-{a.label}.log") as url:
            files = run_modes(qs, out, a.label, a.modes, url, "local", "", pkg, a.workers)
    else:
        key = Path(a.api_key_file).read_text().strip() if a.api_key_file else ""
        files = run_modes(qs, out, a.label, a.modes, a.api_url, a.api_model, key, pkg, a.workers)
    if not a.no_judge:
        print(judge_files(files), flush=True)
    summary = {}
    for f in files:
        rows = load_jsonl(f)
        summary[f.stem] = {"all": score(rows), **{t: score([r for r in rows if r["type"] == t])
                                                  for t in sorted({r["type"] for r in rows})}}
        print(f.stem, json.dumps(summary[f.stem]["all"]), flush=True)
    summary_payload = summary
    if a.sft_facts:
        summary_payload = {
            "schema": "quillmoor-sft-facts-v1",
            "train_file": Path(a.sft_facts).name,
            "expected_ids": [q["id"] for q in qs],
            "expected_questions": len(qs),
            "expected_trained_facts": len({q["fact"] for q in qs if q["type"] == "sft-trained"}),
            "expected_heldout_facts": len({q["fact"] for q in qs if q["type"] == "sft-heldout"}),
            "train_sample_seed": TRAIN_SAMPLE_SEED,
            "validation_fact_overlap": 0,
            "ctx_per_slot": a.ctx_per_slot if a.gguf else None,
            "answer_files": [f.name for f in files],
            "scores": summary,
        }
    write_json(out / f"{a.label}__summary.json", summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
