"""Single-client hit@5 and latency for reranker variants over the same hybrid candidates of a studio index.

    .venv/bin/python tests/corpus/quillmoor/bench/rerank_ab.py --corpus .tmp/quillmoor/s \
        --index ~/.finetune-studio/brain/<pid> [--variants minilm-fp32,bge-fp16-top50]

Results: RESULTS.md section 8 (the reranker default).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import ANSWERABLE, rank_of

from finetune_studio.smart_brain import BrainIndex

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--variants", default="")
args = ap.parse_args()
rag = BrainIndex(Path(args.index).expanduser()).load()
qs = [json.loads(line) for line in (Path(args.corpus) / "eval" / "questions.jsonl").read_text().splitlines()]
qs = [q for q in qs if q["type"] in ANSWERABLE and q["type"] != "verbatim"]


def make(name: str, fp16: bool):
    kw = {"model_kwargs": {"torch_dtype": torch.float16}} if fp16 else {}
    m = CrossEncoder(name, device="cuda", max_length=512, **kw)
    return lambda q, docs: [float(s) for s in m.predict([[q, d] for d in docs], batch_size=32, show_progress_bar=False)]


ALL = [
    ("minilm-fp32", "cross-encoder/ms-marco-MiniLM-L-6-v2", False, 50),
    ("bge-fp32-top50", "BAAI/bge-reranker-v2-m3", False, 50),
    ("bge-fp16-top50", "BAAI/bge-reranker-v2-m3", True, 50),
    ("bge-fp16-top30", "BAAI/bge-reranker-v2-m3", True, 30),
]
variants = [v for v in ALL if not args.variants or v[0] in args.variants.split(",")]
for label, name, fp16, top_n in variants:
    rag._reranker = make(name, fp16)
    rag.search("warm up query", top_k=10, rerank=True, rerank_top_n=top_n)
    lat, ranks = [], []
    for q in qs:
        t = time.perf_counter()
        hits = rag.search(q["question"], top_k=10, rerank=True, rerank_top_n=top_n)
        lat.append(time.perf_counter() - t)
        ranks.append(rank_of(q, hits)[0])
    by = {}
    for q, r in zip(qs, ranks, strict=True):
        by.setdefault(q["type"], []).append(bool(r and r <= 5))
    hit5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
    print(label, f"hit@5 {hit5:.4f}", {t: f"{sum(v)}/{len(v)}" for t, v in by.items()},
          f"p50 {statistics.median(lat)*1000:.0f} ms p95 {sorted(lat)[int(0.95*len(lat))]*1000:.0f} ms",
          f"vram {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)
    rag._reranker = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
