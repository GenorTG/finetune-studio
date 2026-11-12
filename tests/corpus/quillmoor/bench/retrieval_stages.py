"""Where does a question lose its evidence? The rank of the answering passage after each retrieval stage.

    .venv/bin/python tests/corpus/quillmoor/bench/retrieval_stages.py --corpus .tmp/quillmoor/s \
        --index ~/.finetune-studio/brain/<pid> --types reverse,multihop

Stages: meaning search alone, keyword search alone, their RRF fusion, and the index's full search (with its reranker).
For two-step questions it also counts each hop separately in the final top 10. Results: RESULTS.md section 8.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import rank_of

from finetune_studio.smart_brain import BrainIndex
from finetune_studio.smart_brain.rrf import rrf_fuse

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--types", default="reverse,multihop")
args = ap.parse_args()
types = args.types.split(",")
qs = [json.loads(line) for line in (Path(args.corpus) / "eval" / "questions.jsonl").read_text().splitlines()]
qs = [q for q in qs if q["type"] in types]
rag = BrainIndex(Path(args.index).expanduser()).load()
ch = rag.chunks
ids = rag._chunk_ids


def hits_of(order):
    return [{"filename": ch.iloc[rag.idx_map[c]]["filename"], "text": ch.iloc[rag.idx_map[c]]["text"]} for c in order]


stats = Counter()
per_fact = Counter()
for q in qs:
    qv = rag.encode(q["question"])
    dense = [ids[i] for i in np.argsort(-(rag.vectors @ qv))]
    bs = rag.bm25.score(q["question"])
    bm = [ids[i] for i in np.argsort(-bs) if bs[i] > 0]
    fused = [c for c, _ in sorted(rrf_fuse([dense, bm], k=60).items(), key=lambda kv: -kv[1])]
    final = [h["chunk_id"] for h in rag.search(q["question"], top_k=10)]
    row = {}
    for name, order in (("dense", dense), ("bm25", bm), ("rrf", fused), ("final", final)):
        r, _ = rank_of(q, hits_of(order[:50]))
        row[name] = r
        for k in (5, 10, 50):
            if r is not None and r <= k:
                stats[(q["type"], name, k)] += 1
    if q["type"] == "multihop":
        # each hop separately in the final top-10
        for e in q["evidence"]:
            sub = dict(q, evidence=[e])
            r, _ = rank_of(sub, hits_of(final))
            per_fact[(e["fact"].split(":")[0] + ":" + e["fact"].split(":")[-1], r is not None)] += 1
    print(q["type"], row, q["question"][:80], flush=True)

n = Counter(q["type"] for q in qs)
for t in types:
    print(f"== {t} (n={n[t]})")
    for name in ("dense", "bm25", "rrf", "final"):
        print("  ", name, {k: stats[(t, name, k)] for k in (5, 10, 50)})
if per_fact:
    print("multihop per hop found in final top-10:")
    for (k, ok), v in sorted(per_fact.items()):
        print("  ", k, ok, v)
