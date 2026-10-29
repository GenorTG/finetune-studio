"""A/B: meaning search with the embedder's own query / document prompts vs. the app's plain encode().

    .venv/bin/python tests/corpus/quillmoor/bench/prompt_ab.py --corpus .tmp/quillmoor/s --index ~/.finetune-studio/brain/<pid> \
        --embedders intfloat/multilingual-e5-large Qwen/Qwen3-Embedding-0.6B

Same passages (the project index's chunks.parquet), same questions and scoring as retrieval.py; dense ranking only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_jsonl, write_json
from retrieval import ANSWERABLE, KS, rank_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--embedders", nargs="+", required=True)
    a = ap.parse_args()
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    df = pd.read_parquet(Path(a.index).expanduser() / "chunks.parquet")
    texts = df["text"].tolist()
    names = (df["filename"] if "filename" in df else df["source"]).tolist()
    qs = [q for q in load_jsonl(Path(a.corpus) / "eval" / "questions.jsonl") if q["type"] in ANSWERABLE and q["type"] != "verbatim"]
    out = {}
    for emb in a.embedders:
        m = SentenceTransformer(emb, device="cuda")
        res = {"model_prompts": {k: v[:60] for k, v in (getattr(m, "prompts", {}) or {}).items()}}
        for variant in ("plain", "prompts"):
            if variant == "plain":
                D = m.encode(texts, normalize_embeddings=True, batch_size=32)
                Q = m.encode([q["question"] for q in qs], normalize_embeddings=True, batch_size=64)
            else:
                D = m.encode_document(texts, normalize_embeddings=True, batch_size=32)
                Q = m.encode_query([q["question"] for q in qs], normalize_embeddings=True, batch_size=64)
            S = Q @ D.T
            ranks = []
            for i, q in enumerate(qs):
                top = np.argsort(-S[i])[:10]
                hits = [{"filename": names[j], "text": texts[j]} for j in top]
                ranks.append(rank_of(q, hits)[0])
            n = len(ranks)
            res[variant] = {f"hit@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in KS}
            res[variant]["mrr@10"] = round(sum(1 / r for r in ranks if r) / n, 4)
        out[emb] = res
        print(emb, "plain", res["plain"], "prompts", res["prompts"], "| model prompts:", res["model_prompts"], flush=True)
        del m
        import torch
        torch.cuda.empty_cache()
    write_json(Path(a.corpus) / "results" / "prompt_ab.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
