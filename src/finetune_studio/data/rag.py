"""CLI for the portable RAG: build, search, eval, rebuild-vectors, info."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_build(args):
    import shutil
    from finetune_studio.data.rag_portable import PortableRAG, DEFAULT_EMBEDDER
    corpus = Path(args.corpus)
    if args.reset and corpus.exists():
        shutil.rmtree(corpus)
    rag = PortableRAG(corpus)
    result = rag.build_from_directory(
        source_dir=args.source,
        name=args.name or corpus.name,
        embedder=args.embedder or DEFAULT_EMBEDDER,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(result, indent=2))


def _cmd_search(args):
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(args.corpus).load()
    hits = rag.search(args.query, top_k=args.top_k,
                     hybrid=not args.no_hybrid, rerank=not args.no_rerank)
    for h in hits:
        score = h.get("ce_score", h.get("rrf_score", 0))
        print(f"[{h['rank']}] ({h['filename']}, score {score:.3f})")
        print(f"    {h['text'][:200].strip()}...")
        print()


def _cmd_eval(args):
    """Run the eval suite against a corpus + question set."""
    import importlib
    mod = importlib.import_module("finetune_studio.data.rag_eval")
    sys.path.insert(0, str(Path(__file__).parent))  # for qa_set import path
    run_eval_on_corpus(args.corpus, args.qa_set)


def _cmd_rebuild(args):
    from finetune_studio.data.rag_portable import PortableRAG
    rag = PortableRAG(args.corpus)
    result = rag.rebuild_vectors(embedder=args.embedder)
    print(json.dumps(result, indent=2))


def _cmd_info(args):
    from finetune_studio.data.rag_portable import read_json
    m = read_json(Path(args.corpus) / "manifest.json")
    print(json.dumps(m, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(prog="finetune_studio.data.rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("build", help="Parse a directory and build a portable RAG corpus")
    sp.add_argument("corpus")
    sp.add_argument("source")
    sp.add_argument("--name")
    sp.add_argument("--embedder")
    sp.add_argument("--chunk-size", type=int, default=400)
    sp.add_argument("--overlap", type=int, default=80)
    sp.add_argument("--reset", action="store_true", help="Wipe existing corpus first")
    sp.set_defaults(func=_cmd_build)

    sp = sub.add_parser("search", help="Query a corpus")
    sp.add_argument("corpus")
    sp.add_argument("query")
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--no-hybrid", action="store_true")
    sp.add_argument("--no-rerank", action="store_true")
    sp.set_defaults(func=_cmd_search)

    sp = sub.add_parser("eval", help="Run retrieval + LLM-as-judge tests")
    sp.add_argument("corpus")
    sp.add_argument("--qa-set", required=True, help="Path to a JSON list of {id, query, must_contain, expected_source_contains}")
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("rebuild-vectors", help="Re-embed with a (possibly different) model")
    sp.add_argument("corpus")
    sp.add_argument("--embedder")
    sp.set_defaults(func=_cmd_rebuild)

    sp = sub.add_parser("info", help="Show manifest.json")
    sp.add_argument("corpus")
    sp.set_defaults(func=_cmd_info)

    args = p.parse_args()
    args.func(args)


def run_eval_on_corpus(corpus: str, qa_set_path: str):
    """Stand-alone runner used by the CLI. Loads QA set, runs eval, writes history."""
    import time
    from dataclasses import dataclass, field
    from finetune_studio.data.rag_portable import PortableRAG
    from finetune_studio.data.rag_eval import (
        QAEntry, evaluate_retrieval, llm_as_judge, write_results, EvalReport,
    )
    from finetune_studio.models.manager import get_manager

    qa_data = json.loads(Path(qa_set_path).read_text())
    qa = [QAEntry(**{k: v for k, v in q.items() if k in QAEntry.__dataclass_fields__}) for q in qa_data]

    rag = PortableRAG(corpus)
    if not rag.exists():
        print(json.dumps({"error": "corpus not built"}))
        sys.exit(1)
    q = rag.load()

    print(f"Loaded {len(q.chunks)} chunks, vectors shape {q.vectors.shape}")
    print(f"Manifest: embedder={rag.manifest_path.read_text().split(chr(10))[6] if False else 'see manifest.json'}")

    t0 = time.time()
    ret_results, ret_metrics = evaluate_retrieval(q, qa, ks=(1, 3, 5))
    print(f"Recall@1: {ret_metrics['recall_at_k'][1]*100:.1f}%")
    print(f"Recall@5: {ret_metrics['recall_at_k'][5]*100:.1f}%")
    print(f"MRR: {ret_metrics['mrr']:.3f}")

    mgr = get_manager()
    if mgr.active() is None:
        mgr.load("local-default")
    llm_results = llm_as_judge(mgr, q, qa, top_k=5)
    passed = sum(1 for r in llm_results if r.passed)
    print(f"LLM-as-judge: {passed}/{len(qa)} ({passed/len(qa)*100:.1f}%)")

    report = EvalReport(
        corpus=Path(corpus).name,
        timestamp=time.time(),
        embedding_model=q.manifest.embedding_model.name,
        total_questions=len(qa),
        retrieval_results=[r.__dict__ for r in ret_results],
        llm_judge_results=[r.__dict__ for r in llm_results],
        recall_at_k=ret_metrics["recall_at_k"],
        mrr=ret_metrics["mrr"],
        llm_pass_rate=round(passed/len(qa), 3),
        portability_test={},
        timing={"retrieve": time.time()-t0},
    )
    out = write_results(Path(corpus), report)
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    main()
