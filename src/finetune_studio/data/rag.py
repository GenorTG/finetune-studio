"""CLI for the portable RAG: build, search, eval, rebuild-vectors, info."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_build(args):
    import shutil

    from finetune_studio.data.rag_portable import DEFAULT_EMBEDDER, PortableRAG
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
    from finetune_studio.data.rag_eval import run_eval_on_corpus

    try:
        report = run_eval_on_corpus(
            args.corpus,
            args.qa_set,
            run_portability=not args.skip_portability,
            use_llm=not args.skip_llm,
        )
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
    payload = report.to_json()
    print(f"Recall@1: {report.recall_at_k.get(1, 0)*100:.1f}%")
    print(f"Recall@5: {report.recall_at_k.get(5, 0)*100:.1f}%")
    print(f"MRR: {report.mrr:.3f}")
    print(
        f"Fact coverage (must_contain, NOT llm-judge): "
        f"{report.fact_coverage_pass_rate*100:.1f}%"
    )
    print(f"Grounding: {report.grounding_pass_rate*100:.1f}%")
    print(f"No-context unknown: {report.no_context_pass_rate*100:.1f}%")
    print(f"Portability: {report.portability_test.get('status')}")
    print(json.dumps({"ok": True, "summary": {
        "recall_at_k": payload["recall_at_k"],
        "mrr": payload["mrr"],
        "fact_coverage_pass_rate": payload["fact_coverage_pass_rate"],
        "grounding_pass_rate": payload["grounding_pass_rate"],
        "no_context_pass_rate": payload["no_context_pass_rate"],
        "portability": payload["portability_test"].get("status"),
        "metadata": payload["metadata"],
    }}, indent=2))



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

    sp = sub.add_parser("eval", help="Run retrieval + grounding + portability tests")
    sp.add_argument("corpus")
    sp.add_argument("--qa-set", required=True, help="Path to a JSON list of {id, query, must_contain, expected_source_contains}")
    sp.add_argument("--skip-portability", action="store_true", help="Skip tar round-trip portability check")
    sp.add_argument("--skip-llm", action="store_true", help="Skip model answers (retrieval-only metrics)")
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
    """Back-compat shim — prefer ``finetune_studio.data.rag_eval.run_eval_on_corpus``."""
    from finetune_studio.data.rag_eval import run_eval_on_corpus as _run

    report = _run(corpus, qa_set_path)
    print(f"Recall@1: {report.recall_at_k.get(1, 0)*100:.1f}%")
    print(f"Recall@5: {report.recall_at_k.get(5, 0)*100:.1f}%")
    print(f"MRR: {report.mrr:.3f}")
    print(
        f"Fact coverage (must_contain substring; NOT llm-judge): "
        f"{report.fact_coverage_pass_rate*100:.1f}%"
    )
    print(f"Portability: {report.portability_test.get('status')}")
    return report


if __name__ == "__main__":
    main()
