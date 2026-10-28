"""`fts brain {build,search,ask,eval,rebuild-vectors,info,export}` — the Smart Brain from the command line.

Every subcommand takes either ``--project <id|name>`` (the project's index, built from its parsed files exactly like
the WebUI: ``knowledge.index``) or ``--corpus <dir>`` (a standalone index; ``build`` then reads a plain directory with
``--from``). Methods: semantic search (e5 embeddings) + keyword search (BM25), combined by rank fusion, then reranking.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _corpus(args: argparse.Namespace) -> Path:
    if bool(args.project) == bool(args.corpus):
        _die("give exactly one of --project or --corpus")
    if args.corpus:
        return Path(args.corpus)
    from finetune_studio.cli.commands.dataset import resolve_project
    from finetune_studio.data.fs.paths import brain_index_dir

    return brain_index_dir(resolve_project(args.project)["id"])


def _loaded(args: argparse.Namespace):
    from finetune_studio.smart_brain import BrainIndex

    rag = BrainIndex(_corpus(args))
    if not rag.exists():
        _die(f"no index at {rag.dir}: build it first (fts brain build)")
    return rag.load()


def _build(args: argparse.Namespace) -> None:
    from finetune_studio.smart_brain import DEFAULT_EMBEDDER, BrainIndex

    if args.project:
        if args.source:
            _die("--from reads a plain directory; a project index is built from the project's parsed files")
        if args.chunk_size is not None or args.overlap is not None:
            _die("--chunk-size/--overlap apply to --corpus builds; a project indexes its files' passages")
        from finetune_studio.cli.commands.dataset import resolve_project
        from finetune_studio.knowledge import index
        from finetune_studio.knowledge.errors import KnowledgeError

        try:
            result = index.build(resolve_project(args.project)["id"], embedder=args.embedder, reset=args.reset,
                                 fact_cards=args.fact_cards)
        except KnowledgeError as e:
            _die(str(e))
    else:
        if not args.source:
            _die("--corpus needs --from <directory> to read")
        corpus = _corpus(args)
        if args.reset and corpus.exists():
            shutil.rmtree(corpus)
        result = BrainIndex(corpus).build_from_directory(
            args.source, name=args.name or corpus.name, embedder=args.embedder or DEFAULT_EMBEDDER,
            chunk_size=400 if args.chunk_size is None else args.chunk_size,
            overlap=80 if args.overlap is None else args.overlap)
    print(json.dumps(result, indent=2))


def _search(args: argparse.Namespace) -> None:
    hits = _loaded(args).search(args.query, top_k=args.top_k, hybrid=not args.no_hybrid, rerank=not args.no_rerank)
    if args.json:
        print(json.dumps(hits, indent=2, default=str))
        return
    if not hits:
        print("No results.")
    for h in hits:
        score = h.get("ce_score", h.get("rrf_score", 0))
        print(f"[{h['rank']}] ({h['filename']}, score {score:.3f})")
        print(f"    {h['text'][:200].strip()}...\n")


def _ask(args: argparse.Namespace) -> None:
    """One question to a model with the Smart Brain's context (the same prompt layout as the WebUI chat)."""
    from finetune_studio.inference import describe_worker, get_engine
    from finetune_studio.smart_brain.prompt import (
        build_messages,
        format_context_blocks,
    )

    if not Path(args.model).exists():
        _die(f"model not found: {args.model}")
    hits = _loaded(args).search(args.question, top_k=args.top_k)
    engine = get_engine()
    print(f"Loading {args.model}...", file=sys.stderr)
    engine.load(args.model)
    print(f"Served by {describe_worker(engine)}.", file=sys.stderr)
    try:
        messages = build_messages([{"role": "user", "content": args.question}], format_context_blocks(hits),
                                  args.system_prompt)
        answer = engine.generate(messages, max_tokens=args.max_tokens)
    finally:
        engine.unload()
    sources = [{"filename": h["filename"], "score": h.get("ce_score", h.get("rrf_score", 0)), "text": h["text"][:200]}
               for h in hits]
    if args.json:
        print(json.dumps({"response": answer, "sources": sources}, indent=2, default=str))
        return
    print(f"\nAI: {answer}")
    if sources:
        print(f"\n--- Sources ({len(sources)}) ---")
        for s in sources:
            print(f"  [{s['score']:.3f}] {s['filename']}")


def _eval(args: argparse.Namespace) -> None:
    from finetune_studio.smart_brain.evaluation import run_eval_on_corpus

    try:
        report = run_eval_on_corpus(str(_corpus(args)), args.qa_set, run_portability=not args.skip_portability,
                                    use_llm=not args.skip_llm)
    except FileNotFoundError as exc:
        _die(str(exc))
    payload = report.to_json()
    print(f"Recall@1: {report.recall_at_k.get(1, 0) * 100:.1f}%")
    print(f"Recall@5: {report.recall_at_k.get(5, 0) * 100:.1f}%")
    print(f"MRR: {report.mrr:.3f}")
    print(f"Fact coverage (must_contain, NOT llm-judge): {report.fact_coverage_pass_rate * 100:.1f}%")
    print(f"Grounding: {report.grounding_pass_rate * 100:.1f}%")
    print(f"No-context unknown: {report.no_context_pass_rate * 100:.1f}%")
    print(f"Portability: {report.portability_test.get('status')}")
    print(json.dumps({"ok": True, "summary": {k: payload[k] for k in (
        "recall_at_k", "mrr", "fact_coverage_pass_rate", "grounding_pass_rate", "no_context_pass_rate", "metadata")}
        | {"portability": payload["portability_test"].get("status")}}, indent=2))


def _rebuild_vectors(args: argparse.Namespace) -> None:
    from finetune_studio.smart_brain import BrainIndex

    print(json.dumps(BrainIndex(_corpus(args)).rebuild_vectors(embedder=args.embedder), indent=2))


def _info(args: argparse.Namespace) -> None:
    from finetune_studio.smart_brain import read_json

    manifest = _corpus(args) / "manifest.json"
    if not manifest.is_file():
        _die(f"no index at {manifest.parent}")
    info = read_json(manifest)
    if args.project:
        from finetune_studio.cli.commands.dataset import resolve_project
        from finetune_studio.knowledge import index

        info["state"] = index.state(resolve_project(args.project)["id"]).as_dict()
    print(json.dumps(info, indent=2, ensure_ascii=False))


def _export(args: argparse.Namespace) -> None:
    """Standalone server package (the Smart Brain page's "Build & download standalone server")."""
    from pydantic import ValidationError

    from finetune_studio.smart_brain import export_config as ec
    from finetune_studio.smart_brain.server_package import (
        build_package,
        pipeline_text,
    )

    corpus = _corpus(args)
    if not (corpus / "manifest.json").is_file():
        _die(f"no index at {corpus}: build it first (fts brain build)")
    patch = {k: v for k, v in {
        "archive_format": args.format, "include_models": args.models, "encrypt": args.encrypt,
        "semantic_search": args.semantic, "keyword_search": args.keyword, "reranker_enabled": args.reranker,
        "answer_enabled": args.answer, "llm_base_url": args.llm_base_url, "llm_model": args.llm_model,
        "llm_connection": args.llm_connection,
        "top_k": args.top_k, "port": args.port, "host": args.host, "device": args.device,
    }.items() if v is not None}
    conn = patch.pop("llm_connection", None)
    if conn:
        from finetune_studio.webui.routes.brain_export import connection_llm_settings

        try:
            patch = {**connection_llm_settings(conn), **patch}
        except Exception as e:  # noqa: BLE001 - HTTPException from the shared helper: report its reason
            _die(getattr(e, "detail", str(e)))
    passphrase = None
    if args.passphrase_file:
        passphrase = Path(args.passphrase_file).read_text(encoding="utf-8").strip()
    try:
        if args.project:
            from finetune_studio.cli.commands.dataset import resolve_project

            project = resolve_project(args.project)
            cfg = ec.effective_config(project["id"], patch)
            title = args.name or project["name"] or project["id"]
            out_dir = Path(args.out) if args.out else Path("output") / "projects" / project["id"] / "brain-packages"
        else:
            cfg = ec.BrainExportConfig(**patch)
            title = args.name or corpus.name
            out_dir = Path(args.out or ".")
    except (ValidationError, ValueError) as e:
        _die(str(e))
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in title).strip(".-")[:48] or "brain"
    ext = "zip" if cfg.archive_format == "zip" else cfg.archive_format
    out = out_dir / (f"{safe}-smart-brain{'-with-models' if cfg.include_models else ''}"
                     f"{'' if cfg.encrypt else '-PLAINTEXT'}.{ext}")
    try:
        result = build_package(corpus, out, name=title, config=cfg, passphrase=passphrase)
    except (FileNotFoundError, ValueError) as e:
        _die(str(e))
    print(json.dumps({"path": str(result.path), "size": result.size, "encrypted": result.encrypted,
                      "layers": result.layers,
                      "pipeline": pipeline_text(cfg, bundled_reranker=result.layers.get("reranker", False)),
                      **({"passphrase": result.passphrase, "note": "shown once: store it now, it is not saved"}
                         if result.passphrase else {})}, indent=2))


_HANDLERS = {"build": _build, "search": _search, "ask": _ask, "eval": _eval, "rebuild-vectors": _rebuild_vectors,
             "info": _info, "export": _export}


def cmd_brain(args: argparse.Namespace) -> None:
    handler = _HANDLERS.get(args.brain_command or "")
    if handler is None:
        print("Usage: fts brain {build,search,ask,eval,rebuild-vectors,info,export} (--project P | --corpus DIR) ...",
              file=sys.stderr)
        sys.exit(1)
    handler(args)


def add_parser(sub: argparse._SubParsersAction) -> None:
    """``fts brain …`` subcommands (called from ``cli._parser``)."""
    p = sub.add_parser("brain", help="Smart Brain: build, search and ask a project's (or a standalone) index")
    brain = p.add_subparsers(dest="brain_command")

    def target(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument("--project", help="Project id or name (its index is built from its parsed files)")
        sp.add_argument("--corpus", help="Standalone index directory")
        return sp

    sp = target(brain.add_parser("build", help="Build (replace) the index"))
    sp.add_argument("--from", dest="source", help="--corpus only: plain directory of files to parse and index")
    sp.add_argument("--name")
    sp.add_argument("--embedder")
    sp.add_argument("--chunk-size", type=int, default=None,
                    help="--corpus only: chunk size in embedder tokens, default 400 (a project indexes its files' passages)")
    sp.add_argument("--overlap", type=int, default=None, help="--corpus only: overlap in tokens, default 80")
    sp.add_argument("--fact-cards", action=argparse.BooleanOptionalAction, default=None,
                    help="--project only: also index approved Q&A pairs (default: keep the last build's choice)")
    sp.add_argument("--reset", action="store_true", help="Wipe the index directory first")

    sp = target(brain.add_parser("search", help="Search the index"))
    sp.add_argument("query")
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--no-hybrid", action="store_true", help="Semantic search only (no keyword search / fusion)")
    sp.add_argument("--no-rerank", action="store_true")
    sp.add_argument("--json", action="store_true")

    sp = target(brain.add_parser("ask", help="Ask a model one question with the index's context"))
    sp.add_argument("model", help="Path to a model")
    sp.add_argument("question")
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.add_argument("--system-prompt", default="")
    sp.add_argument("--json", action="store_true")

    sp = target(brain.add_parser("eval", help="Retrieval + grounding + portability tests"))
    sp.add_argument("--qa-set", required=True, help="JSON list of {id, query, must_contain, expected_source_contains}")
    sp.add_argument("--skip-portability", action="store_true")
    sp.add_argument("--skip-llm", action="store_true", help="Retrieval-only metrics")

    sp = target(brain.add_parser("rebuild-vectors", help="Re-embed with a (possibly different) embedder"))
    sp.add_argument("--embedder")

    target(brain.add_parser("info", help="Show the index manifest (and, for a project, what is stale)"))

    sp = target(brain.add_parser(
        "export", help="Build the standalone server package (HTTP + MCP), choosing its search layers",
        description="Each search layer is separate: meaning search (vectors), word search (BM25; both = rank "
                    "fusion), the reranker and answer generation (an LLM writes the answer). Unset flags take the "
                    "project's saved export settings, then the studio defaults."))
    bo = argparse.BooleanOptionalAction
    sp.add_argument("--semantic", action=bo, default=None, help="meaning search over the vectors (default on)")
    sp.add_argument("--keyword", action=bo, default=None, help="word search over the BM25 index (default on)")
    sp.add_argument("--reranker", action=bo, default=None, help="cross-encoder rescoring (needs --models)")
    sp.add_argument("--answer", action=bo, default=None,
                    help="answer generation: the server's /ask and rag_answer call an LLM (default off)")
    sp.add_argument("--llm-base-url", help="OpenAI-compatible endpoint for answers, e.g. http://127.0.0.1:11434/v1")
    sp.add_argument("--llm-model", help="model name for answers")
    sp.add_argument("--llm-connection",
                    help="copy the answer endpoint, model and headers from a Settings → AI assistant connection id "
                         "(never its key: set RAG_LLM_API_KEY where the server runs)")
    sp.add_argument("--models", action=bo, default=None,
                    help="bundle the embedder (+ reranker) for offline use (~2 GB, not encrypted)")
    sp.add_argument("--encrypt", action=bo, default=None, help="encrypt the corpus (default on)")
    sp.add_argument("--passphrase-file", help="file holding the passphrase (default: generate one, printed once)")
    sp.add_argument("--format", choices=["tar.gz", "tar", "zip"])
    sp.add_argument("--top-k", type=int)
    sp.add_argument("--host")
    sp.add_argument("--port", type=int)
    sp.add_argument("--device")
    sp.add_argument("--name", help="package title (default: the project name)")
    sp.add_argument("--out", help="output directory (default: output/projects/<id>/brain-packages)")
