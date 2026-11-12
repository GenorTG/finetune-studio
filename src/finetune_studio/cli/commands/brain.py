"""`fts brain {build,search,ask,eval,rebuild-vectors,info,export,bundle,import}` — the Smart Brain from the command line.

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
    from finetune_studio.smart_brain.standalone_server import answer_with_follow_ups

    if not Path(args.model).exists():
        _die(f"model not found: {args.model}")
    index = _loaded(args)
    hits = index.search(args.question, top_k=args.top_k)
    engine = get_engine()
    print(f"Loading {args.model}...", file=sys.stderr)
    engine.load(args.model)
    print(f"Served by {describe_worker(engine)}.", file=sys.stderr)
    try:
        # The Smart Brain chat's answer step, follow-up search included (smart_brain.standalone_server).
        result = answer_with_follow_ups(
            args.question, hits, lambda query: index.search(query, top_k=args.top_k),
            lambda messages: engine.generate(messages, max_tokens=args.max_tokens),
            source_search=lambda query, prefix: index.search(
                query, top_k=args.top_k, source_prefix=prefix),
            follow_ups=args.follow_up, system_prompt=args.system_prompt,
            missing_part_retry=getattr(args, "retry_missing", True))
    finally:
        engine.unload()
    answer, hits = result["answer"], result["hits"]
    sources = [{"filename": h["filename"], "score": h.get("ce_score", h.get("rrf_score", 0)), "text": h["text"][:200]}
               for h in hits]
    if args.json:
        print(json.dumps({"response": answer, "sources": sources, "follow_up_searches": result["follow_up_searches"],
                          "followup": result.get("followup")}, indent=2, default=str))
        return
    print(f"\nAI: {answer}")
    for query in result["follow_up_searches"]:
        print(f"(also searched: {query})")
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

    try:
        print(json.dumps(BrainIndex(_corpus(args)).rebuild_vectors(embedder=args.embedder), indent=2))
    except ValueError as e:
        _die(str(e))


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
    """Standalone server package (the Package page's "Build & download the package")."""
    from pydantic import ValidationError

    from finetune_studio.smart_brain import export_config as ec
    from finetune_studio.smart_brain.server_package import (
        build_package,
        pipeline_text,
    )

    corpus = _corpus(args)
    if not (corpus / "manifest.json").is_file():
        _die(f"no index at {corpus}: build it first (fts brain build)")
    if args.project:     # the web export's gate: out-of-date documents are reviewed (or accepted) before packaging
        from finetune_studio.cli.commands.dataset import resolve_project
        from finetune_studio.knowledge.lint import (
            SUPERSEDED_HINT,
            lint,
            superseded_files,
        )

        stale = superseded_files(lint(resolve_project(args.project)["id"]))
        if stale and not args.accept_superseded:
            _die("Review superseded documents before packaging: " + ", ".join(stale) + ". " + SUPERSEDED_HINT
                 + " (fts brain lint shows them; --accept-superseded packages them anyway.)")
        if stale:
            print(f"Packaging {len(stale)} out-of-date document(s) on purpose (--accept-superseded): "
                  + ", ".join(stale), file=sys.stderr)
    patch = {k: v for k, v in {
        "archive_format": args.format, "include_models": args.models, "encrypt": args.encrypt,
        "semantic_search": args.semantic, "keyword_search": args.keyword, "reranker_enabled": args.reranker,
        "answer_enabled": args.answer, "llm_base_url": args.llm_base_url, "llm_model": args.llm_model,
        "llm_provider": args.llm_provider,
        "llm_connection": args.llm_connection, "include_originals": args.originals,
        "hide_file_names": getattr(args, "hide_file_names", None),
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
    from finetune_studio.smart_brain import package_meta as pm

    history_dir = pm.package_dir(resolve_project(args.project)["id"]) if args.project else out_dir
    base = None
    if args.base:
        from finetune_studio.smart_brain import package_import

        try:
            base = package_import.read_record(args.base, _read_pass(args.base_passphrase_file), out_dir)
        except Exception as e:  # noqa: BLE001 - wrong passphrase / not a package: the reason is the message
            _die(f"cannot read {args.base}: {e}")
        pm.add_history(history_dir, {**base, "origin": "uploaded", "filename": Path(args.base).name,
                                     "name": Path(args.base).name})
    elif args.new_version_of:
        base = pm.latest(history_dir, args.new_version_of)
        if base is None:
            _die(f"no package {args.new_version_of!r} in {pm.history_path(history_dir)}")
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in title).strip(".-")[:48] or "brain"
    ext = "zip" if cfg.archive_format == "zip" else cfg.archive_format
    from finetune_studio.smart_brain import originals as brain_originals
    from finetune_studio.smart_brain import scope as brain_scope

    pid = resolve_project(args.project)["id"] if args.project else None
    docs = brain_scope.documents(corpus, pid)
    try:
        keep, scope_record = brain_scope.resolve(brain_scope.DocumentScope(
            include=args.include or [], exclude=args.exclude or [], tags=args.tag or []), docs)
    except ValueError as e:
        _die(str(e))
    probes = brain_scope.probe_queries(corpus, {d.document_id for d in docs} - keep) if keep is not None else {}
    doc_rows = pm.document_manifest(corpus, pid, keep)
    try:
        record = pm.make_record(base=base, documents=doc_rows, version=args.version)
    except ValueError as e:
        _die(str(e))
    diff = pm.changes(base["documents"] if base else None, doc_rows)
    record["changes"] = pm.summary(diff)
    changes_md = pm.changes_markdown(record, diff)
    out = out_dir / pm.package_filename(safe, record, ext=ext, models=cfg.include_models, encrypted=cfg.encrypt)
    originals = brain_originals.collect(corpus, pid) if cfg.include_originals else []
    try:
        result = build_package(corpus, out, name=title, config=cfg, passphrase=passphrase,
                               identity=_identity_for(args, package=True), originals=originals,
                               documents=keep, scope=scope_record, package=record, changes_md=changes_md,
                               changes_md_public=pm.changes_markdown(record, diff, names=False))
    except (FileNotFoundError, ValueError) as e:
        _die(str(e))
    from finetune_studio.smart_brain.server_package import (
        studio_models_for,
        verify_package,
    )

    verified = verify_package(result.path, result.passphrase or passphrase, studio_models=studio_models_for(corpus),
                              expected_originals={o.document_id: o.sha256 for o in originals}, out_of_scope=probes)
    pm.add_history(history_dir, {**record, "origin": "exported", "filename": result.path.name, "name": title,
                                 "scope": scope_record["label"], "encrypted": result.encrypted, "changes_md": changes_md})
    if args.project:
        from finetune_studio.cli.commands.dataset import resolve_project
        from finetune_studio.knowledge.readiness import record_package

        record_package(resolve_project(args.project)["id"], filename=result.path.name, size=result.size,
                       encrypted=result.encrypted, identity=bool(_identity_for(args)), verified=verified)
    print(json.dumps({"path": str(result.path), "size": result.size, "encrypted": result.encrypted,
                      "verified": verified, "layers": result.layers,
                      "original_files": sum(1 for o in originals if keep is None or o.document_id in keep),
                      "scope": scope_record,
                      "package": {k: record[k] for k in ("package_id", "version", "parent_version", "created")},
                      "changes": record["changes"],
                      "pipeline": pipeline_text(cfg, bundled_reranker=result.layers.get("reranker", False)),
                      **({"passphrase": result.passphrase, "note": "shown once: store it now, it is not saved"}
                         if result.passphrase else {})}, indent=2))


def _identity_for(args: argparse.Namespace, *, package: bool = False) -> dict | None:
    """The project's identity for an export; a standalone ``--corpus`` has none."""
    if not args.project:
        return None
    from finetune_studio.cli.commands.dataset import resolve_project
    from finetune_studio.smart_brain import identity as brain_identity

    pid = resolve_project(args.project)["id"]
    saved = brain_identity.load(pid)
    if not brain_identity.is_set(saved):
        return None
    return brain_identity.package_record(pid) if package else saved


def _identity(args: argparse.Namespace) -> None:
    """Show or change who the Smart Brain speaks for (the page's Identity card). Unset flags keep their value."""
    from finetune_studio.cli.commands.dataset import resolve_project
    from finetune_studio.smart_brain import identity as brain_identity

    if not args.project:
        _die("--project is required")
    pid = resolve_project(args.project)["id"]
    saved = brain_identity.load(pid)
    changes = {k: v for k, v in {"name": args.name, "about": _text_arg(args.about),
                                 "personality": _text_arg(args.personality),
                                 "answer_rules": _text_arg(args.answer_rules)}.items() if v is not None}
    if changes:
        try:
            saved = brain_identity.save(pid, {**saved, **changes})
        except brain_identity.IdentityError as e:
            _die(str(e))
    print(json.dumps({**saved, "instructions": brain_identity.compose(saved)}, indent=2, ensure_ascii=False))


def _pictures(args: argparse.Namespace) -> None:
    """Pictures only OCR read (the Knowledge check's "Read pictures"): show them, or ``--read`` them with the Vision
    reader (the ``pictures`` job; ``--wait`` until it ends)."""
    from finetune_studio.cli.commands.dataset import resolve_project
    from finetune_studio.knowledge import pictures

    if not args.project:
        _die("--project is required")
    pid = resolve_project(args.project)["id"]
    if args.read:
        try:
            started = pictures.start(pid)
        except pictures.PicturesError as e:
            _die(str(e))
        if args.wait:
            from finetune_studio.jobs import launch
            started = {"job": launch.wait(started["id"], timeout=args.timeout)}
        print(json.dumps(started, indent=2, default=str))
        return
    st = pictures.project_status(pid)
    if args.json:
        print(json.dumps(st, indent=2, ensure_ascii=False, default=str))
        return
    print(f"Vision reader: {st['reader']['label']}")
    print(f"{st['pictures']} picture(s), {st['scanned']} scanned page(s): {st['read']} read by a vision model, "
          f"{st['pictures_ocr']} picture(s) and {st['scanned_ocr']} page(s) by OCR only")
    for f in st["files"]:
        print(f"  {f['name']}: {f['pictures_ocr']} picture(s), {f['scanned_ocr']} page(s)"
              + (" (mined: Reset it in Mining first)" if f["mined"] else ""))


def _lint(args: argparse.Namespace) -> None:
    """Knowledge check of a project's documents (the page's Knowledge check). Exits 1 when there are errors."""
    from finetune_studio.cli.commands.dataset import resolve_project
    from finetune_studio.knowledge.lint import lint

    if not args.project:
        _die("--project is required")
    report = lint(resolve_project(args.project)["id"])
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"{report['files']} files: {report['errors']} errors, {report['warnings']} warnings, {report['ignored']} ignored")
        for f in report["findings"]:
            if not f["ignored"]:
                print(f"[{f['severity']}] {f['title']}")
                for line in f["detail"]:
                    print(f"    {line}")
                print(f"    -> {f['fix']}")
    if report["errors"]:
        sys.exit(1)


def _quiz(args: argparse.Namespace) -> None:
    """Write a quiz from the indexed documents (the Testing page's "Write a quiz from your documents")."""
    from finetune_studio.cli.commands.dataset import resolve_project
    from finetune_studio.data.prep.preference import NoGenerator
    from finetune_studio.testing import brain_quiz as bq

    if not args.project:
        _die("--project is required")
    try:
        print(json.dumps(bq.build_brain_quiz(resolve_project(args.project)["id"], questions=args.questions), indent=2))
    except (bq.NoIndex, NoGenerator, bq.NoUsableQuestions) as e:
        _die(str(e))


def _readiness(args: argparse.Namespace) -> None:
    """Is this project's knowledge ready to ship (the Package page's list)? Exits 1 when it is not."""
    from finetune_studio.cli.commands.dataset import resolve_project
    from finetune_studio.knowledge.readiness import readiness

    if not args.project:
        _die("--project is required")
    report = readiness(resolve_project(args.project)["id"])
    for step in report["steps"]:
        print(f"[{step['state']:7}] {step['label']}: {step['detail']}")
    print("Ready to package." if report["ready"] else "Not ready yet.")
    if not report["ready"]:
        sys.exit(1)


def _text_arg(value: str | None) -> str | None:
    """A flag's text, or the contents of a file when written @path."""
    if value is not None and value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def _read_pass(path: str | None) -> str | None:
    return Path(path).read_text(encoding="utf-8").strip() if path else None


def _bundle(args: argparse.Namespace) -> None:
    """Encrypted ``.ftsrag`` studio-to-studio copy of an index (the Smart Brain page's "Download / export")."""
    from finetune_studio.smart_brain import secure_bundle as sb

    corpus = _corpus(args)
    if not (corpus / "manifest.json").is_file():
        _die(f"no index at {corpus}: build it first (fts brain build)")
    out = Path(args.out or ".") / f"{args.name or corpus.name}{sb.BUNDLE_EXT}"
    try:
        path, generated = sb.export_secure_bundle(corpus, out, _read_pass(args.passphrase_file),
                                                  identity=_identity_for(args))
    except ValueError as e:
        _die(str(e))
    print(json.dumps({"path": str(path), "size": path.stat().st_size,
                      **({"passphrase": generated, "note": "shown once: store it now, it is not saved"}
                         if generated else {})}, indent=2))


def _import(args: argparse.Namespace) -> None:
    """Load a ``.ftsrag`` (or legacy .tar/.zip) bundle as a project's index (the page's "Import an index")."""
    from finetune_studio.smart_brain import BrainIndex
    from finetune_studio.smart_brain import secure_bundle as sb
    from finetune_studio.smart_brain.brain_container import ContainerError

    if not args.project:
        _die("--project is required: the bundle becomes that project's index")
    from finetune_studio.cli.commands.dataset import resolve_project
    from finetune_studio.knowledge import index

    pid = resolve_project(args.project)["id"]
    src = Path(args.bundle)
    if not src.is_file():
        _die(f"no such file: {src}")
    target = BrainIndex(_corpus(args))
    if target.exists() and not args.replace:
        _die("this project already has an index: add --replace to replace it with the bundle's")
    from finetune_studio.smart_brain import package_import

    opened = None
    try:
        if not src.name.lower().endswith(sb.BUNDLE_EXT) and package_import.is_server_package(src):
            opened = package_import.import_package(pid, src, _read_pass(args.passphrase_file), overwrite=True)
        elif src.name.lower().endswith(sb.BUNDLE_EXT):
            passphrase = _read_pass(args.passphrase_file)
            if not passphrase:
                _die("this bundle is encrypted: give --passphrase-file")
            sb.import_secure_bundle(src, passphrase, target.dir, overwrite=True)
        else:
            target.import_bundle(src, overwrite=True)
    except (ContainerError, ValueError, FileNotFoundError) as e:
        _die(str(e))
    stats = index.register_import(pid, bundle=src.name)
    from finetune_studio.smart_brain import identity as brain_identity

    brain_identity.adopt_imported(pid, target.dir)
    print(json.dumps({**stats, **({"package": opened} if opened else {})}, indent=2))


_HANDLERS = {"build": _build, "search": _search, "ask": _ask, "eval": _eval, "rebuild-vectors": _rebuild_vectors,
             "info": _info, "export": _export, "bundle": _bundle, "import": _import,
             "identity": _identity, "lint": _lint, "pictures": _pictures, "quiz": _quiz, "readiness": _readiness}


def cmd_brain(args: argparse.Namespace) -> None:
    handler = _HANDLERS.get(args.brain_command or "")
    if handler is None:
        print("Usage: fts brain {build,search,ask,eval,rebuild-vectors,info,export,bundle,import} "
              "(--project P | --corpus DIR) ...",
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
    sp.add_argument("--follow-up", type=int, default=1, choices=range(4), metavar="N",
                    help="follow-up searches the model may ask for before answering (0-3, default 1; 0 = off)")
    sp.add_argument("--no-retry-missing", dest="retry_missing", action="store_false",
                    help="do not search again when the answer says part of the question is not stated")
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
    sp.add_argument("--accept-superseded", action="store_true",
                    help="package documents the knowledge check reports as out of date without accepting each "
                         "finding (the web export refuses them until each is accepted)")
    sp.add_argument("--semantic", action=bo, default=None, help="meaning search over the vectors (default on)")
    sp.add_argument("--keyword", action=bo, default=None, help="word search over the BM25 index (default on)")
    sp.add_argument("--reranker", action=bo, default=None, help="cross-encoder rescoring (needs --models)")
    sp.add_argument("--answer", action=bo, default=None,
                    help="answer generation: the server's /ask and rag_answer call an LLM (default off)")
    sp.add_argument("--llm-base-url", help="OpenAI-compatible endpoint for answers, e.g. http://127.0.0.1:11434/v1")
    sp.add_argument("--llm-model", help="model name for answers")
    sp.add_argument("--llm-provider", choices=["openai", "anthropic"],
                    help="API kind of the answer endpoint: openai (any OpenAI-compatible) or anthropic (Claude)")
    sp.add_argument("--llm-connection",
                    help="copy the answer endpoint, model and headers from a Settings → AI assistant connection id "
                         "(never its key: set RAG_LLM_API_KEY where the server runs)")
    sp.add_argument("--models", action=bo, default=None,
                    help="bundle the embedder (+ reranker) for offline use (~2 GB, not encrypted)")
    sp.add_argument("--encrypt", action=bo, default=None, help="encrypt the corpus (default on)")
    sp.add_argument("--new-version-of", metavar="PACKAGE_ID",
                    help="a new version of a package in this project's history: same id, next version, change report")
    sp.add_argument("--base", metavar="ARCHIVE",
                    help="a new version of this exported package archive (add --base-passphrase-file if encrypted)")
    sp.add_argument("--base-passphrase-file", help="passphrase of the --base archive")
    sp.add_argument("--version", help="version to give the package (default: 1.0.0, or the base's next minor)")
    sp.add_argument("--include", action="append", metavar="PREFIX",
                    help="only documents whose path (folder/file name) starts with PREFIX, e.g. public/ (repeatable)")
    sp.add_argument("--exclude", action="append", metavar="PREFIX",
                    help="leave out documents whose path starts with PREFIX (repeatable)")
    sp.add_argument("--tag", action="append", metavar="TAG", help="only documents with this file tag (repeatable)")
    sp.add_argument("--hide-file-names", dest="hide_file_names", action=bo, default=None,
                    help="a public bot: answers name no source files; /ask returns sources only when asked "
                         '("sources": true) (default off)')
    sp.add_argument("--originals", action=bo, default=None,
                    help="ship the original uploaded files (encrypted with the corpus); anyone who can reach the "
                         "server can download them: leave off for a public bot (default off)")
    sp.add_argument("--passphrase-file", help="file holding the passphrase (default: generate one, printed once)")
    sp.add_argument("--format", choices=["tar.gz", "tar", "zip"])
    sp.add_argument("--top-k", type=int)
    sp.add_argument("--host")
    sp.add_argument("--port", type=int)
    sp.add_argument("--device")
    sp.add_argument("--name", help="package title (default: the project name)")
    sp.add_argument("--out", help="output directory (default: output/projects/<id>/brain-packages)")

    sp = target(brain.add_parser("identity", help="Show or set who the Smart Brain speaks for and how it answers"))
    sp.add_argument("--name", help="who it is")
    sp.add_argument("--about", help="about you or your company (text, or @file)")
    sp.add_argument("--personality", help="how it sounds (text, or @file)")
    sp.add_argument("--answer-rules", help="how it should answer (text, or @file)")

    sp = target(brain.add_parser("lint", help="Check the documents: unreadable, duplicate, out of date, conflicting, secrets"))
    sp.add_argument("--json", action="store_true", help="the full report as JSON")

    sp = target(brain.add_parser("pictures", help="Pictures only OCR read; --read them with the Vision reader"))
    sp.add_argument("--read", action="store_true", help="start the Reading pictures job")
    sp.add_argument("--wait", action="store_true", help="with --read: wait until it ends")
    sp.add_argument("--timeout", type=float, default=3600.0, help="--wait limit in seconds (default 3600)")
    sp.add_argument("--json", action="store_true", help="the full status as JSON")

    sp = target(brain.add_parser("quiz", help="Write a test quiz from the indexed documents (uses the AI assistant)"))
    sp.add_argument("--questions", type=int, default=40, help="how many to attempt (default 40)")

    target(brain.add_parser("readiness", help="Is this knowledge ready to ship? Each step's state"))

    sp = target(brain.add_parser("bundle", help="Encrypted .ftsrag copy of the index for another studio"))
    sp.add_argument("--passphrase-file", help="file holding the passphrase (default: generate one, printed once)")
    sp.add_argument("--name", help="file name without extension (default: the index directory name)")
    sp.add_argument("--out", help="output directory (default: the current one)")

    sp = target(brain.add_parser("import", help="Load a .ftsrag (or legacy .tar/.zip) bundle as a project's index"))
    sp.add_argument("bundle", help="the bundle file (.ftsrag), or an exported Smart Brain package (.tar.gz/.zip): "
                                   "its index and original files become this project's starting point")
    sp.add_argument("--passphrase-file", help="file holding the passphrase (encrypted .ftsrag bundles)")
    sp.add_argument("--replace", action="store_true", help="replace the project's existing index")
