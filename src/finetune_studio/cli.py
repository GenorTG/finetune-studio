"""CLI entry point for finetune-studio. Definition for the `fts` command.

WHAT THIS FILE DOES
==================
This is the main CLI script. It defines all subcommands:
  - fts models: list available models
  - fts train: start a training run
  - fts benchmark: run industry benchmarks
  - fts compare: side-by-side model comparison
  - fts convert: convert model formats (HF → GGUF)
  - fts test: run test suites
  - fts validate: validate training data
  - fts optimize: suggest optimal training config
  - fts rag: RAG operations (ingest, query, list)
  - fts webui: launch the web UI on port 7860

KEY CONCEPTS
============
- Typer: a Python CLI library. Lets you define commands as functions
  with type annotations (Typer auto-generates --help).
- Subcommands: each function decorated with @app.command() becomes a subcommand.
- Callbacks: code that runs before any subcommand (e.g., loading config).
"""

"""Finetune Studio CLI — manage models, training, testing, RAG, and comparisons."""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="finetune-studio",
        description="Finetune Studio — model training & RAG CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── models ──
    p_models = sub.add_parser("models", help="List discovered models")
    p_models.add_argument("--dirs", nargs="*", help="Extra directories to scan")
    p_models.add_argument("--json", action="store_true", help="Output as JSON")

    # ── train ──
    p_train = sub.add_parser("train", help="Start training")
    p_train.add_argument("model", help="Path to base model")
    p_train.add_argument("data", help="Path to training data (JSONL)")
    p_train.add_argument("-o", "--output", default="output", help="Output directory")
    p_train.add_argument("--lr", type=float, default=8e-5, help="Learning rate")
    p_train.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    p_train.add_argument("--batch", type=int, default=2, help="Batch size")
    p_train.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    p_train.add_argument("--max-seq", type=int, default=2048, help="Max sequence length")
    p_train.add_argument("--system-prompt", default="", help="System prompt")
    p_train.add_argument("--no-unsloth", action="store_true", help="Use standard transformers")

    # ── test ──
    p_test = sub.add_parser("test", help="Test a model interactively")
    p_test.add_argument("model", help="Path to model")
    p_test.add_argument("--max-tokens", type=int, default=512)
    p_test.add_argument("--temperature", type=float, default=0.7)

    # ── suite ──
    p_suite = sub.add_parser("suite", help="Run a test suite")
    p_suite.add_argument("model", help="Path to model")
    p_suite.add_argument("suite", help="Path to test suite JSON")
    p_suite.add_argument("--max-tokens", type=int, default=512)
    p_suite.add_argument("--json", action="store_true", help="Output as JSON")

    # ── validate ──
    p_val = sub.add_parser("validate", help="Validate training data")
    p_val.add_argument("files", nargs="+", help="Files to validate")

    # ── convert ──
    p_conv = sub.add_parser("convert", help="Convert data formats")
    p_conv.add_argument("source", help="Source file")
    p_conv.add_argument("target_format", choices=["jsonl", "json", "csv"])
    p_conv.add_argument("-o", "--output", help="Output path")
    p_conv.add_argument("--system-prompt", default="")

    # ── webui ──
    p_web = sub.add_parser("webui", help="Start the WebUI server")
    p_web.add_argument("--host", default="0.0.0.0")
    p_web.add_argument("--port", type=int, default=7860)
    p_web.add_argument("--reload", action="store_true")

    # ── rag ──
    p_rag = sub.add_parser("rag", help="RAG operations")
    rag_sub = p_rag.add_subparsers(dest="rag_command")

    p_rag_ingest = rag_sub.add_parser("ingest", help="Ingest documents into RAG store")
    p_rag_ingest.add_argument("path", help="File or directory to ingest")
    p_rag_ingest.add_argument("--chunk-size", type=int, default=512)
    p_rag_ingest.add_argument("--overlap", type=int, default=50)
    p_rag_ingest.add_argument("--store", default="data/rag_store", help="Store path")

    p_rag_query = rag_sub.add_parser("query", help="Query RAG store")
    p_rag_query.add_argument("question", help="Question to ask")
    p_rag_query.add_argument("--top-k", type=int, default=5)
    p_rag_query.add_argument("--store", default="data/rag_store")
    p_rag_query.add_argument("--json", action="store_true")

    p_rag_list = rag_sub.add_parser("list", help="List indexed documents")
    p_rag_list.add_argument("--store", default="data/rag_store")
    p_rag_list.add_argument("--json", action="store_true")

    p_rag_remove = rag_sub.add_parser("remove", help="Remove document from RAG store")
    p_rag_remove.add_argument("document_id", help="Document ID to remove")
    p_rag_remove.add_argument("--store", default="data/rag_store")

    p_rag_stats = rag_sub.add_parser("stats", help="RAG store statistics")
    p_rag_stats.add_argument("--store", default="data/rag_store")

    p_rag_clear = rag_sub.add_parser("clear", help="Clear RAG store")
    p_rag_clear.add_argument("--store", default="data/rag_store")
    p_rag_clear.add_argument("--confirm", action="store_true")

    # ── compare ──
    p_cmp = sub.add_parser("compare", help="Compare models on test suite")
    p_cmp.add_argument("--models", nargs="+", required=True, help="Model paths (name=path format)")
    p_cmp.add_argument("suite", help="Test suite JSON file")
    p_cmp.add_argument("--max-tokens", type=int, default=512)
    p_cmp.add_argument("--temperature", type=float, default=0.7)
    p_cmp.add_argument("--json", action="store_true", help="Output as JSON")
    p_cmp.add_argument("--report", help="Save report to file")
    p_cmp.add_argument("--real", action="store_true", default=True, help="Use real HuggingFace datasets")

    # ── benchmark ──
    p_bench = sub.add_parser("benchmark", help="Run industry-standard benchmarks")
    p_bench.add_argument("model", help="Path to model (GGUF or safetensors)")
    p_bench.add_argument("--suite", default="all", help="Benchmark suite: all, mmlu, hellaswag, arc, truthfulqa, gsm8k, winogrande (or comma-separated)")
    p_bench.add_argument("--num-samples", type=int, default=100, help="Number of samples per benchmark (default: 100)")
    p_bench.add_argument("--max-tokens", type=int, default=10)
    p_bench.add_argument("--temperature", type=float, default=0.0)
    p_bench.add_argument("--json", action="store_true", help="Output as JSON")
    p_bench.add_argument("--report", help="Save report to file")
    p_bench.add_argument("--real", action="store_true", default=True, help="Use real HuggingFace datasets")

    # ── analyze (data quality) ──
    p_analyze = sub.add_parser("analyze", help="Analyze training data quality")
    p_analyze.add_argument("data", help="Path to training data (JSONL)")
    p_analyze.add_argument("--fix", action="store_true", help="Auto-fix issues")
    p_analyze.add_argument("--output", help="Output fixed data to file")
    p_analyze.add_argument("--json", action="store_true")

    # ── augment (data augmentation) ──
    p_aug = sub.add_parser("augment", help="Augment training data to fix weaknesses")
    p_aug.add_argument("data", help="Path to training data (JSONL)")
    p_aug.add_argument("--output", required=True, help="Output augmented data")
    p_aug.add_argument("--type", default="all", help="Augmentation type: knowledge, refusal, language, hallucination, persona, all")
    p_aug.add_argument("--count", type=int, default=50, help="Number of examples to generate per type")
    p_aug.add_argument("--ratio", type=float, default=0.7, help="Persona ratio for data mixing (0.0-1.0)")

    # ── optimize (config recommendation) ──
    p_opt = sub.add_parser("optimize", help="Get training config recommendations")
    p_opt.add_argument("data", help="Path to training data (JSONL)")
    p_opt.add_argument("--lr", type=float, help="Current learning rate")
    p_opt.add_argument("--epochs", type=int, help="Current epochs")
    p_opt.add_argument("--lora-rank", type=int, help="Current LoRA rank")
    p_opt.add_argument("--json", action="store_true")

    # ── validate (hallucination check) ──
    p_val_h = sub.add_parser("validate-hallucination", help="Check training data for hallucination risks")
    p_val_h.add_argument("data", help="Path to training data (JSONL)")
    p_val_h.add_argument("--json", action="store_true")

    # ── rag-query (RAG-enhanced inference) ──
    p_rag_test = sub.add_parser("rag-test", help="Test model with RAG context")
    p_rag_test.add_argument("model", help="Path to model")
    p_rag_test.add_argument("question", help="Question to ask")
    p_rag_test.add_argument("--store", default="data/rag_store")
    p_rag_test.add_argument("--top-k", type=int, default=5)
    p_rag_test.add_argument("--max-tokens", type=int, default=512)
    p_rag_test.add_argument("--system-prompt", default="")
    p_rag_test.add_argument("--json", action="store_true")

    # ── vram ──
    p_vram = sub.add_parser("vram", help="GPU VRAM profiling & recommendations")
    vram_sub = p_vram.add_subparsers(dest="vram_command")

    p_vram_report = vram_sub.add_parser("report", help="Generate VRAM report for current GPU")
    p_vram_report.add_argument("--output", help="Save report to file")
    p_vram_report.add_argument("--vram", type=float, help="Override available VRAM (GB)")

    p_vram_check = vram_sub.add_parser("check", help="Check if a model fits in VRAM")
    p_vram_check.add_argument("model", help="Model name or size (e.g. 'qwen2.5-7b' or '7b')")
    p_vram_check.add_argument("--method", default="qlora", choices=["qlora", "lora", "full_ft"])
    p_vram_check.add_argument("--batch", type=int, default=2)
    p_vram_check.add_argument("--seq", type=int, default=2048)
    p_vram_check.add_argument("--lora-rank", type=int, default=64)
    p_vram_check.add_argument("--vram", type=float, help="Override available VRAM (GB)")
    p_vram_check.add_argument("--json", action="store_true")

    p_vram_profile = vram_sub.add_parser("profile", help="Profile actual VRAM usage (requires model)")
    p_vram_profile.add_argument("model", help="HuggingFace model ID or local path")
    p_vram_profile.add_argument("--method", default="qlora", choices=["qlora", "lora"])
    p_vram_profile.add_argument("--batch", type=int, default=1)
    p_vram_profile.add_argument("--seq", type=int, default=512)
    p_vram_profile.add_argument("--lora-rank", type=int, default=16)
    p_vram_profile.add_argument("--steps", type=int, default=5)
    p_vram_profile.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "models": cmd_models,
        "train": cmd_train,
        "test": cmd_test,
        "suite": cmd_suite,
        "validate": cmd_validate,
        "convert": cmd_convert,
        "webui": cmd_webui,
        "rag": cmd_rag,
        "compare": cmd_compare,
        "benchmark": cmd_benchmark,
        "analyze": cmd_analyze,
        "augment": cmd_augment,
        "optimize": cmd_optimize,
        "validate-hallucination": cmd_validate_hallucination,
        "rag-test": cmd_rag_test,
        "vram": cmd_vram,
    }

    commands[args.command](args)


def cmd_models(args):
    from finetune_studio.config import settings
    from finetune_studio.models.registry import scan_models
    dirs = settings.model_dirs + (args.dirs or [])
    models = scan_models(dirs)
    if args.json:
        print(json.dumps([{
            "name": m.name, "path": m.path, "format": m.format,
            "size_gb": m.size_gb, "architecture": m.architecture,
        } for m in models], indent=2))
    else:
        if not models:
            print("No models found.")
            return
        print(f"{'Name':<50} {'Format':<12} {'Size':<10} {'Arch'}")
        print("-" * 90)
        for m in models:
            print(f"{m.name:<50} {m.format:<12} {m.size_gb:<10} {m.architecture}")


def cmd_train(args):
    import time

    from finetune_studio.training.data import load_jsonl
    from finetune_studio.training.engine import TrainingConfig, TrainingEngine

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"Error: Data not found: {args.data}")
        sys.exit(1)

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} examples from {args.data}")

    config = TrainingConfig(
        model_path=args.model, output_dir=args.output,
        lora_rank=args.lora_rank, learning_rate=args.lr,
        num_epochs=args.epochs, batch_size=args.batch,
        max_seq_length=args.max_seq, unsloth=not args.no_unsloth,
    )

    engine = TrainingEngine()

    def on_progress(state):
        if state.status == "training":
            pct = (state.current_step / max(state.total_steps, 1)) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            sys.stdout.write(f"\r[{bar}] {pct:.0f}% | Step {state.current_step}/{state.total_steps} | Loss: {state.loss} | ETA: {state.eta}s")
            sys.stdout.flush()
        elif state.status == "done":
            print(f"\n\nTraining complete! Output: {args.output}")
        elif state.status == "error":
            print(f"\n\nError: {state.error}")
        elif state.status in ("loading", "saving"):
            print(f"  {state.message}")

    engine.on_update(on_progress)
    engine.start(config, data, args.system_prompt)

    while engine.state.status in ("loading", "training", "saving"):
        time.sleep(1)

    sys.exit(0 if engine.state.status == "done" else 1)


def cmd_test(args):
    from finetune_studio.testing.inference import InferenceEngine

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    engine = InferenceEngine()
    print(f"Loading {args.model}...")
    engine.load(args.model)
    fmt = "GGUF" if engine.is_gguf else "safetensors"
    print(f"Model loaded ({fmt}). Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = engine.generate(
            [{"role": "user", "content": user_input}],
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        print(f"AI: {response}\n")

    engine.unload()
    print("Model unloaded.")


def cmd_suite(args):
    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.suite import load_test_suite, run_suite, score_results

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.suite):
        print(f"Error: Suite not found: {args.suite}")
        sys.exit(1)

    engine = InferenceEngine()
    print(f"Loading {args.model}...")
    engine.load(args.model)

    cases = load_test_suite(args.suite)
    print(f"Running {len(cases)} test cases...\n")

    results = run_suite(engine, cases, max_tokens=args.max_tokens)
    scores = score_results(results)

    if args.json:
        print(json.dumps({
            "results": [{"name": r.test_name, "passed": r.passed, "response": r.response,
                         "time_ms": r.time_ms, "error": r.error} for r in results],
            "scores": scores,
        }, indent=2))
    else:
        for r in results:
            icon = "✅" if r.passed else "❌"
            print(f"{icon} {r.test_name} ({r.time_ms}ms)")
            if r.error:
                print(f"   Error: {r.error}")
            print(f"   {r.response[:120]}{'...' if len(r.response) > 120 else ''}\n")

        print(f"{'='*50}")
        print(f"Pass rate: {scores['pass_rate']}% ({scores['passed']}/{scores['total']})")

    engine.unload()


def cmd_validate(args):
    from finetune_studio.data.validator import validate_file
    for f in args.files:
        report = validate_file(f)
        icon = "✅" if report["valid"] else "❌"
        print(f"{icon} {report['name']}: {report['stats']}")
        for e in report["errors"]:
            print(f"   ERROR: {e}")
        for w in report["warnings"]:
            print(f"   WARN: {w}")


def cmd_convert(args):
    from pathlib import Path

    from finetune_studio.data.converter import (
        csv_to_jsonl,
        json_to_jsonl,
        jsonl_to_json,
    )

    src = Path(args.source)
    if not src.exists():
        print(f"Error: File not found: {src}")
        sys.exit(1)

    target = args.output or str(src.with_suffix(f".{args.target_format}"))

    if src.suffix == ".jsonl" and args.target_format == "json":
        jsonl_to_json(str(src), target)
    elif src.suffix == ".json" and args.target_format == "jsonl":
        json_to_jsonl(str(src), target)
    elif src.suffix == ".csv" and args.target_format == "jsonl":
        csv_to_jsonl(str(src), target, system_prompt=args.system_prompt)
    else:
        print(f"Error: Cannot convert {src.suffix} -> .{args.target_format}")
        sys.exit(1)

    print(f"Converted: {src} -> {target}")


def cmd_webui(args):
    import uvicorn
    uvicorn.run(
        "finetune_studio.webui.app:app",
        host=args.host, port=args.port, reload=args.reload,
    )


def cmd_rag(args):
    if args.rag_command is None:
        print("Usage: finetune-studio rag {ingest,query,list,remove,stats,clear}")
        sys.exit(1)

    from finetune_studio.rag.manager import RAGManager
    manager = RAGManager(args.store)

    if args.rag_command == "ingest":
        if os.path.isdir(args.path):
            result = manager.ingest_directory(args.path, args.chunk_size, args.overlap)
        else:
            result = manager.ingest_file(args.path, args.chunk_size, args.overlap)
        print(json.dumps(result, indent=2))

    elif args.rag_command == "query":
        results = manager.store.search(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps([{
                "text": r.text, "score": round(r.score, 3),
                "source": r.metadata.get("source", "unknown"),
                "document_id": r.document_id,
            } for r in results], indent=2))
        else:
            if not results:
                print("No results found.")
            for i, r in enumerate(results):
                print(f"\n--- Result {i+1} (score: {r.score:.3f}) ---")
                print(f"Source: {r.metadata.get('source', 'unknown')}")
                print(f"Doc: {r.document_id}")
                print(f"{r.text[:300]}{'...' if len(r.text) > 300 else ''}")

    elif args.rag_command == "list":
        docs = manager.list_documents()
        if args.json:
            print(json.dumps(docs, indent=2))
        else:
            if not docs:
                print("No documents in RAG store.")
            for doc in docs:
                print(f"  {doc['document_id']}: {doc['chunk_count']} chunks, sources: {doc['sources']}")

    elif args.rag_command == "remove":
        result = manager.remove_document(args.document_id)
        print(json.dumps(result, indent=2))

    elif args.rag_command == "stats":
        stats = manager.stats()
        print(json.dumps(stats, indent=2))

    elif args.rag_command == "clear":
        if not args.confirm:
            print("This will clear ALL RAG data. Use --confirm to proceed.")
            sys.exit(1)
        manager.clear()
        print("RAG store cleared.")


def cmd_compare(args):
    import json as json_mod

    from finetune_studio.benchmarks.comparison import comparator
    from finetune_studio.testing.suite import load_test_suite

    if not os.path.exists(args.suite):
        print(f"Error: Suite not found: {args.suite}")
        sys.exit(1)

    # Parse models (format: name=path)
    for m in args.models:
        if "=" in m:
            name, path = m.split("=", 1)
        else:
            name = os.path.basename(m)
            path = m

        if not os.path.exists(path):
            print(f"Error: Model not found: {path}")
            sys.exit(1)

        print(f"Loading {name} from {path}...")
        comparator.load_model(name, path)

    if not comparator.engines:
        print("Error: No models loaded")
        sys.exit(1)

    test_suite = load_test_suite(args.suite)
    config = {"max_tokens": args.max_tokens, "temperature": args.temperature}

    print(f"\nRunning comparison on {len(test_suite)} tests...")
    result = comparator.run_comparison(test_suite, config)

    if args.json:
        print(json_mod.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        for model, stats in result["summary"].items():
            print(f"  {model}: {stats['accuracy']}% ({stats['passed']}/{stats['total']}) | avg {stats['avg_time_ms']}ms")

    if args.report:
        with open(args.report, "w") as f:
            json_mod.dump(result, f, indent=2)
        print(f"\nReport saved to: {args.report}")

    comparator.cleanup()


def cmd_benchmark(args):
    import json as json_mod

    from finetune_studio.benchmarks.real_benchmarks import RealBenchmarkSuite
    from finetune_studio.testing.inference import InferenceEngine

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    engine = InferenceEngine()
    print(f"Loading {args.model}...")
    engine.load(args.model)
    print("Model loaded!")

    suite = RealBenchmarkSuite()

    if args.suite == "all":
        benchmarks = ["mmlu", "hellaswag", "arc_challenge", "truthfulqa", "gsm8k", "winogrande"]
    else:
        benchmarks = [b.strip() for b in args.suite.split(",")]

    print(f"\nRunning {len(benchmarks)} benchmarks with {args.num_samples} samples each...")
    print("This may take a while depending on model speed.\n")

    result = suite.run_all(
        engine,
        num_samples=args.num_samples,
        benchmarks=benchmarks,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Model: {os.path.basename(args.model)}")
    print(f"Samples per benchmark: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print()

    for name, data in result["benchmarks"].items():
        if "error" in data:
            print(f"  {name}: ERROR - {data['error']}")
        else:
            print(f"  {name}: {data['accuracy']}% ({data['correct']}/{data['total']})")

    summary = result["summary"]
    print(f"\nOverall: {summary['total_correct']}/{summary['total_questions']} = {summary['overall_accuracy']}%")

    if args.json:
        print(f"\n{json_mod.dumps(result, indent=2)}")

    if args.report:
        with open(args.report, "w") as f:
            json_mod.dump(result, f, indent=2)
        print(f"\nReport saved to: {args.report}")

    engine.unload()


def cmd_analyze(args):
    import json as json_mod

    from finetune_studio.training.data_quality import (
        DataQualityAnalyzer,
        generate_fixes,
    )

    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(args.data)

    print(f"\n{'='*60}")
    print("DATA QUALITY REPORT")
    print(f"{'='*60}")
    print(f"File: {result['file']}")
    print(f"Total examples: {result['total_examples']}")
    print(f"Severity: {result['severity'].upper()}")
    print()

    if result['stats']:
        print("Statistics:")
        for k, v in result['stats'].items():
            print(f"  {k}: {v}")
        print()

    if result['issues']:
        print("Issues:")
        for issue in result['issues']:
            icon = "🔴" if issue['severity'] == "high" else "🟡" if issue['severity'] == "medium" else "🟢"
            print(f"  {icon} [{issue['severity'].upper()}] {issue['message']}")
        print()

        fixes = generate_fixes(result)
        if fixes:
            print("Suggested fixes:")
            for fix in fixes:
                print(f"  {fix['action']}: {fix['command']}")
    else:
        print("No issues found!")

    if args.json:
        print(f"\n{json_mod.dumps(result, indent=2)}")


def cmd_augment(args):
    import json as json_mod

    from finetune_studio.training.data_augmentation import DataAugmenter
    from finetune_studio.training.data_quality import DataQualityAnalyzer

    # Load existing data
    data = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json_mod.loads(line))
                except json_mod.JSONDecodeError:
                    pass

    print(f"Loaded {len(data)} examples from {args.data}")

    # Analyze weaknesses
    analyzer = DataQualityAnalyzer()
    analysis = analyzer.analyze(args.data)

    weaknesses = []
    for issue in analysis['issues']:
        if 'language' in issue.get('type', ''):
            weaknesses.append('language_balance')
        elif 'hallucination' in issue.get('type', ''):
            weaknesses.append('hallucination_guard')
        elif 'empty' in issue.get('type', ''):
            weaknesses.append('refusal')

    # Add default augmentations
    if args.type == 'all':
        weaknesses.extend(['knowledge', 'refusal'])
    else:
        weaknesses.extend(args.type.split(','))

    weaknesses = list(set(weaknesses))
    print(f"Augmenting for: {', '.join(weaknesses)}")

    # Augment
    augmenter = DataAugmenter()
    augmented = augmenter.augment_dataset(data, weaknesses)

    print(f"Augmented dataset: {len(data)} -> {len(augmented)} examples")

    # Save
    with open(args.output, 'w') as f:
        f.writelines(json_mod.dumps(item, ensure_ascii=False) + '\n' for item in augmented)

    print(f"Saved to {args.output}")


def cmd_optimize(args):
    import json as json_mod

    from finetune_studio.training.config_optimizer import TrainingConfigOptimizer

    # Load data
    data = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json_mod.loads(line))
                except json_mod.JSONDecodeError:
                    pass

    print(f"Analyzing {len(data)} examples...")

    # Build current config
    current = {}
    if args.lr:
        current['learning_rate'] = args.lr
    if args.epochs:
        current['num_epochs'] = args.epochs
    if args.lora_rank:
        current['lora_rank'] = args.lora_rank

    optimizer = TrainingConfigOptimizer()
    recommendations = optimizer.analyze_and_recommend(data, current)

    if args.json:
        print(json_mod.dumps([{'parameter': r.parameter, 'current': r.current_value,
                             'recommended': r.recommended_value, 'reason': r.reason,
                             'priority': r.priority} for r in recommendations], indent=2))
    else:
        print(optimizer.generate_report(recommendations))


def cmd_validate_hallucination(args):
    import json as json_mod

    from finetune_studio.training.hallucination_guard import TrainingDataValidator

    # Load data
    data = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json_mod.loads(line))
                except json_mod.JSONDecodeError:
                    pass

    print(f"Checking {len(data)} examples for hallucination risks...")

    validator = TrainingDataValidator()
    result = validator.validate_dataset(data)

    print(f"\nTotal risks: {result['total_risks']}")
    print(f"Risk types: {result['risk_types']}")
    print(f"Recommendation: {result['recommendation']}")

    if args.json:
        print(json_mod.dumps(result, indent=2))


def cmd_rag_test(args):
    from finetune_studio.rag.query import RAGConfig, RAGQuery
    from finetune_studio.rag.store import VectorStore
    from finetune_studio.testing.inference import InferenceEngine

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    engine = InferenceEngine()
    print(f"Loading {args.model}...")
    engine.load(args.model)

    store = VectorStore(args.store)
    rag = RAGQuery(store, RAGConfig(top_k=args.top_k))

    messages = rag.augment_prompt(args.question, args.system_prompt, args.top_k)
    response = engine.generate(messages, max_tokens=args.max_tokens)

    if args.json:
        sources = rag.retrieve(args.question, args.top_k)
        print(json.dumps({
            "response": response,
            "sources": [{"text": r.text[:200], "score": r.score, "source": r.metadata.get("source")} for r in sources],
        }, indent=2))
    else:
        print(f"\nAI: {response}")
        sources = rag.retrieve(args.question, args.top_k)
        if sources:
            print(f"\n--- Sources ({len(sources)} chunks) ---")
            for r in sources:
                print(f"  [{r.score:.3f}] {r.metadata.get('source', 'unknown')}")

    engine.unload()


def cmd_vram(args):
    from finetune_studio.training.vram_profiler import (
        GPUInfo, ProfileResult, generate_vram_report, estimate_vram,
        profile_training, recommend_for_model, recommend_config,
        MODEL_PRESETS,
    )

    if args.vram_command is None:
        print("Usage: fts vram {report|check|profile}")
        sys.exit(1)

    if args.vram_command == "report":
        vram = getattr(args, 'vram', None)
        report = generate_vram_report(available_vram_gb=vram, output_path=args.output)
        print(report)
        if args.output:
            print(f"\nSaved to {args.output}")

    elif args.vram_command == "check":
        vram = getattr(args, 'vram', None)
        model = args.model.lower().replace(" ", "-")

        # Try to find in presets
        matched_preset = None
        for key in MODEL_PRESETS:
            if model in key or key in model:
                matched_preset = key
                break

        # If no preset match, try parsing as a number (e.g. "7b" or "7")
        if matched_preset is None:
            try:
                size_b = float(model.replace("b", ""))
                est = estimate_vram(
                    model_size_b=size_b, method=args.method,
                    batch_size=args.batch, seq_length=args.seq,
                    lora_rank=args.lora_rank, available_vram_gb=vram,
                )
                if args.json:
                    print(json.dumps(est.to_dict(), indent=2))
                else:
                    _print_vram_check(model, est)
                return
            except ValueError:
                pass

        if matched_preset:
            configs = recommend_for_model(matched_preset, available_vram_gb=vram)
            if args.json:
                out = [{
                    "method": c.method, "lora_rank": c.lora_rank,
                    "batch_size": c.batch_size, "gradient_accumulation": c.gradient_accumulation,
                    "max_seq_length": c.max_seq_length, "estimated_vram_gb": c.estimated_vram_gb,
                    "fits": c.fits, "notes": c.notes,
                } for c in configs]
                print(json.dumps(out, indent=2))
            else:
                print(f"\nRecommendations for {matched_preset} ({MODEL_PRESETS[matched_preset]['params_b']}B):")
                print(f"{'Method':<10} {'Rank':<6} {'Batch':<6} {'GradAcc':<8} {'Seq':<6} {'VRAM':<8} {'Fits':<6} Notes")
                print("-" * 80)
                for c in configs:
                    fit = "✅" if c.fits else "❌"
                    print(f"{c.method:<10} {c.lora_rank:<6} {c.batch_size:<6} {c.gradient_accumulation:<8} {c.max_seq_length:<6} {c.estimated_vram_gb:<8.1f} {fit:<6} {c.notes}")
        else:
            print(f"Unknown model: {args.model}")
            print(f"Known models: {', '.join(sorted(MODEL_PRESETS.keys()))}")
            print("Or use a number like '7b' or '14'")
            sys.exit(1)

    elif args.vram_command == "profile":
        print(f"Profiling {args.model} with {args.method} (batch={args.batch}, seq={args.seq}, rank={args.lora_rank})...")
        print("This will download the model if not cached. Using synthetic data.")
        print()

        result = profile_training(
            model_path=args.model, method=args.method,
            batch_size=args.batch, seq_length=args.seq,
            lora_rank=args.lora_rank, num_steps=args.steps,
        )

        if args.json:
            print(json.dumps({
                "model": result.model_name, "method": result.method,
                "batch_size": result.batch_size, "seq_length": result.seq_length,
                "lora_rank": result.lora_rank, "peak_vram_gb": result.peak_vram_gb,
                "measured_model_gb": result.measured_model_gb,
                "measured_adapters_gb": result.measured_adapters_gb,
                "training_steps": result.training_steps,
                "step_time_s": result.step_time_s,
                "success": result.success, "error": result.error,
            }, indent=2))
        else:
            if result.success:
                print(f"✅ Profile complete")
                print(f"   Peak VRAM:    {result.peak_vram_gb:.2f} GB")
                print(f"   Model load:   {result.measured_model_gb:.2f} GB")
                print(f"   Adapters:     {result.measured_adapters_gb:.2f} GB")
                print(f"   Step time:    {result.step_time_s:.2f}s")
                print(f"   Steps run:    {result.training_steps}")
            else:
                print(f"❌ Profile failed: {result.error}")
                sys.exit(1)


def _print_vram_check(label: str, est):
    """Pretty-print a VRAM check result."""
    fit = "✅ FITS" if est.fits else "❌ TOO LARGE"
    print(f"\n{label} — {est.method.upper()}")
    print(f"  Model weights:    {est.model_weights_gb:.2f} GB")
    print(f"  Gradients:        {est.gradients_gb:.2f} GB")
    print(f"  Optimizer:        {est.optimizer_states_gb:.2f} GB")
    print(f"  Activations:      {est.activations_gb:.2f} GB")
    print(f"  ─────────────────────────")
    print(f"  Total estimated:  {est.total_gb:.2f} GB")
    print(f"  Available:        {est.available_gb:.2f} GB")
    print(f"  Headroom:         {est.headroom_gb:.2f} GB")
    print(f"  Result:           {fit}")
    if est.fits:
        print(f"  Safe batch size:  {est.max_batch_size}")
        print(f"  Safe seq length:  {est.max_seq_length}")


if __name__ == "__main__":
    main()
