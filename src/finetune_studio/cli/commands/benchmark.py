"""`fts benchmark` — run official HuggingFace benchmarks (MMLU, GSM8K, HellaSwag)."""
from __future__ import annotations

import json as json_mod
import os
import sys


def cmd_benchmark(args) -> None:
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
        benchmarks = ["mmlu", "hellaswag", "gsm8k"]
    else:
        benchmarks = [b.strip() for b in args.suite.split(",")]

    full_run = bool(getattr(args, "full_run", False))
    num_samples = None if full_run else args.num_samples
    print(
        f"\nRunning {len(benchmarks)} real benchmarks "
        f"({'full split' if full_run else f'{num_samples} samples each'})..."
    )
    print("This may take a while depending on model speed.\n")

    result = suite.run_all(
        engine,
        num_samples=num_samples,
        benchmarks=benchmarks,
        full_run=full_run,
    )

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS (real / HuggingFace)")
    print(f"{'='*60}")
    print(f"Model: {os.path.basename(args.model)}")
    print(f"Samples per benchmark: {'full' if full_run else args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print()

    for name, data in result["benchmarks"].items():
        if "error" in data:
            print(f"  {name}: ERROR - {data['error']}")
        else:
            print(f"  {name}: {data['accuracy']}% ({data['correct']}/{data['total']})")
            meta = data.get("metadata") or {}
            if meta:
                print(
                    f"       dataset={meta.get('dataset_id')} "
                    f"config={meta.get('dataset_config')} split={meta.get('split')} "
                    f"scoring={meta.get('scoring_method')}"
                )

    summary = result["summary"]
    print(
        f"\nOverall: {summary['total_correct']}/{summary['total_questions']} "
        f"= {summary['overall_accuracy']}%"
    )

    if args.json:
        print(f"\n{json_mod.dumps(result, indent=2)}")

    if args.report:
        with open(args.report, "w") as f:
            json_mod.dump(result, f, indent=2)
        print(f"\nReport saved to: {args.report}")

    engine.unload()
