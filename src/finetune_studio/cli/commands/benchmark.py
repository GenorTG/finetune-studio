"""`fts benchmark` — run industry-standard benchmarks (MMLU, HellaSwag, etc.)."""
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
