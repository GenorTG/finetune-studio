"""`fts compare` — run the same test suite against multiple models."""
from __future__ import annotations

import json as json_mod
import os
import sys


def cmd_compare(args) -> None:
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
