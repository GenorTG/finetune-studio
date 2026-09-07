"""`fts suite` — run a saved JSON test suite against a model."""
from __future__ import annotations

import json
import os
import sys


def cmd_suite(args) -> None:
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
