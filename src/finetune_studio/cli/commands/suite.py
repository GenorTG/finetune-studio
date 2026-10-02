"""`fts suite` — run a saved JSON test suite against a model."""
from __future__ import annotations

import json
import os
import sys


def _result_row(r) -> dict:
    return {
        "name": r.case_name,
        "category": r.category,
        "verdict": r.verdict,
        "judge": r.judge,
        "judge_reasoning": r.judge_reasoning,
        "response": r.model_answer,
        "time_ms": r.time_ms,
        "error": r.error,
    }


def cmd_suite(args) -> None:
    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.suite import (
        apply_heuristic_judging,
        load_test_suite,
        run_suite,
        score_results,
    )

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.suite):
        print(f"Error: Suite not found: {args.suite}")
        sys.exit(1)

    engine = InferenceEngine()
    print(f"Loading {args.model}...")
    engine.load(args.model)
    try:
        cases = load_test_suite(args.suite)
        print(f"Running {len(cases)} test cases...\n")
        results = run_suite(engine, cases, max_tokens=args.max_tokens)
    finally:
        engine.unload()

    # run_suite only collects transcripts; without a judging pass every case has
    # an empty verdict and any "pass rate" would be a fabricated 0%.
    apply_heuristic_judging(results)
    scores = score_results(results)
    # A score exists only if at least one case received a verdict.
    scored = scores["judged"] > 0
    unjudged = scores["unjudged"]

    if args.json:
        print(json.dumps({
            "results": [_result_row(r) for r in results],
            "scores": scores if scored else None,
            "scores_unavailable": (
                "" if scored else f"no case was judged ({unjudged}/{scores['total']} unjudged)"
            ),
        }, indent=2))
    else:
        for r in results:
            icon = {"pass": "✅", "partial": "🟡", "fail": "❌"}.get(r.verdict, "❔")
            print(f"{icon} {r.case_name} [{r.verdict or 'unjudged'}] ({r.time_ms}ms)")
            if r.error:
                print(f"   Error: {r.error}")
            resp = r.model_answer
            print(f"   {resp[:120]}{'...' if len(resp) > 120 else ''}\n")

        print(f"{'='*50}")
        if scored:
            print(f"Judge: heuristic ({scores['judged']}/{scores['total']} judged)")
            print(f"Pass rate: {scores['pass_rate']}% ({scores['passed']}/{scores['total']})")
        else:
            print(f"Error: no score — {unjudged}/{scores['total']} cases unjudged")
        if scored and unjudged:
            print(f"Warning: {unjudged} case(s) unjudged (errored or no verdict); counted as not passed")

    if unjudged:
        if args.json:
            print(f"Error: {unjudged}/{scores['total']} cases unjudged", file=sys.stderr)
        sys.exit(1)
