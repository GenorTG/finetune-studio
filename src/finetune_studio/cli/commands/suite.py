"""`fts suite` — ask a model every question of a saved JSON test suite and record the raw answers.

Like the Testing page, this only records what the model was asked, what it answered and the answer key. It does
not score. Pass ``--judge [PROVIDER_ID]`` to have an AI judge (any provider row; default: the helper seat) read the
saved answers afterwards, or ``--out FILE`` to keep the transcripts for later judging or review in the UI.
"""
from __future__ import annotations

import json
import os
import sys


def _result_row(r, verdict: dict | None = None) -> dict:
    row = {
        "name": r.case_name,
        "category": r.category,
        "question": r.question,
        "correct_answer": r.correct_answer,
        "response": r.model_answer,
        "transcript": r.transcript,
        "time_ms": r.time_ms,
        "error": r.error,
    }
    if verdict is not None:
        row["judge"] = verdict
    return row


def load_cases_or_exit(path: str) -> list:
    """Load a suite file; malformed JSON/shape or zero usable cases is a clean exit 1."""
    from finetune_studio.testing.suite import load_test_suite

    try:
        cases = load_test_suite(path)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"Error: invalid suite {path}: {exc}")
        sys.exit(1)
    if not cases:
        print(f"Error: suite {path} has no usable cases (need 'question' or 'messages' per case)")
        sys.exit(1)
    return cases


def _judge_all(results: list, provider_id: str) -> tuple[list[dict | None], str]:
    """Judge ``results`` with a provider row. Returns (per-case judge dicts, judge label)."""
    from finetune_studio.testing.judge import (
        JudgeCase,
        default_judge_provider_id,
        judge_many,
        open_judge,
    )

    provider_id = provider_id or default_judge_provider_id()
    out: list[dict | None] = [None] * len(results)
    with open_judge(provider_id) as judge:
        items = [
            (str(i), JudgeCase(r.question, r.correct_answer, r.model_answer, list(r.keywords), r.expect_abstain))
            for i, r in enumerate(results) if not (r.error and not r.model_answer)
        ]

        def record(key: str, res) -> None:
            out[int(key)] = {"verdict": res.verdict, "reasoning": res.reasoning, "error": res.error,
                             "model": judge.model}

        judge_many(judge, items, record)
        return out, judge.label


def cmd_suite(args) -> None:
    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.suite import run_suite

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.suite):
        print(f"Error: Suite not found: {args.suite}")
        sys.exit(1)

    cases = load_cases_or_exit(args.suite)  # fail before the slow model load

    log = sys.stderr if args.json else sys.stdout
    engine = InferenceEngine()
    print(f"Loading {args.model}...", file=log)
    engine.load(args.model)
    try:
        print(f"Running {len(cases)} test cases...", file=log)
        results = run_suite(engine, cases, max_tokens=args.max_tokens)
    finally:
        engine.unload()

    verdicts: list[dict | None] = [None] * len(results)
    judge_label = ""
    if args.judge is not None:
        try:
            verdicts, judge_label = _judge_all(results, args.judge)
        except Exception as exc:  # noqa: BLE001 - JudgeUnavailable and provider errors alike
            print(f"Error: judging failed: {exc}", file=sys.stderr)
            sys.exit(1)

    rows = [_result_row(r, v) for r, v in zip(results, verdicts, strict=True)]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    errors = sum(1 for r in results if r.error)
    judged = [v for v in verdicts if v and v["verdict"]]
    if args.json:
        print(json.dumps({"results": rows, "judged": len(judged), "judge": judge_label or None}, indent=2))
    else:
        for r, v in zip(results, verdicts, strict=True):
            tag = (v or {}).get("verdict") or "unjudged"
            print(f"[{tag}] {r.case_name} ({r.time_ms}ms)")
            if r.error:
                print(f"   Error: {r.error}")
            print(f"   Q: {r.question[:120]}")
            print(f"   key: {r.correct_answer[:120]}")
            resp = r.model_answer
            print(f"   A: {resp[:120]}{'...' if len(resp) > 120 else ''}")
            if v and v.get("reasoning"):
                print(f"   judge: {v['reasoning'][:160]}")
            print()
        print("=" * 50)
        if args.judge is None:
            print(f"{len(results)} answers recorded, none judged (use --judge to judge them, or review in the Testing page).")
        else:
            counts = {k: sum(1 for v in judged if v["verdict"] == k) for k in ("pass", "partial", "fail")}
            print(f"Judge {judge_label}: {counts['pass']} pass, {counts['partial']} partial, {counts['fail']} fail, "
                  f"{len(results) - len(judged)} not judged (of {len(results)})")
        if args.out:
            print(f"Transcripts written to {args.out}")

    if errors or (args.judge is not None and len(judged) < len(results) - errors):
        sys.exit(1)
