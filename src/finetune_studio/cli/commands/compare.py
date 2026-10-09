"""`fts compare` — ask several models the same questions and show their answers side by side.

Like `fts suite` and the Compare tab, this only records what each model answered next to the answer key; it does
not score. Pass ``--judge [PROVIDER_ID]`` to have an AI judge (any provider row; default: the helper seat) read the
saved answers afterwards, or ``--report FILE`` to keep them for later judging or review.
"""
from __future__ import annotations

import json as json_mod
import os
import sys


def _parse_models(specs: list[str]) -> list[tuple[str, str]]:
    """``name=path`` or a bare path (named after its basename); exits 1 on a missing path or a repeated name."""
    models: list[tuple[str, str]] = []
    for m in specs:
        name, path = m.split("=", 1) if "=" in m else (os.path.basename(m.rstrip("/")) or m, m)
        if not os.path.exists(path):
            print(f"Error: Model not found: {path}")
            sys.exit(1)
        if any(name == n for n, _ in models):
            print(f"Error: model name '{name}' is used twice; give each model its own name=path")
            sys.exit(1)
        models.append((name, path))
    return models


def _judge_models(results: dict[str, list], provider_id: str) -> tuple[dict[str, list[dict | None]], str]:
    """Judge every model's answers with ONE loaded judge. Returns ({model: per-case judge dicts}, judge label)."""
    from finetune_studio.testing.judge import (
        JudgeCase,
        default_judge_provider_id,
        judge_many,
        open_judge,
    )

    out: dict[str, list[dict | None]] = {name: [None] * len(res) for name, res in results.items()}
    with open_judge(provider_id or default_judge_provider_id()) as judge:
        items = [
            ((name, i), JudgeCase(r.question, r.correct_answer, r.model_answer, list(r.keywords), r.expect_abstain))
            for name, res in results.items() for i, r in enumerate(res) if not (r.error and not r.model_answer)
        ]

        def record(key: tuple[str, int], res) -> None:
            out[key[0]][key[1]] = {"verdict": res.verdict, "reasoning": res.reasoning, "error": res.error,
                                   "model": judge.model}

        judge_many(judge, items, record)  # type: ignore[arg-type]
        return out, judge.label


def cmd_compare(args) -> None:
    from finetune_studio.cli.commands.suite import load_cases_or_exit
    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.suite import run_suite

    if not os.path.exists(args.suite):
        print(f"Error: Suite not found: {args.suite}")
        sys.exit(1)
    cases = load_cases_or_exit(args.suite)  # fail before loading any model
    models = _parse_models(args.models)

    log = sys.stderr if args.json else sys.stdout
    results: dict[str, list] = {}
    for name, path in models:  # one model resident at a time
        engine = InferenceEngine()
        print(f"Loading {name} from {path}...", file=log)
        engine.load(path)
        try:
            print(f"Running {len(cases)} questions on {name}...", file=log)
            results[name] = run_suite(engine, cases, max_tokens=args.max_tokens, temperature=args.temperature)
        finally:
            engine.unload()

    verdicts: dict[str, list[dict | None]] = {name: [None] * len(res) for name, res in results.items()}
    judge_label = ""
    if args.judge is not None:
        try:
            verdicts, judge_label = _judge_models(results, args.judge)
        except Exception as exc:  # noqa: BLE001 - JudgeUnavailable and provider errors alike
            print(f"Error: judging failed: {exc}", file=sys.stderr)
            sys.exit(1)

    report = {
        "judge": judge_label or None,
        "models": [name for name, _ in models],
        "cases": [
            {
                "name": case.name, "category": case.category, "question": case.question,
                "correct_answer": case.correct_answer,
                "answers": {
                    name: {"response": results[name][i].model_answer, "time_ms": results[name][i].time_ms,
                           "error": results[name][i].error, "judge": verdicts[name][i]}
                    for name, _ in models if i < len(results[name])
                },
            }
            for i, case in enumerate(cases)
        ],
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json_mod.dump(report, f, indent=2, ensure_ascii=False)

    if args.json:
        print(json_mod.dumps(report, indent=2, ensure_ascii=False))
    else:
        for c in report["cases"]:
            print(f"Q: {c['question'][:160]}")
            print(f"   key: {c['correct_answer'][:160]}")
            for name, a in c["answers"].items():
                tag = (a["judge"] or {}).get("verdict") or "unjudged"
                print(f"   [{tag}] {name} ({a['time_ms']}ms): {a['response'][:160]}"
                      f"{'...' if len(a['response']) > 160 else ''}")
                if a["error"]:
                    print(f"      Error: {a['error']}")
            print()
        print("=" * 50)
        if args.judge is None:
            print(f"{len(cases)} questions x {len(models)} models recorded, none judged "
                  "(use --judge to judge them, or compare in the Compare tab).")
        else:
            for name, _ in models:
                judged = [v for v in verdicts[name] if v and v["verdict"]]
                counts = {k: sum(1 for v in judged if v["verdict"] == k) for k in ("pass", "partial", "fail")}
                print(f"  {name}: {counts['pass']} pass, {counts['partial']} partial, {counts['fail']} fail, "
                      f"{len(cases) - len(judged)} not judged (of {len(cases)})")
            print(f"Judge: {judge_label}")
        if args.report:
            print(f"Report saved to: {args.report}")

    errors = sum(1 for res in results.values() for r in res if r.error)
    unjudged = args.judge is not None and any(
        not (v and v["verdict"]) and not (r.error and not r.model_answer)
        for name in results for r, v in zip(results[name], verdicts[name], strict=True)
    )
    if errors or unjudged:
        sys.exit(1)
