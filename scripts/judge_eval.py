#!/usr/bin/env python3
"""Measure a judge model against hand-labelled cases (is it a good judge?).

The gold file lists cases (question, answer key, model answer) with the verdict a careful reviewer gave. The chosen
provider row judges every case exactly as the Testing page would (``finetune_studio.testing.judge``), and the script
prints accuracy, a confusion matrix and every disagreement with the judge's reasoning, so a judge model (or a prompt
change) can be compared before it is trusted with a whole run.

    .venv/bin/python scripts/judge_eval.py --provider local-default
    .venv/bin/python scripts/judge_eval.py --provider local-qwen30b-a3b --out .tmp/judge-eval-qwen30b.json

A local GGUF judge takes the GPU while it runs (check headroom first) and is unloaded afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_GOLD = ROOT / "tests" / "corpus" / "korvane" / "eval" / "judge_gold.json"
LABELS = ("pass", "partial", "fail")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="", help="provider row id (default: the configured judge / helper seat)")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out", type=Path, help="write per-case results as JSON")
    ap.add_argument("--quiet", action="store_true", help="summary only, no per-case disagreements")
    args = ap.parse_args(argv)

    from finetune_studio.testing import judge as j

    gold = json.loads(args.gold.read_text(encoding="utf-8"))["cases"]
    provider = args.provider or j.default_judge_provider_id()
    results: list[dict] = []
    try:
        with j.open_judge(provider) as judge:
            print(f"Judge: {judge.label} ({judge.model}), prompt v{j.JUDGE_PROMPT_VERSION}, {len(gold)} gold cases", flush=True)
            items = [(g["id"], j.JudgeCase(g["question"], g["correct_answer"], g["model_answer"], list(g.get("keywords") or []),
                                           bool(g.get("expect_abstain")))) for g in gold]
            fresh: dict[str, j.JudgeResult] = {}
            j.judge_many(judge, items, lambda k, r: fresh.__setitem__(k, r))
    except j.JudgeUnavailable as exc:
        print(f"Error: judge unavailable: {exc}", file=sys.stderr)
        return 1

    confusion: Counter = Counter()
    for g in gold:
        res = fresh.get(g["id"]) or j.JudgeResult(error="not judged")
        results.append({"id": g["id"], "label": g["label"], "verdict": res.verdict, "reasoning": res.reasoning, "error": res.error})
        confusion[(g["label"], res.verdict or "none")] += 1
    right = sum(n for (lab, got), n in confusion.items() if lab == got)
    print(f"\naccuracy {right}/{len(gold)} = {right / len(gold):.1%}")
    print(f"{'gold \\ judge':<14}" + "".join(f"{c:>9}" for c in (*LABELS, "none")))
    for lab in LABELS:
        print(f"{lab:<14}" + "".join(f"{confusion[(lab, c)]:>9}" for c in (*LABELS, "none")))
    lenient = sum(confusion[(a, b)] for a, b in (("fail", "partial"), ("fail", "pass"), ("partial", "pass")))
    strict = sum(confusion[(a, b)] for a, b in (("pass", "partial"), ("pass", "fail"), ("partial", "fail")))
    print(f"too lenient: {lenient}   too strict: {strict}   no verdict: {sum(n for (_, got), n in confusion.items() if got == 'none')}")
    if not args.quiet:
        for r, g in zip(results, gold, strict=True):
            if r["verdict"] != r["label"]:
                print(f"\n[{r['id']}] gold={r['label']} judge={r['verdict'] or 'none'}\n  key:    {g['correct_answer'][:100]!r}\n"
                      f"  answer: {g['model_answer'][:180]!r}\n  judge:  {(r['reasoning'] or r['error'])[:300]!r}")
    if args.out:
        args.out.write_text(json.dumps({"provider": provider, "model": judge.model, "prompt_version": j.JUDGE_PROMPT_VERSION,
                                        "accuracy": right / len(gold), "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
