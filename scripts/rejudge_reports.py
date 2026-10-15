#!/usr/bin/env python3
"""Re-judge saved test-run reports with an AI judge, offline (no running server needed).

Input: report JSON files shaped ``{"results": [{name|case_name, category, question, correct_answer, model_answer,
keywords, verdict, ...}]}`` — the old Testing-page reports (``.tmp/ragtrace/*.json``) and the files
``scripts/rag_reader_compare.py`` writes. Every answered case is read by the judge (``finetune_studio.testing.judge``:
any provider row, default = the configured judge, else the helper seat); a case whose category is ``unanswerable``
(or that has ``expect_abstain``) is judged as "the model should decline". Nothing here compares strings.

Writes ``<input>.judged.json`` next to each input with the new ``verdict`` / ``judge_reasoning`` / ``judge_model``
per case (the earlier verdict is kept as ``old_verdict``), and prints ``old verdict -> new verdict`` counts per
file plus every case whose verdict changed.

    .venv/bin/python scripts/rejudge_reports.py .tmp/ragtrace/gemma12-capauto-k20.json [...] [--provider ID]

A local GGUF judge replaces whatever model is resident on the GPU while it runs and is unloaded afterwards.
A case the judge cannot read stays without a verdict (``judge_error`` says why); it is never guessed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

NONE = "(none)"
UNJUDGED = "(unjudged)"


def case_name(row: dict[str, Any], index: int) -> str:
    return str(row.get("name") or row.get("case_name") or f"case-{index}")


def load_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The report object and its list of case rows; a clear error for any other shape."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: expected an object with a 'results' list of case objects")
    return data, rows


def is_answered(row: dict[str, Any]) -> bool:
    """A case the model actually answered (a run error or an empty reply leaves nothing to judge)."""
    return bool(str(row.get("model_answer") or "").strip()) and not row.get("error")


def to_judge_case(row: dict[str, Any]) -> Any:
    from finetune_studio.testing.judge import JudgeCase

    abstain = row.get("category") == "unanswerable" or bool(row.get("expect_abstain"))
    return JudgeCase(
        question=str(row.get("question") or ""),
        correct_answer=str(row.get("correct_answer") or ""),
        model_answer=str(row.get("model_answer") or ""),
        keywords=[str(k) for k in (row.get("keywords") or [])],
        expect_abstain=abstain,
    )


def rejudge_file(path: Path, judge: Any) -> tuple[Path, Counter, list[tuple[str, str, str, str]], int]:
    """Judge one report with the open ``judge``. Returns (output path, transition counts, flipped cases, skipped)."""
    from finetune_studio.testing import judge as judge_mod

    data, rows = load_results(path)
    items = [(str(i), to_judge_case(r)) for i, r in enumerate(rows) if is_answered(r)]
    fresh: dict[int, Any] = {}

    def record(key: str, res: Any) -> None:
        fresh[int(key)] = res

    judge_mod.judge_many(judge, items, record)

    transitions: Counter = Counter()
    flipped: list[tuple[str, str, str, str]] = []
    for i, row in enumerate(rows):
        old = str(row.get("verdict") or "") or NONE
        row["old_verdict"] = row.get("verdict", "")
        res = fresh.get(i)
        if res is None:                       # not answered: no judge call, no verdict
            new, row["verdict"], row["judge_error"] = UNJUDGED, "", "no model answer to judge"
        else:
            new = res.verdict or UNJUDGED
            row.update(verdict=res.verdict, judge="ai", judge_model=judge.model, judge_reasoning=res.reasoning,
                       judge_error=res.error, judge_confidence=res.confidence)
            row["passed"] = res.verdict == "pass"
        transitions[(old, new)] += 1
        if new != UNJUDGED and old != new:
            flipped.append((case_name(row, i), old, new, str(row.get("judge_reasoning") or "")))

    data["rejudge"] = {"provider_id": judge.provider_id, "judge_model": judge.model, "source": path.name,
                       "judged": len(fresh), "cases": len(rows)}
    out = path.with_name(f"{path.stem}.judged.json")
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return out, transitions, flipped, len(rows) - len(items)


def print_report(path: Path, out: Path, transitions: Counter, flipped: list[tuple[str, str, str, str]], skipped: int) -> None:
    total = sum(transitions.values())
    print(f"\n{path.name}  ->  {out.name}   ({total} cases{f', {skipped} without an answer' if skipped else ''})")
    print(f"  {'old verdict':<12} -> {'new verdict':<12} {'cases':>5}")
    for (old, new), n in sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {old:<12} -> {new:<12} {n:>5}{'' if old == new else '   *'}")
    new_counts = Counter()
    for (_, new), n in transitions.items():
        new_counts[new] += n
    print("  new totals: " + ", ".join(f"{k}={new_counts[k]}" for k in ("pass", "partial", "fail", UNJUDGED) if new_counts[k]))
    if flipped:
        print(f"  flipped ({len(flipped)}):")
        for name, old, new, why in flipped:
            print(f"    {name}: {old} -> {new}  {why[:140]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="saved report JSON file(s)")
    ap.add_argument("--provider", default="", metavar="PROVIDER_ID",
                    help="judge provider row id (default: the configured judge, else the helper seat)")
    args = ap.parse_args(argv)

    from finetune_studio.testing import judge as judge_mod

    missing = [f for f in args.files if not f.is_file()]
    if missing:
        print(f"Error: not a file: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    provider = args.provider or judge_mod.default_judge_provider_id()
    try:
        with judge_mod.open_judge(provider) as judge:
            print(f"Judge: {judge.label} ({judge.model}), {'parallel' if judge.concurrent else 'sequential'}", flush=True)
            for path in args.files:
                try:
                    print_report(path, *rejudge_file(path, judge))
                except (ValueError, json.JSONDecodeError) as exc:
                    print(f"Error: {path}: {exc}", file=sys.stderr)
                    return 2
    except judge_mod.JudgeUnavailable as exc:
        print(f"Error: judge unavailable: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
