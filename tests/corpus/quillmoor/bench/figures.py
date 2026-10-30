"""Figures for the Quillmoor write-up, drawn from the judged result files (plain SVG, no plotting library).

    .venv/bin/python tests/corpus/quillmoor/bench/figures.py --corpus .tmp/quillmoor/s --out docs/benchmarks/img

Writes quillmoor-answers.svg (Smart Brain before/after per answer model), quillmoor-question-types.svg (API model per
question type) and quillmoor-sft-passes.svg (trained-fact recall per fine-tuning recipe). Scores are the AI judge's
(pass 1, partial 0.5, fail 0) exactly as RESULTS.md reports them.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path

POINTS = {"pass": 1.0, "partial": 0.5, "fail": 0.0}
COLORS = ("#9aa5b1", "#4a90d9", "#1f6f3f", "#c77d2e")
FONT = "font-family='Helvetica, Arial, sans-serif'"


def judged(path: Path) -> list[dict]:
    return [r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip()) if r.get("verdict") in POINTS]


def score(rows: list[dict]) -> float:
    return 100 * sum(POINTS[r["verdict"]] for r in rows) / len(rows)


def pass_rate(rows: list[dict]) -> float:
    return 100 * sum(r["verdict"] == "pass" for r in rows) / len(rows)


def bars(title: str, groups: list[str], series: list[str], values: list[list[float | None]], note: str,
         colors: tuple[str, ...] = COLORS) -> str:
    """Horizontal grouped bars: one group per label, one bar per series, 0-100 %."""
    left, width, bar_h, gap = 210, 470, 16, 14
    group_h = bar_h * len(series) + gap
    legend_y = 46
    top = legend_y + 26
    height = top + group_h * len(groups) + 46
    out = [(f"<svg xmlns='http://www.w3.org/2000/svg' width='{left + width + 70}' height='{height}' "
            f"viewBox='0 0 {left + width + 70} {height}' {FONT}>"),
           "<rect width='100%' height='100%' fill='#ffffff'/>",
           f"<text x='16' y='26' font-size='16' font-weight='bold' fill='#1d2733'>{escape(title)}</text>"]
    x = left
    for i, name in enumerate(series):
        out.append(f"<rect x='{x}' y='{legend_y - 10}' width='12' height='12' fill='{colors[i % len(colors)]}'/>")
        out.append(f"<text x='{x + 16}' y='{legend_y}' font-size='12' fill='#1d2733'>{escape(name)}</text>")
        x += 22 + 7 * len(name)
    for tick in range(0, 101, 20):
        tx = left + width * tick / 100
        out.append(f"<line x1='{tx}' y1='{top - 6}' x2='{tx}' y2='{top + group_h * len(groups) - gap + 4}' "
                   "stroke='#e3e8ee' stroke-width='1'/>")
        out.append(f"<text x='{tx}' y='{top + group_h * len(groups) + 8}' font-size='10' fill='#6b7785' "
                   f"text-anchor='middle'>{tick}%</text>")
    for g, label in enumerate(groups):
        gy = top + g * group_h
        out.append(f"<text x='{left - 10}' y='{gy + bar_h * len(series) / 2 + 4}' font-size='12' fill='#1d2733' "
                   f"text-anchor='end'>{escape(label)}</text>")
        for s, value in enumerate(values[g]):
            y = gy + s * bar_h
            if value is None:
                continue
            w = max(1.0, width * value / 100)
            out.append(f"<rect x='{left}' y='{y}' width='{w:.1f}' height='{bar_h - 3}' fill='{colors[s % len(colors)]}'/>")
            out.append(f"<text x='{left + w + 4:.1f}' y='{y + bar_h - 6}' font-size='10' fill='#1d2733'>{value:.1f}%</text>")
    out.append(f"<text x='16' y='{height - 12}' font-size='10' fill='#6b7785'>{escape(note)}</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    answers = Path(a.corpus) / "results" / "answers"
    sft = Path(a.corpus) / "results" / "sft-facts"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    models = [("DeepSeek v4.1 Flash (API)", "deepseek-flash"), ("Qwen 3.5 4B", "qwen35-4b-base"),
              ("Llama 3.2 3B", "llama32-3b-base"), ("Gemma 3 4B", "gemma3-4b-base")]
    configs = [("MiniLM reranker (before)", "rag-default"), ("bge-reranker-v2-m3", "rag-e5-bge"),
               ("bge + follow-up search", "rag-e5-bge-fu1")]
    values = [[score(judged(answers / f"{m}__{c}.jsonl")) for _, c in configs] for _, m in models]
    (out / "quillmoor-answers.svg").write_text(bars(
        "Smart Brain answers, 376 Quillmoor questions", [n for n, _ in models], [n for n, _ in configs], values,
        "AI-judged score (pass 1, partial 0.5). Same questions, judge and E5-large hybrid index for every bar."))

    types = [("paraphrase", "reworded"), ("verbatim", "verbatim"), ("superseded", "changed value"),
             ("reverse", "reverse look-up"), ("multihop", "two-step"), ("unanswerable", "not in the documents")]
    by: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for _, c in configs:
        for r in judged(answers / f"deepseek-flash__{c}.jsonl"):
            by[r["type"]][c].append(r)
    (out / "quillmoor-question-types.svg").write_text(bars(
        "API answer model by question type", [n for _, n in types], [n for n, _ in configs],
        [[score(by[t][c]) for _, c in configs] for t, _ in types],
        "DeepSeek v4.1 Flash on the same exported packages; 26-140 questions per type."))

    recipes = [("4 passes (old recipe)", "sftfacts-llama32-3b-tuned"),
               ("Balanced preset, 2 epochs", "sftfacts-llama32-3b-tuned-tier-balanced"),
               ("Precision preset, 4 epochs", "sftfacts-llama32-3b-tuned-tier-precision"),
               ("Balanced, corrected (6)", "sftfacts-llama32-3b-tuned-tier-balanced-fixed"),
               ("Precision, corrected (12)", "sftfacts-llama32-3b-tuned-tier-precision-fixed"),
               ("12 passes, 1 phrasing", "sftfacts-llama32-3b-tuned-12ep-r32"),
               ("3 phrasings x 4 passes", "sftfacts-llama32-3b-tuned-x3-grouped-4ep-r32"),
               ("3 phrasings + I-don't-know rows", "sftfacts-llama32-3b-tuned-x3-abstain-4ep-r32")]
    rows = []
    for label, stem in recipes:
        f = sft / f"{stem}__closed.jsonl"
        if not f.exists():
            continue
        recs = judged(f)
        seen = [r for r in recs if r["type"] == "sft-trained"]
        held = [r for r in recs if r["type"] == "sft-heldout"]
        rows.append((label, [pass_rate(seen), pass_rate(held)]))
    (out / "quillmoor-sft-passes.svg").write_text(bars(
        "Fine-tuning Llama 3.2 3B on 4,306 facts", [r[0] for r in rows],
        ["recalls a trained fact", "says it does not know a held-out fact"], [r[1] for r in rows],
        "Exact-pass rate, AI-judged; one question per fact (392-431 trained, 392-431 held-out).",
        colors=("#1f6f3f", "#c77d2e")))
    print(f"wrote {len(list(out.glob('quillmoor-*.svg')))} figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
