"""Training rows with three phrasings per fact while keeping the validation split fact-disjoint.

    .venv/bin/python tests/corpus/quillmoor/bench/train_variants.py --corpus .tmp/quillmoor/s

train/train.jsonl has one row per fact (the direct question). This writes train/train_x3_grouped.jsonl: that row plus
two generic re-phrasings built from the attribute's label ("Tell me the <label> of <entity>.", "<entity>: <label>?").
All three rows share `pair_id=fact` so the trainer's validation split keeps every fact entirely on one side.
None of the re-phrasings is an evaluation prompt (the paraphrase set uses hand-written ``para`` templates).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmgen.specs import spec_for


def build_rows(train_rows: list[dict[str, Any]], facts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Add two training phrasings per fact and group all its rows for validation splitting."""
    out = []
    for row in train_rows:
        fact_id = str(row["fact"])
        out.append({**row, "pair_id": fact_id})
        fact = facts[fact_id]
        label = fact["attr"].split(":", 1)[1].replace("_", " ") if ":" in fact["attr"] else fact["attr"]
        etype, attr = fact["etype"], fact["attr"]
        try:
            col = spec_for(etype, attr, label=label, code=label.upper()).col
        except KeyError:
            col = label
        if attr.startswith("rule:"):
            col = label
        entity = fact["entity"]
        answer = row["messages"][1]["content"]
        for question in (f"Tell me the {col.lower()} of {entity}.", f"{entity}: {col.lower()}?"):
            out.append({
                "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                "fact": fact_id,
                "pair_id": fact_id,
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    corpus = Path(a.corpus)
    facts = {f["id"]: f for f in (json.loads(x) for x in (corpus / "manifest" / "facts.jsonl").read_text().splitlines())}
    train_rows = [json.loads(x) for x in (corpus / "train" / "train.jsonl").read_text().splitlines() if x.strip()]
    out = build_rows(train_rows, facts)
    dest = corpus / "train" / "train_x3_grouped.jsonl"
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"{len(out)} rows -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
