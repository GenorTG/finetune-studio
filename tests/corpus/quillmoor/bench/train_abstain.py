"""Training rows that also teach "I don't know": the x3 phrasings plus abstention rows, fact-disjoint as before.

    .venv/bin/python tests/corpus/quillmoor/bench/train_abstain.py --corpus .tmp/quillmoor/s [--reserve 0.1 --fictional 600]

Every tuned model scored 0 % correct abstention on app-held-out facts: SFT rows only ever answer, so a model learns
to answer everything. This writes train/train_x3_abstain.jsonl from train/train_x3_grouped.jsonl:

- reserved facts: a seeded share of facts keeps its three phrasings but answers each with ABSTAIN (a known entity,
  an attribute the model was never taught: exactly the held-out situation). These rows carry ``abstain_fact``
  instead of ``fact``, so the seen/held-out evaluation never scores them as trained facts.
- fictional entities: the canonical question of a random fact with its entity swapped for a made-up one of the same
  kind (a name or code that exists nowhere in the corpus), answered ABSTAIN.

All rows of a fact share ``pair_id``, so the app's validation split still keeps each fact on one side.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

ABSTAIN = "I don't know: that is not in the material I was trained on."
SEED = 20261030


def fictional_entity(entity: str, names: set[str], rng: random.Random, pool: list[str]) -> str:
    """A same-shaped entity that is not in the corpus: codes get new digits, names a new word combination."""
    for _ in range(200):
        if re.search(r"\d", entity):   # codes (UVR-4655-R, QP-572087, INC-2021-4003): replace every digit
            cand = re.sub(r"\d", lambda _m: str(rng.randrange(10)), entity)
        else:                          # names (people, companies, sites, vessels): swap the last word with another's
            other = rng.choice(pool).split()
            words = entity.split()
            cand = " ".join([*words[:-1], other[-1]]) if len(words) > 1 else f"{words[0]} {other[-1]}"
        if cand != entity and cand not in names:
            return cand
    raise RuntimeError(f"no fictional variant found for {entity!r}")


def build_rows(rows: list[dict[str, Any]], canonical: list[dict[str, Any]], facts: dict[str, dict[str, Any]], *,
               reserve: float, fictional: int, seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    fact_ids = sorted({str(r["fact"]) for r in rows})
    reserved = set(rng.sample(fact_ids, round(len(fact_ids) * reserve)))
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["fact"] in reserved:
            out.append({"messages": [r["messages"][0], {"role": "assistant", "content": ABSTAIN}],
                        "abstain_fact": r["fact"], "pair_id": r["pair_id"]})
        else:
            out.append(r)
    names = {f["entity"] for f in facts.values()}
    by_type: dict[str, list[str]] = {}
    for f in facts.values():
        by_type.setdefault(f["etype"], []).append(f["entity"])
    picks = rng.sample(canonical, fictional)
    for i, row in enumerate(picks):
        fact = facts[row["fact"]]
        entity = fact["entity"]
        question = row["messages"][0]["content"]
        if entity not in question:
            continue
        fake = fictional_entity(entity, names, rng, sorted(set(by_type[fact["etype"]])))
        out.append({"messages": [{"role": "user", "content": question.replace(entity, fake)},
                                 {"role": "assistant", "content": ABSTAIN}],
                    "fictional": fake, "pair_id": f"fictional-{i}"})
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--reserve", type=float, default=0.1, help="share of facts answered ABSTAIN instead")
    ap.add_argument("--fictional", type=int, default=600, help="questions about made-up entities")
    a = ap.parse_args()
    corpus = Path(a.corpus)
    def load(p: Path) -> list[dict[str, Any]]:
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    facts = {f["id"]: f for f in load(corpus / "manifest" / "facts.jsonl")}
    rows = build_rows(load(corpus / "train" / "train_x3_grouped.jsonl"), load(corpus / "train" / "train.jsonl"),
                      facts, reserve=a.reserve, fictional=a.fictional)
    dest = corpus / "train" / "train_x3_abstain.jsonl"
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n_abstain = sum(1 for r in rows if r["messages"][1]["content"] == ABSTAIN)
    print(f"{len(rows)} rows ({n_abstain} abstain) -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
