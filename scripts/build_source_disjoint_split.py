"""Build a source-disjoint training JSONL and held-out benchmark suite."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def build_split(project_dir: Path, train_path: Path, suite_path: Path, *, seed: int = 42) -> dict:
    pairs = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted((project_dir / "qa" / "pairs").glob("*.json"))]
    approved = [p for p in pairs if p.get("status", "approved") == "approved"
                and p.get("question") and p.get("answer") and p.get("source_id")]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for pair in approved:
        by_source[str(pair["source_id"])].append(pair)
    source_ids = sorted(by_source)
    random.Random(seed).shuffle(source_ids)
    target = max(1, round(len(approved) * 0.2))
    held_sources: list[str] = []
    held_count = 0
    for source_id in source_ids:
        held_sources.append(source_id)
        held_count += len(by_source[source_id])
        if held_count >= target:
            break
    held_set = set(held_sources)
    held = [p for p in approved if p["source_id"] in held_set]
    train = [p for p in approved if p["source_id"] not in held_set]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(json.dumps({
        "conversations": [
            {"from": "human", "value": p["question"]},
            {"from": "gpt", "value": p["answer"]},
        ],
        "source_id": p["source_id"],
        "chunk_idx": p.get("chunk_idx", 0),
    }, ensure_ascii=False) for p in train) + "\n", encoding="utf-8")
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(json.dumps({
        "name": "source-disjoint-held-out",
        "version": "1.0",
        "seed": seed,
        "held_out_source_ids": sorted(held_set),
        "cases": [{
            "name": f"source-held-{i:03d}",
            "category": "source-disjoint",
            "question": p["question"],
            "correct_answer": p["answer"],
            "source_id": p["source_id"],
            "chunk_idx": p.get("chunk_idx", 0),
            "keywords": p.get("keywords", []),
        } for i, p in enumerate(held)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "total": len(approved), "train": len(train), "held_out": len(held),
        "train_sources": len(source_ids) - len(held_set),
        "held_out_sources": sorted(held_set),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("train_path", type=Path)
    parser.add_argument("suite_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_split(args.project_dir, args.train_path, args.suite_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
