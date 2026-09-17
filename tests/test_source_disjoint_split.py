from __future__ import annotations

import json
from pathlib import Path

from scripts.build_source_disjoint_split import build_split


def test_source_disjoint_split_has_no_source_overlap(tmp_path: Path) -> None:
    project = tmp_path / "project" / "qa" / "pairs"
    project.mkdir(parents=True)
    for i in range(10):
        source = f"source-{i % 4}"
        (project / f"q{i}.json").write_text(json.dumps({
            "id": f"q{i}", "status": "approved", "source_id": source,
            "question": f"Question {i}?", "answer": f"Answer {i}.",
        }), encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    suite_path = tmp_path / "suite.json"
    result = build_split(tmp_path / "project", train_path, suite_path)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    train_sources = {
        json.loads(line)["source_id"] for line in train_path.read_text().splitlines()
    }
    held_sources = set(suite["held_out_source_ids"])
    assert result["held_out"] > 0
    assert train_sources.isdisjoint(held_sources)
    assert {c["source_id"] for c in suite["cases"]} == held_sources
