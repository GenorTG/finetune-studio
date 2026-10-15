"""Regression tests for the A2 data-prep audit (2026-10-01).

1. knowledge.sources.passages indexes chunk files by their own NNNN filename instead of glob() order, so a missing chunk
   file leaves a gap instead of silently shifting every later chunk's index down by one (mining keys items and pairs off it).
2. audit.audit_qa_pairs compares the export line count against the same deduplicate_qa_pairs() the exporter actually runs,
   instead of raw len(pairs), so a legitimate (source, chunk, question) collapse is not misreported as export data loss.
"""
from __future__ import annotations

import json

import pytest

from finetune_studio import db
from finetune_studio.data.audit import audit_qa_pairs
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.data.fs.paths import file_dir, project_dir
from finetune_studio.knowledge.sources import passages
from tests.knowledge_helpers import add_source


@pytest.fixture()
def proj() -> str:
    return db.create_project(name="audit", description="")["id"]


def test_passages_preserve_index_across_gap(proj: str) -> None:
    sha = "gapsha1234567"
    fd = file_dir(proj, sha)
    chunks_dir = fd / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "0000.txt").write_text("first chunk", encoding="utf-8")
    # 0001.txt intentionally missing (simulates a partial/failed write)
    (chunks_dir / "0002.txt").write_text("third chunk", encoding="utf-8")

    chunks = passages(proj, sha)

    assert len(chunks) == 3
    assert chunks[0] == "first chunk"
    assert chunks[1] == ""  # the gap, not "third chunk" shifted down
    assert chunks[2] == "third chunk"


def test_audit_qa_pairs_does_not_flag_legitimate_dedup_as_loss(proj: str) -> None:
    sid = add_source(proj, "one.txt", "Mira owns the policy. The charter names her as its owner.").id
    # Two approved pairs for the SAME (source, chunk, question) — a
    # legitimate duplicate the exporter's deduplicate_qa_pairs() collapses
    # into one training row.
    qa_fs.write_qa_pair(proj, {
        "id": "q1", "source_id": sid, "chunk_idx": 1,
        "question": "Who owns the policy?", "answer": "Mira owns the policy.",
        "chunk_text": "Mira owns the policy.", "status": "approved",
    })
    qa_fs.write_qa_pair(proj, {
        "id": "q2", "source_id": sid, "chunk_idx": 1,
        "question": "who owns the policy?", "answer": "Mira owns the policy, per the charter.",
        "chunk_text": "Mira owns the policy.", "status": "approved",
    })
    export_path = project_dir(proj) / "train.jsonl"
    row = {"conversations": [
        {"from": "human", "value": "Who owns the policy?"},
        {"from": "gpt", "value": "Mira owns the policy, per the charter."},
    ], "source_id": sid, "chunk_idx": 1}
    export_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = audit_qa_pairs(proj, exported_path=str(export_path))

    assert report["approved_pair_count"] == 2
    assert report["expected_export_count"] == 1
    assert report["duplicate_pairs_collapsed"] == 1
    assert report["export_count"] == 1
    assert "approved_pair_count_differs_from_export_count" not in [
        e.get("error") for e in report["errors"]
    ]
    assert report["passed"] is True
