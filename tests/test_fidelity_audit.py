"""Deterministic evidence checks for parser, dataset, suite, and scoring audits."""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.audit import audit_qa_pairs, audit_source, audit_suite_cases
from finetune_studio.data.fs import paths
from finetune_studio.data.fs.metadata import hash_bytes
from finetune_studio.testing.audit import recompute_cases
from finetune_studio.testing.generate_suite import generate_suite_from_training_data


def test_source_audit_catches_raw_and_parse_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "_PROJECTS", tmp_path / "projects")
    pid = "audit"
    raw = b"Alpha policy owner is Mira.\n\nThe review occurs every Friday."
    sha = hash_bytes(raw)
    fd, _ = pfs.store_file(pid, raw, original_filename="policy.txt")
    from finetune_studio.data.prep.ingest import parse_and_chunk

    ingested = parse_and_chunk(pid, raw, "policy.txt", sha256=sha, mime_type="text/plain")
    source = {"id": sha[:12], "sha256": sha, "filename": "policy.txt", "chunk_count": ingested.chunk_count}
    assert audit_source(pid, source)["ok"] is True
    (fd / "parsed.txt").write_text("corrupted", encoding="utf-8")
    report = audit_source(pid, source)
    assert "persisted_parse_differs_from_deterministic_reparse" in report["errors"]


def test_empty_audit_is_not_reported_as_pass(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "_PROJECTS", tmp_path / "projects")
    from finetune_studio.data.audit import audit_project_sources

    report = audit_project_sources("empty")
    assert report["status"] == "not_applicable"


def test_dataset_audit_flags_unknown_source_and_export_loss(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "_PROJECTS", tmp_path / "projects")
    pid = "dataset"
    source = {"id": "s1", "sha256": "a" * 64, "filename": "one.txt", "chunk_count": 1}
    pfs.write_qa_source(pid, source)
    pfs.write_qa_pair(pid, {"id": "q1", "source_id": "s1", "chunk_idx": 1,
                            "question": "Who owns the policy?", "answer": "Mira owns the policy.",
                            "chunk_text": "Mira owns the policy.", "status": "approved"})
    pfs.write_qa_pair(pid, {"id": "q2", "source_id": "missing", "chunk_idx": 1,
                            "question": "What is missing?", "answer": "It is missing.",
                            "chunk_text": "It is missing.", "status": "approved"})
    export = tmp_path / "train.jsonl"
    export.write_text("{}\n", encoding="utf-8")
    report = audit_qa_pairs(pid, exported_path=str(export))
    assert report["passed"] is False
    assert any(e.get("error") == "unknown_source_id" for e in report["errors"])
    assert "approved_pair_count_differs_from_export_count" in [e.get("error") for e in report["errors"]]


def test_suite_audit_detects_truncation_and_suite_preserves_provenance(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    rows = [{"conversations": [{"from": "human", "value": "Who owns the policy?"},
                                {"from": "gpt", "value": "Mira owns it."}],
             "source_id": "s1", "chunk_idx": 2}]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = generate_suite_from_training_data(str(dataset), str(tmp_path / "suite"))
    suite = Path(result["suite_path"])
    data = json.loads(suite.read_text(encoding="utf-8"))
    assert data[0]["source_id"] == "s1"
    assert data[0]["chunk_idx"] == 2
    audit = audit_suite_cases(str(suite), str(dataset))
    assert audit["ok"] is True
    data.pop()
    suite.write_text(json.dumps(data), encoding="utf-8")
    assert "suite_dataset_count_mismatch" in audit_suite_cases(str(suite), str(dataset))["errors"]


def test_recompute_cases_does_not_trust_stored_verdict() -> None:
    cases = [{"case_name": "n", "category": "numeric", "question": "How many units were shipped?",
              "correct_answer": "8 (Source: budget.csv)", "model_answer": "8",
              "verdict": "fail", "transcript": [{"role": "user", "content": "How many units were shipped?"},
                                                     {"role": "assistant", "content": "8"}]}]
    report = recompute_cases(cases)
    assert report["recomputed_scores"]["passed"] == 1
    assert report["disagreements"]
    assert report["passed"] is False
