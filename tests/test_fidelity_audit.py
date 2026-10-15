"""Deterministic evidence checks for parser, dataset, suite, and scoring audits."""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio import db
from finetune_studio.data.audit import audit_qa_pairs, audit_source, audit_suite_cases
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.knowledge import sources
from finetune_studio.testing.audit import audit_cases
from finetune_studio.testing.generate_suite import generate_suite_from_training_data
from tests.knowledge_helpers import add_source


def test_source_audit_catches_raw_and_parse_drift() -> None:
    pid = db.create_project(name="audit", description="")["id"]
    source = add_source(pid, "policy.txt", "Alpha policy owner is Mira.\n\nThe review occurs every Friday.")
    assert audit_source(pid, source)["ok"] is True
    (sources.source_dir(pid, source) / "parsed.txt").write_text("corrupted", encoding="utf-8")
    report = audit_source(pid, source)
    assert "persisted_parse_differs_from_deterministic_reparse" in report["errors"]


def test_source_audit_accepts_a_hand_edit() -> None:
    pid = db.create_project(name="audit-edit", description="")["id"]
    source = add_source(pid, "policy.txt", "Alpha policy owner is Mira.\n\nThe review occurs every Friday.")
    edited = sources.set_text(pid, source.id, "Alpha policy owner is Mira Kowal. The review occurs every Friday.")
    assert "persisted_parse_differs_from_deterministic_reparse" not in audit_source(pid, edited)["errors"]


def test_empty_audit_is_not_reported_as_pass() -> None:
    from finetune_studio.data.audit import audit_project_sources

    report = audit_project_sources(db.create_project(name="empty", description="")["id"])
    assert report["status"] == "not_applicable"


def test_dataset_audit_flags_unknown_source_and_export_loss(tmp_path) -> None:
    pid = db.create_project(name="dataset", description="")["id"]
    sid = add_source(pid, "one.txt", "Mira owns the policy. It is reviewed every Friday.").id
    qa_fs.write_qa_pair(pid, {"id": "q1", "source_id": sid, "chunk_idx": 1,
                            "question": "Who owns the policy?", "answer": "Mira owns the policy.",
                            "chunk_text": "Mira owns the policy.", "status": "approved"})
    qa_fs.write_qa_pair(pid, {"id": "q2", "source_id": "missing", "chunk_idx": 1,
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
    doc = json.loads(suite.read_text(encoding="utf-8"))
    assert doc["meta"]["coverage"] == "full"
    assert doc["cases"][0]["source_id"] == "s1"
    assert doc["cases"][0]["chunk_idx"] == 2
    audit = audit_suite_cases(str(suite), str(dataset))
    assert audit["ok"] is True
    doc["cases"].pop()
    suite.write_text(json.dumps(doc), encoding="utf-8")
    assert "suite_dataset_count_mismatch" in audit_suite_cases(str(suite), str(dataset))["errors"]


def _saved_run(rows: list[dict]) -> tuple[str, list[dict]]:
    from finetune_studio import db

    pid = db.create_project(name="audit")["id"]
    rid = db.create_run(pid, "run")["id"]
    bid = db.create_benchmark(rid, "quiz", {})["id"]
    for i, r in enumerate(rows):
        db.create_case(bid, rid, f"c{i}", "g", r["q"], r["key"], r["answer"], [])
    return bid, db.list_cases(bid)


def test_audit_cases_reports_awaiting_and_where_verdicts_came_from(temp_db) -> None:
    from finetune_studio.db import judgements as jdb

    bid, cases = _saved_run([{"q": "Q1", "key": "8", "answer": "8"}, {"q": "Q2", "key": "9", "answer": "x"},
                             {"q": "Q3", "key": "1", "answer": "y"}])
    jdb.add_judgement(cases[0]["id"], kind="ai", verdict="pass", judge_model="m")
    jdb.add_judgement(cases[1]["id"], kind="human", verdict="fail")
    from finetune_studio import db

    report = audit_cases(bid, db.list_cases(bid))
    assert report["case_count"] == 3 and report["awaiting"] == 1
    assert report["verdict_sources"] == {"ai": 1, "human": 1}
    assert report["verdicts_without_judgement"] == [] and report["passed"] is True


def test_audit_cases_flags_a_verdict_nobody_gave(temp_db) -> None:
    from finetune_studio import db

    bid, cases = _saved_run([{"q": "Q1", "key": "8", "answer": "8"}])
    db.update_case(cases[0]["id"], verdict="pass", judge="ai")  # written behind the judgement table's back
    report = audit_cases(bid, db.list_cases(bid))
    assert report["verdicts_without_judgement"] == [cases[0]["id"]] and report["passed"] is False
