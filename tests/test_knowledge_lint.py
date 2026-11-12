"""Knowledge check: each rule fires on what it is for and stays quiet otherwise; ignores persist."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finetune_studio.knowledge import lint as kl
from finetune_studio.knowledge.sources import Source

BODY = " ".join(f"Paragraph {i} explains how the warehouse handles pallets during the morning shift." for i in range(30))


def src(sid: str, name: str, *, status: str = "ready", chunks: int = 3, sha: str = "", error: str = "") -> Source:
    return Source(id=sid, file_id=f"f{sid}", version=1, sha256=sid * 4, filename=name, mime_type="text/plain", raw_path="",
                  size_bytes=1, uploaded_at=0.0, status=status, text_sha=sha or f"t{sid}", parser="x", char_count=9,
                  chunk_count=chunks, error=error, edited_at=None, deleted=False)


def rules(findings: list[kl.Finding]) -> list[str]:
    return sorted(f.rule for f in findings)


def test_clean_documents_have_no_findings() -> None:
    assert kl._check_files([src("a", "guide.txt"), src("b", "notes.txt")],
                           {"a": BODY, "b": "Entirely different words about invoices and shipping terms. " * 20}) == []


def test_unreadable_and_empty_files_are_errors() -> None:
    found = kl._check_files([src("a", "bad.pdf", status="error", chunks=0, error="encrypted PDF"),
                             src("b", "blank.pdf", chunks=0)], {})
    assert rules(found) == ["unreadable", "unreadable"] and {f.severity for f in found} == {"error"}
    assert found[0].detail == ["encrypted PDF"]


def test_thin_and_garbled_text_but_not_tables() -> None:
    assert rules(kl._check_files([src("a", "scan.pdf")], {"a": "Page 1"})) == ["thin"]
    numbers = "12 34 56 78 90 " * 100
    assert rules(kl._check_files([src("a", "dump.txt")], {"a": numbers})) == ["garbled"]
    assert kl._check_files([src("a", "rates.csv")], {"a": numbers}) == []


@pytest.mark.parametrize(("text", "severity"), [
    ("-----BEGIN RSA PRIVATE " + "KEY-----\nabc " + BODY, "error"),
    ("key AKIA" + "IOSFODNN7EXAMPLE here " + BODY, "error"),
    ("token sk" + "-abcdefghijklmnopqrstuvwxyz123456 here " + BODY, "error"),
    ("password: hunter2secret " + BODY, "warning"),
])
def test_secrets_are_flagged_and_the_value_is_not_repeated(text: str, severity: str) -> None:
    found = [f for f in kl._check_files([src("a", "notes.txt")], {"a": text}) if f.rule in ("secret", "credential")]
    assert len(found) == 1 and found[0].severity == severity
    shown = json.dumps(found[0].__dict__)
    assert "hunter2secret" not in shown and "IOSFODNN7EXAMPLE" not in shown


def test_exact_and_near_duplicates() -> None:
    assert rules(kl._check_files([src("a", "one.txt", sha="same"), src("b", "two.txt", sha="same")],
                                 {"a": BODY, "b": BODY})) == ["duplicate"]
    rng = random.Random(1)
    long_text = " ".join("".join(rng.choices("abcdefghij", k=6)) for _ in range(400))
    near = long_text + " One closing sentence was added to this copy."
    assert rules(kl._check_files([src("a", "one.txt"), src("b", "two.txt")], {"a": long_text, "b": near})) == ["near_duplicate"]


@pytest.mark.parametrize("name", ["handbook_superseded.doc", "policy-OLD.pdf", "contract draft.docx"])
def test_old_names(name: str) -> None:
    assert rules(kl._check_files([src("a", name)], {"a": BODY})) == ["old_version"]


def test_a_word_inside_a_name_is_not_an_old_version() -> None:
    assert kl._check_files([src("a", "goldman_report.pdf"), src("b", "scaffold.txt")],
                           {"a": BODY, "b": "Other words entirely, about invoices and terms of shipping. " * 20}) == []


def test_old_version_from_the_text_and_from_a_newer_sibling() -> None:
    other = "Different words about invoices and shipping terms for the customer. " * 20
    assert rules(kl._check_files([src("a", "guide.txt")], {"a": "This document is superseded by the 2024 edition. " + BODY})) == ["old_version"]
    found = kl._check_files([src("a", "manual_v2.pdf"), src("b", "manual_v3.pdf")], {"a": BODY, "b": other})
    assert rules(found) == ["old_version"] and found[0].files[0]["name"] == "manual_v2.pdf"


def test_a_newer_version_in_another_file_type_counts() -> None:
    other = "Different words about invoices and shipping terms for the customer. " * 20
    found = kl._check_files([src("a", "policy-443-v1.epub"), src("b", "policy-443-v2.odt")], {"a": BODY, "b": other})
    assert rules(found) == ["old_version"] and found[0].files[0]["name"] == "policy-443-v1.epub"


def test_a_newer_version_with_its_own_year_in_the_name_counts() -> None:
    other = "Different words about invoices and shipping terms for the customer. " * 20
    found = kl._check_files([src("a", "Employee_Handbook_v1_2023.pdf"), src("b", "Employee_Handbook_v2_2025.docx"),
                             src("c", "terms_of_service_v3_2023-08-01.pdf"), src("d", "terms_of_service_v4_2025.docx")],
                            {"a": BODY, "b": other, "c": "Leases end with ninety days of notice to the customer. " * 20,
                             "d": "Holiday entitlement rises for every staff member this year. " * 20})
    assert sorted(f.files[0]["name"] for f in found) == ["Employee_Handbook_v1_2023.pdf", "terms_of_service_v3_2023-08-01.pdf"]


def test_a_lower_version_of_a_later_year_is_not_older() -> None:
    other = "Different words about invoices and shipping terms for the customer. " * 20
    found = kl._check_files([src("a", "budget_2024_v3.xlsx"), src("b", "budget_2025_v1.xlsx")], {"a": BODY, "b": other})
    assert found == []


def _index(tmp: Path, rows: list[tuple[str, str]]) -> Path:
    corpus = tmp / "corpus"
    corpus.mkdir()
    pd.DataFrame([{"id": f"c{i}", "document_id": d, "text": t} for i, (d, t) in enumerate(rows)]).to_parquet(corpus / "chunks.parquet")
    (corpus / "vectors.idx.json").write_text(json.dumps({f"c{i}": i for i in range(len(rows))}))
    np.save(corpus / "vectors.npy", np.ones((len(rows), 4), dtype=np.float32) / 2)   # every passage "means the same"
    return corpus


def conflicts(monkeypatch, tmp_path: Path, rows: list[tuple[str, str]], ids: str = "ab") -> list[kl.Finding]:
    corpus = _index(tmp_path, rows)
    monkeypatch.setattr(kl, "brain_index_dir", lambda pid: corpus)
    return kl._check_conflicts("p", [src(i, f"{i}.txt") for i in sorted({d for d, _ in rows} | set(ids))])


def test_conflict_same_label_different_value(monkeypatch, tmp_path: Path) -> None:
    found = conflicts(monkeypatch, tmp_path, [("a", "The probation period is 2 months."), ("b", "The probation period is 3 months.")])
    assert rules(found) == ["conflict"] and found[0].detail == ['"probation period": 2 vs 3']


@pytest.mark.parametrize(("left", "right"), [
    ("The probation period is 2 months.", "The probation period is 2 days."),
    ("The annual salary is EUR 2,000.", "The annual salary is $2,000."),
])
def test_different_units_are_not_reported_as_a_conflict(monkeypatch, tmp_path: Path,
                                                        left: str, right: str) -> None:
    found = conflicts(monkeypatch, tmp_path, [
        ("a", left),
        ("b", right),
    ])
    assert found == []


def test_no_conflict_for_equal_values_or_the_same_file(monkeypatch, tmp_path: Path) -> None:
    assert conflicts(monkeypatch, tmp_path, [("a", "The probation period is 3 months."), ("b", "The probation period is 3 months."),
                                             ("a", "The probation period is 9 months.")][:2]) == []


def test_no_conflict_for_noise_labels_and_dates(monkeypatch, tmp_path: Path) -> None:
    assert conflicts(monkeypatch, tmp_path, [("a", "Press the button 4 times. The contract ends 31 December 2024."),
                                             ("b", "Press the button 10 times. The contract ends 31.12.2025.")]) == []


def test_a_template_field_is_not_a_conflict(monkeypatch, tmp_path: Path) -> None:
    rows = [(d, f"The station raised {v} euros.") for d, v in zip("abcde", ("100", "200", "300", "400", "500"), strict=True)]
    assert conflicts(monkeypatch, tmp_path, rows, ids="abcde") == []


def test_the_key_is_stable_and_ignores_persist(client) -> None:
    pid = client.post("/api/projects", json={"name": "lint-ignore"}).json()["id"]
    a, b = kl._finding("conflict", "warning", "t", "f", [src("a", "1"), src("b", "2")]), kl._finding("conflict", "warning", "x", "y", [src("b", "2"), src("a", "1")])
    assert a.key == b.key
    kl.set_ignored(pid, a.key, True)
    assert a.key in kl.ignored_keys(pid)
    kl.set_ignored(pid, a.key, False)
    assert kl.ignored_keys(pid) == set()


def test_empty_project_lints_clean(client) -> None:
    pid = client.post("/api/projects", json={"name": "lint-empty"}).json()["id"]
    assert client.get(f"/api/projects/{pid}/brain/lint").json() == {
        "files": 0, "reading": 0, "errors": 0, "warnings": 0, "ignored": 0, "findings": []}


def test_the_api_ignores_a_finding_and_refuses_junk(client, monkeypatch) -> None:
    pid = client.post("/api/projects", json={"name": "lint-api"}).json()["id"]
    url = f"/api/projects/{pid}/brain/lint/ignore"
    assert client.post(url, json={"key": "abc123", "ignored": True}).status_code == 200
    assert "abc123" in kl.ignored_keys(pid)
    assert client.post(url, json={"key": ""}).status_code == 422
    assert client.get("/api/projects/nope/brain/lint").status_code == 404


def test_the_page_shows_the_check_and_can_ignore() -> None:
    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/brain.html").read_text()
    assert 'id="brain-lint"' in html and "/brain/lint" in html and "/brain/lint/ignore" in html


def test_a_pair_with_an_out_of_date_file_is_not_a_conflict(monkeypatch, tmp_path: Path) -> None:
    corpus = _index(tmp_path, [("a", "Annual leave is 26 days."), ("b", "Annual leave is 30 days.")])
    monkeypatch.setattr(kl, "brain_index_dir", lambda pid: corpus)
    assert kl._check_conflicts("p", [src("a", "a.txt"), src("b", "b.txt")], outdated={"a"}) == []


def test_a_form_field_is_found_over_the_whole_index() -> None:
    texts = [f"The station raised {v} euros." for v in (100, 200, 300, 400)] + ["The probation period is 3 months."]
    labels = kl._template_labels(texts, ["a", "b", "c", "d", "e"])
    assert any(label == "station raised" for label, _unit in labels)
    assert not any(label == "probation period" for label, _unit in labels)


def test_excluded_documents_skips_old_versions_and_later_duplicates(monkeypatch) -> None:
    def finding(rule: str, ids: str, ignored: bool = False) -> dict:
        return {"rule": rule, "ignored": ignored, "files": [{"id": i} for i in ids]}
    report = {"findings": [finding("old_version", "a"), finding("duplicate", "bc"),
                           finding("near_duplicate", "de", ignored=True), finding("conflict", "fg")]}
    monkeypatch.setattr(kl, "lint", lambda pid: report)
    assert kl.excluded_documents("p") == {"a": "out of date", "c": "duplicate"}


def test_the_quiz_leaves_out_superseded_documents(monkeypatch, tmp_path: Path) -> None:
    from finetune_studio.testing import brain_quiz

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pd.DataFrame([{"document_id": "old", "filename": "handbook-v1.md", "text": "26 days"},
                  {"document_id": "new", "filename": "handbook-v2.md", "text": "30 days"}]).to_parquet(corpus / "chunks.parquet")
    monkeypatch.setattr(brain_quiz, "brain_index_dir", lambda pid: corpus)
    monkeypatch.setattr(kl, "excluded_documents", lambda pid: {"old": "out of date"})
    rows = brain_quiz.index_rows("p")
    assert [r["document_id"] for r in rows] == ["new"]
    monkeypatch.setattr(kl, "excluded_documents", lambda pid: {"old": "x", "new": "y"})
    with pytest.raises(brain_quiz.NoIndex):
        brain_quiz.index_rows("p")
