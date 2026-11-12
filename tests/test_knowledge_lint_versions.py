"""Knowledge check: amendments linked to what they amend, older workbook sheets, year-named versions, lone drafts, and
the CLI export's superseded-document gate (Tarrowfield 2026-11-12)."""
from __future__ import annotations

import argparse

import pytest

from finetune_studio.knowledge import lint as kl
from finetune_studio.knowledge import lint_versions as lv
from finetune_studio.knowledge.sources import Source

BODY = " ".join(f"Clause {i} sets out how the equipment is serviced and invoiced every quarter." for i in range(40))


def src(sid: str, name: str) -> Source:
    return Source(id=sid, file_id=f"f{sid}", version=1, sha256=sid * 4, filename=name, mime_type="x", raw_path="",
                  size_bytes=1, uploaded_at=0.0, status="ready", text_sha=f"t{sid}", parser="x", char_count=9,
                  chunk_count=3, error="", edited_at=None, deleted=False)


MSA = src("msa", "MSA_Harrowgate_Mutual_TWL-MSA-2025-014_signed_2025-03-18.pdf")
AM1 = src("am1", "Harrowgate_Amendment_No1_2025-09-09.docx")
SHEET = src("deal", "Harrowgate_deal_sheet_OPP-24-0917.xlsx")
MAIL = src("mail", "RE_Harrowgate_amendment_SLA_credits.eml")
LEASE = src("lease", "Brannock_Master_Lease_TWL-ML-2022-087.pdf")
AM2 = src("am2", "Brannock_Amendment_No2_effective_2025-04-01_signed_scan.pdf")
TEXTS = {
    "msa": "Master Agreement TWL-MSA-2025-014 between Tarrowfield and Harrowgate. " + BODY,
    "am1": "Amendment No. 1 to Master Agreement TWL-MSA-2025-014\nRecitals (A) ... With effect from the date of this "
           "Amendment clause 9.2 is deleted and replaced. " + BODY,
    "deal": "=== Sheet: Deal (4 rows) ===\nOPP-24-0917 | Harrowgate | 120 units",
    "mail": "Subject: RE: Harrowgate amendment SLA credits\nAmendment No. 1 changes the credits. " + BODY,
    "lease": "Master Lease Agreement — Brannock Engineering Group Ltd " + BODY,
    "am2": "[Figure 1 · page 1 · scanned page · read by gemma]\nAMENDMENT No. 2 to the Master Lease Agreement dated "
           "1 June 2022 between Tarrowfield and Brannock. Supersedes Schedule 2.\n[End of figure 1] " + BODY,
}


def test_amendments_link_to_the_agreement_by_reference_or_name() -> None:
    found = {a.amendment.id: a for a in lv.amendments([MSA, AM1, SHEET, MAIL, LEASE, AM2], TEXTS)}
    assert set(found) == {"am1", "am2"}                       # the e-mail about the amendment is not one
    assert found["am1"].base.id == "msa" and found["am1"].number == "1"
    assert "cites TWL-MSA-2025-014" in found["am1"].evidence[0]
    assert any("with effect from" in e for e in found["am1"].evidence)
    assert found["am2"].base.id == "lease" and "brannock" in found["am2"].evidence[0]


def test_the_pair_is_reported_as_amended_and_not_as_near_duplicate() -> None:
    sources = [MSA, AM1, LEASE, AM2]
    texts = {**TEXTS, "am1": TEXTS["msa"][:60] + BODY, "msa": TEXTS["msa"]}   # nearly the same wording
    findings, pairs = kl._check_versions(sources, {**texts, "am1": "Amendment No. 1 to TWL-MSA-2025-014. " + BODY})
    amended = [f for f in findings if f.rule == "amended"]
    assert len(amended) == 2 and amended[0].severity == "warning"
    assert amended[0].fix.startswith("Amended: check the amendment's terms are used")
    assert frozenset(("msa", "am1")) in pairs
    near = kl._check_files(sources, {**texts, "am1": "Amendment No. 1 to TWL-MSA-2025-014. " + BODY}, pairs)
    assert not [f for f in near if f.rule == "near_duplicate" and {x["id"] for x in f.files} == {"msa", "am1"}]


def test_older_sheets_by_year_and_by_title() -> None:
    text = ("=== Sheet: Bands 2025 (12 rows) ===\nSalary bands effective 1 April 2025\n"
            "=== Sheet: Bands 2024 (10 rows) ===\nPrevious bands (1 April 2024) — superseded\n"
            "=== Sheet: Rules (3 rows) ===\nRule | Detail")
    assert lv.older_sheets(text) == [("Bands 2024", "Bands 2025")]
    titled = "=== Sheet: Current (2 rows) ===\nRates\n=== Sheet: Old rates (2 rows) ===\nSuperseded rates | x"
    assert lv.older_sheets(titled) == [("Old rates", "Current")]
    assert lv.older_sheets("=== Sheet: Q1 (2 rows) ===\na\n=== Sheet: Q2 (2 rows) ===\nb") == []
    s = src("sal", "salary_bands_2025.xlsx")
    found, _ = kl._check_versions([s], {"sal": text})
    assert [f.rule for f in found] == ["old_sheet"] and '"Bands 2024"' in found[0].title


def test_year_named_versions_but_not_dated_series() -> None:
    files = [src("a", "salary_bands_2024.xlsx"), src("b", "salary_bands_2025.xlsx"),
             src("c", "Board_minutes_2025-05-22.docx"), src("d", "Board_minutes_2025-09-18.pdf")]
    found = [f for f in kl._check_files(files, {s.id: BODY + s.id for s in files}) if f.rule == "old_version"]
    assert [f.files[0]["name"] for f in found] == ["salary_bands_2024.xlsx"]
    assert "salary_bands_2025.xlsx" in found[0].title


def test_a_lone_draft_is_info_and_a_draft_with_a_final_is_out_of_date() -> None:
    lone = [src("k", "Kestrel_Lane_proposal_v3_DRAFT.docx"), src("o", "Other_proposal.docx")]
    found = kl._check_files(lone, {"k": BODY, "o": BODY.replace("Clause", "Item") + " different words"})
    assert [(f.rule, f.severity) for f in found if f.files[0]["id"] == "k"] == [("draft", "info")]
    pair = [src("k", "Kestrel_Lane_proposal_v3_DRAFT.docx"), src("f", "Kestrel_Lane_proposal_v4.docx")]
    found = kl._check_files(pair, {"k": BODY, "f": BODY.replace("Clause", "Item")})
    olds = [f for f in found if f.rule == "old_version"]
    assert olds and olds[0].files[0]["id"] == "k"


def test_info_findings_are_counted_apart() -> None:
    report = {"findings": [{"rule": "old_version", "ignored": False, "files": [{"name": "old.pdf"}]},
                           {"rule": "old_version", "ignored": True, "files": [{"name": "kept.pdf"}]},
                           {"rule": "draft", "ignored": False, "files": [{"name": "d.docx"}]}]}
    assert kl.superseded_files(report) == ["old.pdf"]


def _args(**kw) -> argparse.Namespace:
    base = {"project": "p", "corpus": None, "accept_superseded": False}
    return argparse.Namespace(**{**base, **kw})


def test_cli_export_refuses_unreviewed_superseded_documents(monkeypatch, tmp_path, capsys) -> None:
    from finetune_studio.cli.commands import brain as cli_brain
    from finetune_studio.cli.commands import dataset

    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(cli_brain, "_corpus", lambda args: tmp_path)
    monkeypatch.setattr(dataset, "resolve_project", lambda ref: {"id": "p"})
    monkeypatch.setattr(kl, "lint", lambda pid: {"findings": [
        {"rule": "old_version", "ignored": False, "files": [{"name": "Employee_Handbook_v1_2023.pdf"}]}]})
    with pytest.raises(SystemExit):
        cli_brain._export(_args())
    assert "Employee_Handbook_v1_2023.pdf" in capsys.readouterr().err

    class Reached(Exception):
        pass

    def stop(*_a, **_k):
        raise Reached

    monkeypatch.setattr(cli_brain, "_read_pass", stop, raising=False)
    from finetune_studio.smart_brain import export_config as ec
    monkeypatch.setattr(ec, "effective_config", stop)
    with pytest.raises((Reached, SystemExit)) as e:
        cli_brain._export(_args(accept_superseded=True, format=None, models=None, encrypt=None, semantic=None,
                                keyword=None, reranker=None, answer=None, llm_base_url=None, llm_model=None,
                                llm_provider=None, llm_connection=None, originals=None, top_k=None, port=None,
                                host=None, device=None, passphrase_file=None))
    assert e.type is Reached
    assert "on purpose (--accept-superseded)" in capsys.readouterr().err
