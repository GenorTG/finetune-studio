"""The title mining questions are scoped by (``statements.document_title``): the document's own name, not its file name."""
from __future__ import annotations

from finetune_studio.data.mining.statements import document_title, title_from_filename

CAR = ("[Figure 1 · page 1 · scanned page · read by gemma-4-12b-it-qat-q4_0.gguf]\n"
       "KORVANE COLD CHAIN - CORRECTIVE ACTION REQUEST - form KC-QF-131 rev D (paper copy, signed) CAR no. KC-CAR-24-088. "
       "Opened: 17.09.2024. Raised by: B. Aaltonen (Quality). Action owner: J. Lindqvist (Head of Ops Gdynia).")


def test_scanned_form_read_as_one_line_is_named_by_heading_and_record_number() -> None:
    assert document_title(CAR, "scanned_corrective_action_0917.pdf") == \
        "Korvane Cold Chain - Corrective Action Request KC-CAR-24-088"


def test_reader_markers_are_skipped() -> None:
    pptx = "=== Slide 1 ===\nKorvane Cold Chain Sp. z o.o. — Board Pack Q3 2024\nManagement accounts to 30 September 2024."
    assert document_title(pptx, "board_pack_q3_2024.pptx") == "Korvane Cold Chain Sp. z o.o. — Board Pack Q3 2024"


def test_email_is_named_by_its_subject_without_reply_prefixes() -> None:
    eml = ("From: B. Aaltonen <b@korvane.example>\nTo: S. Karpinski <s@oakhaven.example>\n"
           "Subject: RE: RE: Fwd: OD-231 / night of 13-14 June - your questions\n\nHello Szymon,")
    assert document_title(eml, "thread_oakhaven_incident_2024_06.eml") == "OD-231 / night of 13-14 June - your questions"


def test_plain_title_line_and_filename_fallback_are_unchanged() -> None:
    assert document_title("# RB-017 — KC-Hub 200 gateway offline\n\n> owner: platform", "runbook.md") == \
        "RB-017 — KC-Hub 200 gateway offline"
    table = "opp_id | account | stage\n--- | --- | ---\nOPP-24-0301 | Fruttoria | won"
    assert document_title(table, "crm_export_2024_q3.csv") == "Crm Export 2024 Q3"
    # a long line without a caps heading still falls back to the file name
    prose = "this memo explains " + "the plan in some detail " * 8
    assert document_title(prose, "q3_plan_0917.txt") == "Q3 Plan"


def test_filename_title_keeps_years_but_drops_other_numbers_and_hashes() -> None:
    assert title_from_filename("board_pack_q3_2024.pptx") == "Board Pack Q3 2024"
    assert title_from_filename("scan_0917.pdf") == "Scan"
    assert title_from_filename("3f9a2c7b1e.pdf") == ""
