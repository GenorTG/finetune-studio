"""Regression tests for discrepancies found in the A1 data/parsers audit.

Covers:
  - data/parsers.py (top-level module) was dead/unreachable code, shadowed by
    the data/parsers/ package of the same import name; it duplicated every
    format parser and could never actually run. It has been removed.
  - docx parser silently dropped table content from the plain-text output
    (tables were only written into `structured`, which no caller consumes).
  - json parser raised an unhandled JSONDecodeError on malformed JSON instead
    of degrading to a warning like the sibling xml/jsonl parsers do; this
    crashed whole-project audits (data/audit.py) on a single bad file.
  - email parser could store a non-string `email.message.Message` object in
    the `text` field when `get_body()` was used as the no-body fallback,
    instead of calling `.get_content()` on the returned MIME part.
  - sentence_transformer_local._apply_pooling_compat had an elif branch that
    could never execute (dead code).
"""
from __future__ import annotations

import email.policy
import importlib
import json as _json
from email import message_from_string
from pathlib import Path

import pytest

docx_lib = pytest.importorskip("docx")


def test_top_level_parsers_module_was_removed() -> None:
    """data/parsers.py duplicated the live parsers/ package and could never
    be imported under its own name (the package always shadows it) — it was
    pure dead code. Confirm it no longer exists on disk.
    """
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "src" / "finetune_studio" / "data" / "parsers.py").exists()


def test_parsers_import_resolves_to_the_package() -> None:
    import finetune_studio.data.parsers as m

    importlib.reload(m)
    assert m.__file__.endswith("parsers/__init__.py")
    assert hasattr(m, "get_parser_for")
    assert hasattr(m, "parse")


def test_docx_table_content_is_included_in_text(tmp_path: Path) -> None:
    from finetune_studio.data.parsers.docx import parse as docx_parse

    doc = docx_lib.Document()
    doc.add_paragraph("intro paragraph")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "HeliosMarkUniqueToken42-A"
    table.cell(0, 1).text = "HeliosMarkUniqueToken42-B"
    table.cell(1, 0).text = "row2-a"
    table.cell(1, 1).text = "row2-b"
    path = tmp_path / "with_table.docx"
    doc.save(str(path))

    result = docx_parse(path)
    assert "intro paragraph" in result["text"]
    # Previously only present in structured["tables"], never in text — which
    # means it was unreachable by every downstream consumer (RAG ingest,
    # training-data prep) since they only read result["text"].
    assert "HeliosMarkUniqueToken42-A" in result["text"]
    assert "row2-b" in result["text"]
    assert result["structured"]["table_count"] == 1


def test_json_parser_degrades_on_malformed_json_instead_of_raising(tmp_path: Path) -> None:
    from finetune_studio.data.parsers.json import parse as json_parse

    path = tmp_path / "broken.json"
    path.write_text('{"a": 1, "b": }', encoding="utf-8")

    # Must not raise — previously an unguarded json.loads() propagated
    # JSONDecodeError through the dispatcher into callers such as
    # data/audit.py's audit_project_sources(), crashing the whole batch.
    result = json_parse(path)
    assert result["metadata"]["warnings"]
    assert "parse_error" in result["structured"]
    assert result["text"]  # falls back to raw text, not silently empty


def test_json_parser_still_parses_valid_json(tmp_path: Path) -> None:
    from finetune_studio.data.parsers.json import parse as json_parse

    path = tmp_path / "ok.json"
    path.write_text(_json.dumps({"a": 1}), encoding="utf-8")
    result = json_parse(path)
    assert result["structured"]["schema"] == "object"
    assert not result["metadata"]["warnings"]


def test_dispatcher_survives_one_malformed_json_in_a_batch(tmp_path: Path) -> None:
    """Mirrors how data/audit.py loops `parse_bytes`/`parse` over every
    source in a project with no per-file try/except — a crash on file N
    used to take down the whole batch report.
    """
    from finetune_studio.data.parsers import parse as dispatch_parse

    good = tmp_path / "good.json"
    good.write_text(_json.dumps({"ok": True}), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    results = [dispatch_parse(p) for p in (good, bad)]
    assert results[0]["structured"]["schema"] == "object"
    assert "parse_error" in results[1]["structured"]


def test_email_parser_returns_string_text_when_no_plain_part_has_content() -> None:
    from finetune_studio.data.parsers.email import parse as email_parse

    # A non-multipart text/plain body that is whitespace-only is skipped by
    # the `body.strip()` check, so parts_text stays empty and the code falls
    # through to the get_body() path — which previously stored the raw
    # EmailMessage *object* into "text" instead of its decoded string.
    raw = (
        "Subject: Test\r\n"
        "From: a@b.com\r\n"
        "To: c@d.com\r\n"
        "Content-Type: text/plain\r\n"
        "\r\n"
        "   \r\n"
    )
    msg = message_from_string(raw, policy=email.policy.default)
    assert msg.get_body(preferencelist=("plain",)) is not None  # sanity: triggers the fallback path

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".eml", mode="w", delete=False, encoding="utf-8") as f:
        f.write(raw)
        tmp_path = Path(f.name)
    try:
        result = email_parse(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    assert isinstance(result["text"], str)
    # Must be JSON-serializable (an EmailMessage object is not).
    _json.dumps(result)


def test_pooling_compat_cls_token_branch_is_reachable_and_not_duplicated() -> None:
    from finetune_studio.data.sentence_transformer_local import _apply_pooling_compat

    data = {"pooling_mode_cls_token": True}
    changed = _apply_pooling_compat(data, 0)
    assert changed
    assert data["pooling_mode"] == "cls"

    data2 = {"pooling_mode_max_tokens": True}
    _apply_pooling_compat(data2, 0)
    assert data2["pooling_mode"] == "max"

    data3: dict = {}
    _apply_pooling_compat(data3, 0)
    assert data3["pooling_mode"] == "mean"
