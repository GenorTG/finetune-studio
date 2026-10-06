"""File library TYPE column: Office MIME types get the format name, not a 70-character wrapped string.

Found in the live user walkthrough: an uploaded .xlsx/.docx row showed
``VND.OPENXMLFORMATS-OFFICEDOCUMENT.SPREADSHEETML.SHEET`` over eight lines.
"""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
        / "project_data.html").read_text(encoding="utf-8")


def test_office_mime_types_map_to_format_names() -> None:
    for label in ("DOCX", "XLSX", "PPTX", "ODT", "EPUB"):
        assert f"'{label}'" in HTML
    assert "spreadsheetml" in HTML and "wordprocessingml" in HTML


def test_unknown_long_mime_subtypes_are_truncated() -> None:
    assert "raw.length > 14" in HTML
