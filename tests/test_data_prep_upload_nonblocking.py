"""Regression: data-prep file-library upload must not use blocking alert().

A native alert() after upload freezes the page for automation (the OpenClaw
browser tool's upload call times out behind the open dialog) and for real
users it's a modal they must dismiss. The report goes to an inline
role=status element plus a toast instead.
"""
from __future__ import annotations

import re
from pathlib import Path

_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "data_prep.html"
)


def _src() -> str:
    return _TEMPLATE.read_text(encoding="utf-8")


def test_data_prep_has_no_blocking_alerts() -> None:
    assert re.search(r"(?<![\w.])alert\(", _src()) is None


def test_upload_report_renders_inline_status() -> None:
    src = _src()
    assert 'id="fl-upload-status"' in src
    assert 'role="status"' in src
    assert "function flReport(" in src
    body = src.split("function flShowUploadReport(", 1)[1].split("\nfunction ", 1)[0]
    assert "flReport(" in body


def test_upload_input_targetable_by_automation() -> None:
    # The file input keeps a stable id so browser tooling can set files on it.
    assert 'id="fl-upload-input"' in _src()
