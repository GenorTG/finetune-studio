"""Badges/buttons that read below 4.5:1 in the live visual audit use the theme's text tokens.

--warn / --err are FILL colours; base.html defines darker --warn-text / --err-text / --cyan-text for
the light theme (dark theme falls back to the bright fill). Live audit (2026-10-06): the partial
verdict badge (3.99), the pending status (3.99/3.28), the data-prep question text (4.17) and the
danger button on dark (4.01) were all painted straight from the fill colour.
"""
from __future__ import annotations

import re
from pathlib import Path

WEBUI = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"
CSS = (WEBUI / "static" / "css" / "app.css").read_text(encoding="utf-8")
DATA_PREP = (WEBUI / "templates" / "data_prep.html").read_text(encoding="utf-8")


def _rule(css: str, selector: str) -> str:
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert m, f"rule not found: {selector}"
    return m.group(1)


def test_verdict_partial_text_uses_warn_text() -> None:
    assert "var(--warn-text" in _rule(CSS, ".verdict-badge.verdict-partial")


def test_danger_button_text_uses_err_text() -> None:
    assert "var(--err-text" in _rule(CSS, ".btn.danger")


def test_data_prep_pending_and_question_text_use_theme_text_tokens() -> None:
    assert "var(--warn-text" in _rule(DATA_PREP, ".dp-status-pending")
    assert "var(--cyan-text" in _rule(DATA_PREP, ".dp-results-table .q")


def test_data_prep_table_headers_are_not_tiny() -> None:
    m = re.search(r"font-size:\s*(\d+)px", _rule(DATA_PREP, ".dp-results-table .meta"))
    assert m and int(m.group(1)) >= 11
