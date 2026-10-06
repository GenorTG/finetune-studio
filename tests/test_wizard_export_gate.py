"""Quick-start wizard: the export gate offers its remedies, and the quiz reports both skills."""
from __future__ import annotations

from pathlib import Path

TEMPLATE = (Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
            / "project_wizard.html")


def _html() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def test_blocked_export_offers_force_and_file_library() -> None:
    html = _html()
    # the 409 body names the files; the wizard turns the two remedies into controls
    assert 'id="wiz-dataset-gate"' in html
    assert "r.status === 409" in html and "d.uncovered_files" in html
    assert '"?force=true"' in html and "window.wizExport(true)" in html
    assert "Open file library to delete it" in html


def test_force_is_never_implicit() -> None:
    """Only an explicit click passes force; the chained Run-all path and the plain button do not."""
    html = _html()
    assert 'onclick="wizExport()"' in html
    assert "force === true" in html


def test_quiz_result_separates_memory_from_with_context_rows() -> None:
    html = _html()
    assert 'x.transcript[0].role === "system"' in html
    assert "from memory (no context)" in html
