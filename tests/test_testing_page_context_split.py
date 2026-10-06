"""The Testing page reports 'from memory' and 'with retrieved context' separately (same rule as the wizard)."""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
        / "project_testing.html").read_text(encoding="utf-8")


def test_results_summary_splits_plain_and_grounded_rows() -> None:
    assert "c.transcript[0].role === 'system'" in HTML
    assert "from memory (no context)" in HTML and "answering from retrieved context" in HTML
    assert 'id="case-scores-split"' in HTML
