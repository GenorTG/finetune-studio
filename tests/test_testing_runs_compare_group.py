"""The Testing page lists the runs of one comparison as a group (kind=compare rows share config.compare.group_id)."""
from __future__ import annotations

from pathlib import Path

from finetune_studio.webui import (
    app as _app,  # noqa: F401  (routes import in app order)
)
from finetune_studio.webui.routes.testing import _compare_membership, _run_summary

TEMPLATE = Path(__file__).resolve().parent.parent / "src" / "finetune_studio" / "webui" / "templates" / "project_testing.html"


def test_run_summary_carries_the_comparison_membership_only_for_compare_rows() -> None:
    row = {"id": "b1", "kind": "compare", "config": {"compare": {"group_id": "g1", "index": 1, "label": "Base", "models": ["Base", "Tuned"]}}}
    assert _compare_membership(row) == {"group_id": "g1", "label": "Base", "index": 1, "models": ["Base", "Tuned"]}
    assert _run_summary(row)["compare"]["group_id"] == "g1"
    assert _compare_membership({"id": "b2", "kind": "suite", "config": {"compare": {"group_id": "g1"}}}) is None
    assert _compare_membership({"id": "b3", "kind": "compare", "config": {}}) is None
    assert _run_summary({"id": "b4", "kind": "rag"})["compare"] is None


def test_runs_table_renders_a_group_header_linking_to_the_compare_page() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "tp-group" in html and "/compare?group=" in html and "open side by side" in html
    assert "r.compare" in html and "tp-in-group" in html
