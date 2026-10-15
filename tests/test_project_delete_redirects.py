"""Deleting a project from its overview must leave the page, not reload it (the reload 404s on the deleted id)."""
from __future__ import annotations

from pathlib import Path

WEBUI = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"


def test_overview_delete_button_redirects_to_the_projects_list() -> None:
    html = (WEBUI / "templates" / "project.html").read_text(encoding="utf-8")
    line = next(ln for ln in html.splitlines() if 'data-method="DELETE"' in ln and "Delete this project" in ln)
    assert 'data-redirect="/projects"' in line and "data-reload" not in line


def test_action_handler_honours_data_redirect() -> None:
    js = (WEBUI / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert js.count("btn.dataset.redirect") >= 4          # both click handlers read and use it
    assert "location.assign(btn.dataset.redirect)" in js
