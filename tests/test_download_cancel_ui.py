"""The Activity drawer drives POST /api/hf/download/cancel, and a cancelled job reads "cancelled", not "error"."""
from __future__ import annotations

from pathlib import Path

from finetune_studio.webui.routes import activity

_WEB = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"


def test_activity_js_offers_cancel_for_running_downloads_only() -> None:
    src = (_WEB / "static" / "js" / "activity.js").read_text(encoding="utf-8")
    assert '"/api/hf/download/cancel?job_id=" + encodeURIComponent(btn.dataset.dlCancel)' in src
    assert 't.kind === "download" && isActive && t.id' in src
    assert "const isActive = !isDone && !isErr && !isCancelled;" in src
    # a failed cancel says so and gives the button back
    assert "Could not cancel the download" in src and "btn.disabled = false" in src


def test_base_template_loads_the_new_activity_js_and_filters_cancelled() -> None:
    base = (_WEB / "templates" / "base.html").read_text(encoding="utf-8")
    assert "activity.js?v=17" in base
    assert '<option value="cancelled">Cancelled</option>' in base


def test_cancelled_is_its_own_status() -> None:
    assert activity._norm_status("cancelled") == "cancelled"
    assert activity._norm_status("canceled") == "cancelled"
    assert activity._norm_status("failed") == "error"
    css = (_WEB / "static" / "css" / "app.css").read_text(encoding="utf-8")
    assert ".status-cancelled" in css
