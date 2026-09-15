"""Static assertions for app.js delegate() (E2E-10 / E2E-11) + data_prep form."""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_APP_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "app.js"
_DATA_PREP = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "data_prep.html"


def _app_js() -> str:
    return _APP_JS.read_text(encoding="utf-8")


def _data_prep() -> str:
    return _DATA_PREP.read_text(encoding="utf-8")


def test_doc_click_listeners_once_guard() -> None:
    src = _app_js()
    assert "_docClickListenersBound" in src
    assert re.search(
        r"if\s*\(\s*!_docClickListenersBound\s*\)",
        src,
    ), "document click listeners must be gated by a once-guard"


def test_json_form_branch_has_catch() -> None:
    src = _app_js()
    # Locate the JSON (non-multipart) form branch around api.post
    idx = src.find("api.post(url, data)")
    assert idx != -1, "expected api.post(url, data) in form handler"
    window = src[idx : idx + 400]
    assert ".catch(" in window, "JSON form branch must notify on rejection"


def test_data_action_checks_ok() -> None:
    src = _app_js()
    # data-action path should use _ok (checks !r.ok) or an explicit r.ok check
    assert "closest(\"[data-action]\")" in src or "closest('[data-action]')" in src
    # After the data-action fetch, either .then(_ok) or r.ok
    action_idx = src.find('closest("[data-action]")')
    if action_idx < 0:
        action_idx = src.find("closest('[data-action]')")
    window = src[action_idx : action_idx + 900]
    assert (
        ".then(_ok)" in window
        or "r.ok" in window
        or "!r.ok" in window
    ), "data-action branch must check response ok / use _ok"


def test_data_prep_no_fake_starting_timeout() -> None:
    src = _data_prep()
    # No setTimeout fake reset tied to 'Starting…'
    assert not re.search(
        r"Starting…[\s\S]{0,200}setTimeout\(",
        src,
    )
    assert not re.search(
        r"setTimeout\([\s\S]{0,200}Starting…",
        src,
    )


def test_data_prep_status_and_eventsource() -> None:
    src = _data_prep()
    assert 'id="prep-status"' in src
    assert 'role="status"' in src
    assert "EventSource" in src
    assert "/data-prep/runs/" in src
