"""Static pins for the Guide shell: base.html wiring, CSS, JS contract, cache-busting."""
from __future__ import annotations

import re
from pathlib import Path

W = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"


def _r(rel: str) -> str:
    return (W / rel).read_text(encoding="utf-8")


def test_base_has_a_guide_entry_and_a_panel_outside_content() -> None:
    base = _r("templates/base.html")
    assert 'id="sb-guide"' in base and 'aria-controls="guide-panel"' in base
    panel = base.index('id="guide-panel"')
    # The panel sits in the shell, not inside <main id="content"> (that is swapped on every SPA navigation).
    assert panel < base.index('<main class="content" id="content">')
    for element in ("guide-msgs", "guide-input", "guide-send", "guide-stop", "guide-clear", "guide-close", "guide-ctx"):
        assert f'id="{element}"' in base
    assert '/static/js/guide.js?v=' in base


def test_guide_panel_is_after_the_page_scripts_swap_point_only_in_shell() -> None:
    """spa.js swaps #content and #page-scripts only; neither may contain the panel."""
    base = _r("templates/base.html")
    content_start = base.index('<main class="content" id="content">')
    assert 'id="guide-panel"' not in base[content_start:]


def test_css_is_tokens_only_and_cache_busted() -> None:
    css = _r("static/css/app.css")
    block = css[css.index("GUIDE PANEL"):]
    # No hex colours in the guide block: only var(--token) or rgba() tints.
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b", re.sub(r"/\*.*?\*/", "", block, flags=re.DOTALL))
    for rule in (".guide-pulse", ".guide-prefilled", ".guide-tag", ".guide-tool", "body.guide-open .app"):
        assert rule in block
    assert "prefers-reduced-motion" in block
    assert "app.css?v=82" in _r("templates/base.html")


def test_js_contract() -> None:
    js = _r("static/js/guide.js")
    # Survives SPA navigation (shell lives outside #content) and hard reloads (sessionStorage mirror).
    assert "sessionStorage" in js and "fts:navigated" in js and "ftsSPA.navigate" in js
    assert 'fetch("/api/guide/chat"' in js and "getReader()" in js
    # Every server event type is handled.
    for ev in ("thinking", "tool_call", "tool_result", "ui", "final", "error", "status"):
        assert f'"{ev}"' in js
    # The three UI effects and nothing that submits a form or clicks a control.
    for fn in ("doNavigate", "doHighlight", "doPrefill"):
        assert fn in js
    assert ".click(" not in js and ".submit(" not in js and "requestSubmit" not in js
    # Model text is rendered through text nodes: innerHTML is only ever cleared, never fed model output.
    assert all(m == '""' for m in re.findall(r"innerHTML\s*=\s*([^;]+);", js))
    assert "window.ftsGuide" in js


def test_chat_page_marks_agent_mode_as_the_guide() -> None:
    tpl = _r("templates/chat_v2.html")
    assert "docked Guide" in tpl or "docked <b>Guide</b>" in tpl
    assert "ftsGuide.open()" in tpl
