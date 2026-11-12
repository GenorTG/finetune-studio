"""Focus, #anchor jumps and scrollIntoView land below the sticky session bar, not under it (WCAG 2.4.11).

Found by the B3 control sweep: Settings' Local network switch, scrolled to the top of the window, sat under the
104 px sticky bar and a click hit the bar's Models tab instead.
"""
from __future__ import annotations

from pathlib import Path

_WEB = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"


def test_page_scroll_padding_follows_the_bar_height() -> None:
    css = (_WEB / "static" / "css" / "app.css").read_text(encoding="utf-8")
    assert "html { scroll-padding-top: calc(var(--session-bar-h, 104px) + 8px); }" in css


def test_nav_script_measures_the_sticky_bar() -> None:
    js = (_WEB / "static" / "js" / "nav_overflow.js").read_text(encoding="utf-8")
    assert 'setProperty("--session-bar-h", h + "px")' in js
    assert 'pos === "sticky" || pos === "fixed"' in js     # a bar that does not stick reserves nothing
    assert "new ResizeObserver(apply).observe(bar)" in js   # the bar grows when its tab row wraps
    assert "trackBarHeight();" in js


def test_base_loads_the_new_assets() -> None:
    base = (_WEB / "templates" / "base.html").read_text(encoding="utf-8")
    assert "app.css?v=135" in base and "nav_overflow.js?v=3" in base
