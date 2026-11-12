"""Page timers stop when the SPA leaves the page (found 2026-11-04, follow-up of #118).

The SPA swaps the page content and re-runs the new page's scripts, but timers and document listeners of the old page
keep running. The Pairs page's Preference and Study set builds polled their progress every second for the whole build
(up to two hours) after the user left; Compare kept refreshing a running comparison every 2 s; Model Library added one
more Escape listener per visit. Rule pinned here: every ``setInterval`` in a page template either checks in its first
lines that its page is still on screen, or is cleared from an ``fts:beforeNavigate`` handler or a still-on-screen check
(directly or through the teardown functions that reach it). ``base.html`` lives for the whole document and is exempt.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
PAGES = sorted(p for p in TEMPLATES.glob("*.html") if p.name != "base.html" and "setInterval(" in p.read_text())

_ASSIGN = re.compile(r"(?:(?:var|let|const)\s+)?([A-Za-z_$][\w$]*)\s*=\s*setInterval\(")
_FUNC = re.compile(r"function\s+([A-Za-z_$][\w$]*)\s*\(|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
                   r"(?:function\s*)?\([^)]*\)\s*(?:=>)?")
_ON_SCREEN = re.compile(r"isConnected|if \(!\$\(|if \(!document\.getElementById")


def _teardowns(src: str) -> str:
    """Text of every fts:beforeNavigate handler (a generous window after each registration)."""
    return "\n".join(src[m.start(): m.start() + 600] for m in re.finditer(r"fts:beforeNavigate", src))


def _owners(src: str, timer: str) -> set[str]:
    """The timer's name, the teardown it is stored in (``unsub = () => clearInterval(timer)``) and the named functions
    that call one of those (``function stop() { ... stopLive() ... }``), followed until nothing new is found."""
    names = {timer}
    for m in re.finditer(r"([A-Za-z_$][\w$]*)\s*=\s*(?:\(\)\s*=>|function\s*\(\)\s*\{)\s*\{?\s*clearInterval\("
                         + re.escape(timer) + r"\)", src):
        names.add(m.group(1))
    funcs = {(m.group(1) or m.group(2)): src[m.end(): m.end() + 400] for m in _FUNC.finditer(src)}
    grew = True
    while grew:
        grew = False
        for fn, body in funcs.items():
            if fn not in names and any(re.search(rf"\b{re.escape(o)}\b", body) for o in names):
                names.add(fn)
                grew = True
    return names


def _guards(src: str) -> str:
    """What runs when a page notices it is gone (``if (!$('x')) { stopWatching(); return; }``)."""
    return "\n".join(src[m.start(): m.start() + 200] for m in _ON_SCREEN.finditer(src))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_timer_stops_when_the_page_is_left(page: Path) -> None:
    src = page.read_text(encoding="utf-8")
    teardown = _teardowns(src) + "\n" + _guards(src)
    for m in _ASSIGN.finditer(src):
        timer = m.group(1)
        head = "\n".join(src[m.end():].splitlines()[:4])
        if _ON_SCREEN.search(head):
            continue
        owners = _owners(src, timer)
        assert any(re.search(rf"\b{re.escape(o)}\b", teardown) for o in owners), (
            f"{page.name}: setInterval stored in {timer!r} neither checks its page is on screen nor is cleared on "
            f"fts:beforeNavigate")


def test_compare_stops_following_a_comparison_when_the_page_is_left() -> None:
    src = (TEMPLATES / "project_compare.html").read_text(encoding="utf-8")
    assert "if (!S.gid || !$('cp-view-card')) return;" in src
    assert "addEventListener('fts:beforeNavigate', function() { clearTimeout(S.timer);" in src


def test_model_library_registers_its_escape_listener_once() -> None:
    src = (TEMPLATES / "hf_models.html").read_text(encoding="utf-8")
    assert src.count("document.addEventListener('keydown'") == 1
    assert "if (!window._hfModalEscListener)" in src
