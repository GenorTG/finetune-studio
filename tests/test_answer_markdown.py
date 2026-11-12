"""Answers show their Markdown (**bold**, `code`, headings, lists) instead of raw asterisks, in the studio's Smart Brain
chat and on a package's own chat page; the text is escaped first, so a document can never inject HTML."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "finetune_studio"
BRAIN = (ROOT / "webui" / "templates" / "brain.html").read_text()
PKG = (ROOT / "smart_brain" / "standalone_server.py").read_text()


def _md_inline(source: str) -> str:
    return re.search(r"const mdInline = \(html\) => String\(html\)\n(?:\s+\.replace\(.*\)\n?)+;?", source).group(0)


def test_both_chats_escape_first_then_render() -> None:
    assert "citeLinks(mdInline(esc(reply)), msgId, groups)" in BRAIN
    assert "if (cls === 'bot') d.innerHTML = mdInline(esc(text)); else d.textContent = text;" in PKG
    assert _md_inline(BRAIN).split("\n", 1)[1].split() == _md_inline(PKG).split("\n", 1)[1].split()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_md_inline_output() -> None:
    esc = ("const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')"
           ".replace(/\"/g, '&quot;').replace(/'/g, '&#39;');\n")
    script = esc + _md_inline(PKG).rstrip(";") + ";\n" + (
        "process.stdout.write(mdInline(esc('### Changes\\n- **Scope:** <b>x</b> `TWL-1`\\n3 * 4 = 12')));")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert out == ("<strong>Changes</strong>\n• <strong>Scope:</strong> &lt;b&gt;x&lt;/b&gt; <code>TWL-1</code>\n"
                   "3 * 4 = 12")
