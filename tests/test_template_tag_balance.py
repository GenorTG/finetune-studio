"""Every template closes its structural tags with the matching tag.

A ``<details>`` card closed by ``</div>`` (data_prep.html, c59e79a) made the browser end ``#content`` early, so the
page's script landed outside it: a full load still ran it, but spa.js only re-runs scripts inside ``#content`` /
``#page-scripts``, and the Pairs page was dead after any in-app navigation (2026-10-17 browser walkthrough).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATES = sorted((Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates").glob("*.html"))
TAGS = r"div|details|summary|section|form|fieldset|table|thead|tbody|nav|main|header|footer|aside|ul|ol|pre|label|select|textarea"


def _blank(m: re.Match[str]) -> str:
    return "\n" * m.group(0).count("\n")   # keep line numbers


def tag_problems(text: str) -> list[str]:
    text = re.sub(r"<script\b.*?</script>", _blank, text, flags=re.S)
    text = re.sub(r"<style\b.*?</style>", _blank, text, flags=re.S)
    text = re.sub(r"\{#.*?#\}", _blank, text, flags=re.S)
    stack: list[tuple[str, int]] = []
    problems: list[str] = []
    for m in re.finditer(rf"<(/?)({TAGS})\b[^>]*?(/?)>", text):
        if m.group(3):
            continue
        tag, line = m.group(2), text[:m.start()].count("\n") + 1
        if not m.group(1):
            stack.append((tag, line))
        elif stack and stack[-1][0] == tag:
            stack.pop()
        else:
            problems.append(f"line {line}: </{tag}> closes <{stack[-1][0]}> from line {stack[-1][1]}" if stack
                            else f"line {line}: stray </{tag}>")
    problems += [f"<{t}> from line {ln} is never closed" for t, ln in stack]
    return problems


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_structural_tags_are_balanced(path: Path) -> None:
    assert tag_problems(path.read_text(encoding="utf-8")) == []


def test_the_checker_catches_a_details_closed_by_div() -> None:
    assert tag_problems('<details class="card"><summary>x</summary><div>y</div></div>') != []


def test_no_table_cell_is_a_flex_container() -> None:
    """display:flex on a <td> takes it out of table layout: its buttons overflowed the card (demo shoot 2026-10-23).
    Put a <div class="flex ..."> inside the cell instead."""
    import re
    root = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"
    bad = [str(f.relative_to(root)) for f in [*root.glob("templates/**/*.html"), *root.glob("static/js/*.js")]
           if re.search(r"<td class=\\?\"[^\"]*\bflex\b", f.read_text(encoding="utf-8"))]
    assert bad == []


def test_brain_status_shows_a_readable_embedder_name() -> None:
    """The index status printed the cache key 'shared:embedder:intfloat__…@<hash>' (demo shoot 2026-10-23)."""
    src = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/brain.html").read_text(encoding="utf-8")
    assert "esc(embedderLabel(d.embedder))" in src and "function embedderLabel(raw)" in src
