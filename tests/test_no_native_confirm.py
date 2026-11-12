"""Confirmations use the app's own dialog (``fts.confirm``), never the browser's bare ``confirm()``.

Model Library's Delete and the Testing page's Delete run / Judge all again opened the native dialog: unstyled, it names
the host instead of the action, has no danger colour, and blocks the whole tab while it is open.
"""
from __future__ import annotations

import re
from pathlib import Path

WEBUI = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui"
_BARE = re.compile(r"(?<![\w.$])confirm\(")


def test_no_page_calls_the_native_confirm() -> None:
    hits = []
    for path in [*WEBUI.glob("templates/**/*.html"), *WEBUI.glob("static/js/**/*.js")]:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _BARE.search(line) and "function confirm" not in line:
                hits.append(f"{path.relative_to(WEBUI)}:{n}: {line.strip()[:100]}")
    assert not hits, "use fts.confirm(message, {title, danger, okText}):\n" + "\n".join(hits)
