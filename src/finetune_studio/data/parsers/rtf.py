"""RTF (Rich Text Format) parser."""

from __future__ import annotations

import re
from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from striprtf.striprtf import rtf_to_text
        text = rtf_to_text(raw)
        return make_result(text, {"type": "rtf", "library": "striprtf"}, parser="rtf_v1")
    except ImportError:
        pass
    # Minimal fallback
    text = raw.replace("\\par", "\n").replace("\\pard", "\n")
    text = text.replace("{\\rtf1", "").replace("}", "").replace("{", "")
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return make_result(text, {"type": "rtf", "fallback": "regex"},
                       parser="rtf_v1",
                       warnings=["striprtf not installed; used regex fallback"])


if __name__ == "__main__":
    cli_run(parse)
