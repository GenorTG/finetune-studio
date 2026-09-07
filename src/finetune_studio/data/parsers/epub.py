"""EPUB parser (extracts text from all .xhtml/.html chapters)."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    chapters = []
    warnings = []
    try:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith((".xhtml", ".html", ".htm")):
                    continue
                content = z.read(name).decode("utf-8", errors="replace")
                text = _html_to_text(content)
                if text:
                    chapters.append({"file": name, "text": text, "char_count": len(text)})
    except Exception as e:  # noqa: BLE001
        return make_result("", {"type": "epub", "error": str(e)},
                           parser="epub_v1", warnings=[f"epub parse failed: {e}"])
    text = "\n\n".join(c["text"] for c in chapters)
    structured = {
        "type": "epub",
        "chapter_count": len(chapters),
        "chapters": [{"file": c["file"], "char_count": c["char_count"]} for c in chapters],
    }
    return make_result(text, structured, parser="epub_v1", warnings=warnings)


def _html_to_text(content: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        text = re.sub(r"<script.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


if __name__ == "__main__":
    cli_run(parse)
