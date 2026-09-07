"""HTML / XHTML parser."""

from __future__ import annotations

import re
from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        # Strip non-content tags
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Extract a structured view
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        headings = []
        for level in range(1, 7):
            for h in soup.find_all(f"h{level}"):
                t = h.get_text(strip=True)
                if t:
                    headings.append({"level": level, "text": t})
        links = []
        for a in soup.find_all("a", href=True):
            links.append({"text": a.get_text(strip=True), "href": a["href"]})
        structured = {
            "type": "html",
            "title": title,
            "headings": headings[:100],
            "link_count": len(links),
        }
        return make_result(text, structured, parser="html_v1")
    except ImportError:
        # Fallback without beautifulsoup
        text = re.sub(r"<script.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return make_result(text, {"type": "html", "fallback": "regex"},
                           parser="html_v1", warnings=["beautifulsoup4 not installed; using regex fallback"])


if __name__ == "__main__":
    cli_run(parse)
