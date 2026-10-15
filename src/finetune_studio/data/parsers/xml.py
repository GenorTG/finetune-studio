"""XML parser."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        xml_text = ET.tostring(root, encoding="unicode")
        # Walk and extract text per element
        elements = []
        for elem in root.iter():
            text = "".join(elem.itertext()).strip()
            if text:
                elements.append({"tag": _strip_ns(elem.tag), "text": text})
        structured = {
            "type": "xml",
            "root_tag": _strip_ns(root.tag),
            "element_count": len(elements),
            "elements": elements[:200],
        }
        return make_result(xml_text, structured, parser="xml_v1")
    except ET.ParseError as e:
        # Fall back to raw text
        return make_result(raw, {"type": "xml", "parse_error": str(e), "fallback": "text"},
                           parser="xml_v1", warnings=[f"XML parse failed: {e}"])


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


if __name__ == "__main__":
    cli_run(parse)
