"""OpenDocument (ODT/ODS/ODP) parser."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ._base import cli_run, make_result


_CONTENT_FILES = {
    ".odt": "content.xml",
    ".ods": "content.xml",
    ".odp": "content.xml",
}


def parse(path: Path) -> dict:
    warnings = []
    ext = path.suffix.lower()
    try:
        with zipfile.ZipFile(path) as z:
            content_path = _CONTENT_FILES.get(ext)
            if not content_path:
                return make_result("", {"type": ext.lstrip("."), "error": "no content.xml mapping"},
                                   parser="odf_v1", warnings=[f"unsupported odf extension: {ext}"])
            content = z.read(content_path).decode("utf-8", errors="replace")
            root = ET.fromstring(content)
            texts = []
            for elem in root.iter():
                if elem.tag.endswith("}p") or elem.tag.endswith("}h"):
                    t = "".join(elem.itertext()).strip()
                    if t:
                        texts.append(t)
            text = "\n".join(texts)
            structured = {"type": ext.lstrip("."), "paragraph_count": len(texts), "extracted_elements": len(texts)}
            return make_result(text, structured, parser="odf_v1")
    except Exception as e:  # noqa: BLE001
        return make_result("", {"type": ext.lstrip("."), "error": str(e)},
                           parser="odf_v1", warnings=[f"ODF parse failed: {e}"])


if __name__ == "__main__":
    cli_run(parse)
