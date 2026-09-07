"""Legacy .doc (Word 97-2003) parser. Shells out to antiword / catdoc / textract."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ._base import cli_run, make_result


def parse(path: Path) -> dict:
    text, method = _try_cli(path, "antiword", ["antiword", str(path)])
    if not text:
        text, method = _try_cli(path, "catdoc", ["catdoc", str(path)])
    if not text:
        try:
            import textract
            text = textract.process(str(path)).decode("utf-8", errors="replace")
            method = "textract"
        except (ImportError, Exception):
            text = ""
    warnings = []
    if not text:
        text = f"[DOC: {path.name} — install antiword (apt install antiword) or textract to read]"
        warnings.append("no legacy-DOC parser available; placeholder returned")
    structured = {"type": "doc", "extraction_method": method}
    return make_result(text, structured, parser="doc_v1", warnings=warnings)


def _try_cli(path: Path, name: str, cmd: list[str]):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout, name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "", name + "-failed"


if __name__ == "__main__":
    cli_run(parse)
