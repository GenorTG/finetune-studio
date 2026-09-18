"""Legacy .doc (Word 97-2003) parser.

Fallback chain: antiword → catdoc → textract → pure-Python olefile scan.
"""
from __future__ import annotations

import contextlib
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
        except Exception:  # noqa: BLE001 -textract raises broadly on bad files
            text = ""
    if not text:
        text = _olefile_extract(path)
        if text:
            method = "olefile-scan"
    warnings = []
    if not text:
        text = f"[DOC: {path.name} — install antiword (apt install antiword) or textract to read]"
        warnings.append("no legacy-DOC parser available; placeholder returned")
    structured = {"type": "doc", "extraction_method": method}
    return make_result(text, structured, parser="doc_v1", warnings=warnings)


def _olefile_extract(path: Path) -> str:
    """Pure-Python legacy .doc text extraction via olefile + raw decode.

    Scans the WordDocument stream for runs of printable text, stopping per
    at control chars. Quality is below antiword but far above a placeholder:
    body text survives well enough for chunking + Q&A mining.
    """
    try:
        import olefile
        ole = olefile.OleFileIO(str(path))
    except Exception:  # noqa: BLE001 — not an OLE2 file, or olefile missing
        return ""
    try:
        stream_name = next(
            (n for n in ole.listdir()
             if "WordDocument" in "".join(n)), None)
        if stream_name is None:
            return ""
        data = ole.openstream(stream_name).read()
    except Exception:  # noqa: BLE001
        return ""
    finally:
        with contextlib.suppress(Exception):
            ole.close()

    runs: list[str] = []
    cur: list[str] = []
    # Modern Word 97+ stores body text as UTF-16LE primarily — scan both
    # byte-wise (CP1252-ish) and 16-bit-wise, keeping the richer harvest.
    i = 0
    while i + 1 < len(data):
        ch = data[i] | (data[i + 1] << 8)
        # ASCII range + typical typographic codepoints
        if (0x20 <= ch < 0x7F) or ch in (0x2014, 0x2013, 0x2019, 0x201C,
                                        0x201D, 0x2026, 0x00E9, 0x00F6):
            cur.append(chr(ch))
            i += 2
            continue
        if len(cur) >= 8:
            runs.append("".join(cur).strip())
        cur = []
        i += 1
    if len(cur) >= 8:
        runs.append("".join(cur).strip())
    # Keep the fat text runs: drop binary noise (< 12 chars)
    body = "\n".join(r for r in runs if len(r) >= 12)
    return body


def _try_cli(path: Path, name: str, cmd: list[str]) -> tuple[str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout, name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "", name + "-failed"


if __name__ == "__main__":
    cli_run(parse)
