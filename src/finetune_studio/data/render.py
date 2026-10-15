"""Render a document with LibreOffice (headless): the fallback for formats without a parser, or whose parser found no text.

Genor 2026-10-14: "default to opening the file and a vision model analysing its contents if it's unsupported yet". The
order (``parsers.parse``): native parser -> if it has no parser (a binary file) or its text is thin, render to PDF here and
read every page as a figure (``data/figures.py``: the Vision reader during mining, OCR without one; picture-like pages are
held back without a reader) -> if LibreOffice cannot render it either, fail with the reason.

Each conversion runs in its own LibreOffice profile directory: a desktop LibreOffice the user has open holds the default
profile's lock, and a second headless instance on the same profile exits silently with no output.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

TIMEOUT_S = 180
# Formats LibreOffice is worth trying on when no parser knows them (drawings, Visio, legacy/other office, publisher).
RENDERABLE = frozenset({
    ".odg", ".fodg", ".vsd", ".vsdx", ".vdx", ".pub", ".wpd", ".wps", ".wk1", ".wks", ".123", ".cdr", ".svg", ".emf",
    ".wmf", ".pptm", ".ppt", ".pps", ".ppsx", ".potx", ".pot", ".dotx", ".dot", ".docm", ".xlsm", ".xlsb", ".xlt",
    ".xltx", ".odt", ".ods", ".odp", ".fodt", ".fods", ".fodp", ".rtf", ".doc", ".docx", ".xls", ".xlsx", ".pptx", ".sxw",
    ".sxc", ".sxi", ".sxd", ".abw", ".zabw", ".hwp", ".lwp", ".key", ".numbers", ".pages", ".mml", ".dif", ".slk",
})


class RenderError(RuntimeError):
    pass


def soffice() -> str | None:
    return shutil.which("soffice") or shutil.which("libreoffice")


def install_hint() -> str:
    return "install LibreOffice (Debian/Ubuntu: sudo apt-get install -y libreoffice-core libreoffice-writer " \
           "libreoffice-calc libreoffice-impress libreoffice-draw)"


def convert(path: Path, target: str, out_dir: Path) -> Path:
    """``path`` converted by LibreOffice to ``target`` ("pdf", "xlsx", ...) inside ``out_dir``. Raises ``RenderError``."""
    exe = soffice()
    if not exe:
        raise RenderError(f"LibreOffice not found on PATH: {install_hint()}")
    with tempfile.TemporaryDirectory(prefix="_lo_profile_") as profile:
        cmd = [exe, f"-env:UserInstallation=file://{profile}", "--headless", "--norestore", "--convert-to", target,
               "--outdir", str(out_dir), str(path)]
        env = {**os.environ, "HOME": profile}  # LibreOffice also writes caches under $HOME
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=env, check=False)
        except subprocess.TimeoutExpired:
            raise RenderError(f"LibreOffice did not finish converting {path.name} in {TIMEOUT_S} s") from None
    out = out_dir / f"{path.stem}.{target.split(':', 1)[0]}"
    if r.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        detail = " ".join((r.stderr or r.stdout or "").split())[:300] or "no output file"
        raise RenderError(f"LibreOffice could not convert {path.name} to {target}: {detail}")
    return out
