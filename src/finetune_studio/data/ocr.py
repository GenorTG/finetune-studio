"""Lightweight OCR helper.

Uses tesseract (already on most systems) via pytesseract.

Language support: English + Polish out of the box (tessdata_fast is tiny — a few MB).
Adds a project-side install fallback so we never fail just because the system
package wasn't installed. If user runs:
    python -m finetune_studio.data.ocr install
we'll download eng.traineddata + pol.traineddata from tessdata_fast and put them in
~/.local/share/tessdata/ (where tesseract also picks them up via TESSDATA_PREFIX).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_USER_TESSDATA = Path.home() / ".local" / "share" / "tessdata"
_DOWNLOAD_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main"

# Default bilingual: English + Polish. Override per call.
DEFAULT_LANGS = "eng+pol"


def tessdata_dir() -> Path:
    """Where we look for user-installed tessdata. Creates the dir if needed."""
    _USER_TESSDATA.mkdir(parents=True, exist_ok=True)
    return _USER_TESSDATA


def install(languages: list[str] | None = None) -> dict:
    """Download tessdata_fast .traineddata files for the given languages.

    Returns {"ok": [...], "skipped": [...], "errors": [...]} so the caller can report.
    Use:
        python -m finetune_studio.data.ocr install
    """
    languages = languages or ["eng", "pol"]
    out = {"ok": [], "skipped": [], "errors": []}
    d = tessdata_dir()
    for lang in languages:
        dest = d / f"{lang}.traineddata"
        if dest.exists() and dest.stat().st_size > 100_000:
            out["skipped"].append(lang)
            continue
        url = f"{_DOWNLOAD_URL}/{lang}.traineddata"
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            out["ok"].append(lang)
        except Exception as e:  # noqa: BLE001
            out["errors"].append({"lang": lang, "error": str(e)})
    return out


def _tessdata_prefix() -> str:
    """Return the directory tesseract should look in for .traineddata files."""
    # System tessdata is /usr/share/tessdata (on Garuda/Arch) — but if user has
    # no sudo we can't install there. Always point tesseract at our user dir.
    return str(tessdata_dir()) + "/"


def _tesseract_cmd() -> Optional[list[str]]:
    """Find the tesseract binary, or None if not installed."""
    p = shutil.which("tesseract")
    return [p] if p else None


def is_available() -> bool:
    """Is tesseract usable from this app right now?"""
    if not _tesseract_cmd():
        return False
    # Quick sanity check — try listing languages via TESSDATA_PREFIX
    try:
        env = os.environ.copy()
        env["TESSDATA_PREFIX"] = _tessdata_prefix()
        r = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        # Sometimes --list-langs writes to stderr, sometimes hangs on missing dir
        combined = (r.stdout + "\n" + r.stderr).lower()
        return any(code in combined for code in ("eng", "pol"))
    except Exception:
        return False


def installed_languages() -> list[str]:
    """List languages whose .traineddata is present in our user tessdata dir."""
    d = tessdata_dir()
    return sorted(p.stem for p in d.glob("*.traineddata"))


def ocr_image(image_path: str | Path, languages: str = DEFAULT_LANGS,
              psm: int = 3, oem: int = 1) -> str:
    """OCR a single image. Returns extracted text.

    psm 3 = fully automatic page segmentation (default)
    oem 1 = LSTM neural net only (fast)
    """
    cmd = _tesseract_cmd()
    if not cmd:
        raise RuntimeError("tesseract binary not found on PATH")
    env = os.environ.copy()
    env["TESSDATA_PREFIX"] = _tessdata_prefix()
    out_base = Path("/tmp") / f"_ocr_{os.getpid()}_{abs(hash(str(image_path))) % 100000}"
    out_path = Path(str(out_base) + ".txt")
    try:
        r = subprocess.run(
            cmd + [str(image_path), str(out_base), "-l", languages, "--psm", str(psm), "--oem", str(oem)],
            capture_output=True, text=True, timeout=180, env=env,
        )
        if r.returncode != 0:
            raise RuntimeError(f"tesseract failed: {r.stderr.strip()}")
        return out_path.read_text(encoding="utf-8", errors="replace")
    finally:
        # Clean up the temp output file tesseract writes
        try:
            out_path.unlink()
        except OSError:
            pass


def ocr_image_object(img, languages: str = DEFAULT_LANGS,
                    psm: int = 3, oem: int = 1) -> str:
    """OCR a PIL.Image directly (no disk roundtrip)."""
    try:
        import pytesseract
    except ImportError as e:
        raise RuntimeError("pytesseract not installed: pip install pytesseract") from e
    # pytesseract respects TESSDATA_PREFIX env var
    old_prefix = os.environ.get("TESSDATA_PREFIX")
    os.environ["TESSDATA_PREFIX"] = _tessdata_prefix()
    try:
        return pytesseract.image_to_string(img, lang=languages, config=f"--psm {psm} --oem {oem}")
    finally:
        if old_prefix is None:
            os.environ.pop("TESSDATA_PREFIX", None)
        else:
            os.environ["TESSDATA_PREFIX"] = old_prefix


def ocr_pdf(pdf_path: str | Path, languages: str = DEFAULT_LANGS, dpi: int = 200) -> list[dict]:
    """OCR every page of a PDF. Returns [{"page": i, "text": "..."}, ...]."""
    try:
        from pdf2image import convert_from_path
    except ImportError as e:
        raise RuntimeError("pdf2image not installed: pip install pdf2image") from e
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        raise RuntimeError(f"pdf2image failed (poppler not installed?): {e}") from e
    out = []
    for i, img in enumerate(images, 1):
        try:
            text = ocr_image_object(img, languages=languages)
            out.append({"page": i, "text": text.strip()})
        except Exception as e:  # noqa: BLE001
            log.warning("OCR failed on page %d: %s", i, e)
            out.append({"page": i, "text": "", "error": str(e)})
    return out


# CLI: python -m finetune_studio.data.ocr install|status|test
def _cli() -> int:
    import argparse
    import sys
    p = argparse.ArgumentParser(prog="finetune_studio.data.ocr")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install", help="Download tessdata_fast for eng+pol")
    sub.add_parser("status", help="Show OCR availability + installed languages")
    sub.add_parser("test", help="Generate a test image and OCR it")
    args = p.parse_args()
    if args.cmd == "install":
        result = install()
        print(result)
        return 0 if not result.get("errors") else 1
    if args.cmd == "status":
        print({
            "tesseract_binary": _tesseract_cmd(),
            "tessdata_dir": str(tessdata_dir()),
            "installed_languages": installed_languages(),
            "available": is_available(),
        })
        return 0
    if args.cmd == "test":
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        img = Image.new("RGB", (700, 100), "white")
        ImageDraw.Draw(img).text((10, 30), "Hello OCR World! Zażółć gęślą jaźń.", fill="black", font=font)
        path = Path("/tmp/_ocr_smoke.png")
        img.save(path)
        for lang in ("eng", "pol", "eng+pol"):
            print(f"{lang:8s} -> {ocr_image(path, languages=lang)!r}")
        return 0
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
