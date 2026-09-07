"""Image parser — PNG, JPG, JPEG, TIFF, BMP, WEBP, GIF.

Routes through OCR. Uses tessdata_fast eng+pol by default (a few MB combined).
"""

from __future__ import annotations

from pathlib import Path

from ..ocr import DEFAULT_LANGS, ocr_image
from ._base import cli_run, make_result


SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def parse(path: Path, languages: str = DEFAULT_LANGS) -> dict:
    warnings = []
    try:
        text = ocr_image(path, languages=languages)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"OCR failed: {e}")
        text = ""
    # Best-effort image metadata
    width, height, fmt = None, None, None
    try:
        from PIL import Image
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format
    except Exception:
        pass
    structured = {
        "type": "image",
        "image_format": fmt,
        "width": width,
        "height": height,
        "ocr_languages": languages,
    }
    return make_result(text, structured, parser=f"image_v1_ocr_{languages}",
                       warnings=warnings, languages=languages)


if __name__ == "__main__":
    cli_run(parse)
