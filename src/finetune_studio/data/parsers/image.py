"""Image parser — PNG, JPG, JPEG, TIFF, BMP, WEBP, GIF.

Routes through OCR. Uses tessdata_fast eng+pol by default (a few MB combined).
"""

from __future__ import annotations

from pathlib import Path

from .. import figures as figs
from ..ocr import DEFAULT_LANGS
from ._base import cli_run, make_result


def parse(path: Path, languages: str = DEFAULT_LANGS) -> dict:
    warnings: list[str] = []
    found = figs.extract(path)
    if not found:
        warnings.append("not a readable image")
    # the whole image is one figure: a screenshot of a table or a chart needs the vision reader, OCR is a first pass
    text = "\n\n".join(figs.ocr_block(f, languages, warnings) for f in found)  # a multi-page TIFF: one block per page
    # Best-effort image metadata
    width, height, fmt = None, None, None
    try:
        from PIL import Image
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format
    except Exception as e:  # noqa: BLE001 - metadata is best-effort; PIL raises many unrelated types
        warnings.append(f"image metadata unavailable: {e}")
    structured = {
        "type": "image",
        "image_format": fmt,
        "width": width,
        "height": height,
        "ocr_languages": languages,
        "figures": figs.summary(found),
    }
    return make_result(text, structured, parser=f"image_v1_ocr_{languages}",
                       warnings=warnings, languages=languages)


if __name__ == "__main__":
    cli_run(parse)
