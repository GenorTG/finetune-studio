"""The OCR built-in service must work for data ingestion on a fresh box.

Covers:
- ``install_hint()`` returns a platform-appropriate install command;
- ``is_available(languages)`` correctly reports which languages are usable;
- ``_ensure_tessdata()`` self-installs missing languages when
  ``FTS_OCR_AUTOINSTALL=1`` (the default);
- ``ocr_image()`` raises an actionable ``RuntimeError`` naming the install
  command when the tesseract binary itself is missing;
- the image parser returns extracted text for a real PNG round-trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont


def test_install_hint_is_a_real_command() -> None:
    from finetune_studio.data import ocr

    hint = ocr.install_hint()
    assert hint
    # Every documented hint starts with a verb the user can run.
    assert any(
        hint.startswith(verb)
        for verb in ("brew install", "sudo apt", "sudo pacman", "sudo dnf", "Download")
    )


def test_installed_languages_reports_what_tesseract_can_use() -> None:
    from finetune_studio.data import ocr

    installed = ocr.installed_languages()
    # Either tesseract + tessdata are present (we expect eng+pol in CI), or
    # the function still returns a list (possibly empty) without raising.
    assert isinstance(installed, list)
    if ocr.is_available():
        assert "eng" in installed


def test_is_available_matches_installed_languages() -> None:
    from finetune_studio.data import ocr

    if not ocr.is_available():
        pytest.skip("tesseract binary not installed in this env")
    # Asking for an installed language should report available; asking for
    # a non-installed one should not.
    have = set(ocr.installed_languages())
    if "eng" in have:
        assert ocr.is_available("eng")
    assert not ocr.is_available("zz_NOPE_DOES_NOT_EXIST")


def test_ensure_tessdata_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio import data as data_pkg  # noqa: F401
    from finetune_studio.data import ocr

    if not ocr.is_available():
        pytest.skip("tesseract binary not installed in this env")
    # First call with languages we have: no-op, must not raise.
    ocr._ensure_tessdata("eng")
    # Calling twice in a row must be safe (the AUTO_INSTALL guard).
    ocr._ensure_tessdata("eng")
    # Opt-out via env var must not raise even with a missing language.
    monkeypatch.setenv("FTS_OCR_AUTOINSTALL", "0")
    ocr._ensure_tessdata("zz_NOPE_DOES_NOT_EXIST")


def test_ensure_tessdata_passes_missing_languages_to_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.data import ocr

    calls: list[list[str]] = []
    monkeypatch.setattr(ocr, "_AUTO_INSTALL_DONE", False)
    monkeypatch.setattr(ocr, "installed_languages", list)
    monkeypatch.setattr(
        ocr,
        "install",
        lambda languages=None: calls.append(sorted(languages or [])) or {"errors": []},
    )
    ocr._ensure_tessdata("eng+pol")
    assert calls == [["eng", "pol"]]


def test_ocr_image_round_trips_a_real_png(tmp_path: Path) -> None:
    """The image parser must return extracted text via OCR end-to-end."""
    from finetune_studio.data.parsers.image import parse as parse_image

    if not _has_usable_ocr():
        pytest.skip("tesseract + eng/pol tessdata not installed in this env")

    marker = "HeliosMarkUniqueToken42"
    img_path = tmp_path / "ocr_smoke.png"
    img = Image.new("RGB", (900, 180), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 60), marker, fill="black", font=font)
    img.save(img_path)

    result = parse_image(img_path)
    text = (result.get("text") or "").lower()
    assert "heliosmarkuniquetoken42" in text, (
        "image parser did not extract text via OCR; warnings="
        + repr(result.get("warnings"))
    )


def _has_usable_ocr() -> bool:
    from finetune_studio.data import ocr

    return ocr.is_available("eng+pol")


def test_missing_binary_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When tesseract is not on PATH, the error names the install command."""
    from finetune_studio.data import ocr

    monkeypatch.setattr(ocr, "_tesseract_cmd", lambda: None)
    with pytest.raises(RuntimeError) as ei:
        ocr.ocr_image("/nonexistent.png")
    msg = str(ei.value)
    assert "tesseract" in msg.lower()
    # Must contain the install hint, not just say "not found".
    assert any(verb in msg for verb in ("brew install", "sudo apt", "sudo pacman", "sudo dfn")), (
        "missing-binary error should include a platform-correct install command; got: " + msg
    )
