"""HTML parser must not emit the <title> and the identical <h1> as two lines (live walkthrough: a generated
answer read 'Meridian Sync 4.2 Meridian Sync 4.2 Version 4.2 adds offline mode.')."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from finetune_studio.data.parsers.html import parse


@pytest.fixture(params=["bs4", "regex-fallback"])
def parser_path(request, monkeypatch):
    """Both implementations: the live host ran the regex fallback for days without bs4 installed."""
    if request.param == "regex-fallback":
        monkeypatch.setitem(sys.modules, "bs4", None)       # import raises ImportError
    else:
        pytest.importorskip("bs4")
    return request.param


def _parse(tmp_path: Path, html: str) -> str:
    f = tmp_path / "p.html"
    f.write_text(html, encoding="utf-8")
    return parse(f)["text"]


def test_title_equal_to_h1_appears_once(tmp_path: Path, parser_path: str) -> None:
    text = _parse(tmp_path, "<html><head><title>Meridian Sync 4.2</title></head>"
                            "<body><h1>Meridian Sync 4.2</h1><p>Adds offline mode.</p></body></html>")
    assert text.count("Meridian Sync 4.2") == 1 and "Adds offline mode." in text


def test_title_is_kept_when_the_page_has_no_matching_heading(tmp_path: Path, parser_path: str) -> None:
    text = _parse(tmp_path, "<html><head><title>Release notes</title></head><body><p>Fixed a crash.</p></body></html>")
    assert text.replace("\n", " ").startswith("Release notes") and "Fixed a crash." in text
