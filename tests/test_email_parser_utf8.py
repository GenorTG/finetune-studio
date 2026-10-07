"""8-bit UTF-8 e-mails keep their non-ASCII text (was mojibake / literal \\uXXXX escapes through a str round-trip)."""
from __future__ import annotations

from pathlib import Path

from finetune_studio.data.parsers import email as email_parser


def test_8bit_utf8_body_and_headers(tmp_path: Path) -> None:
    raw = (
        "From: Sofia Brandão <s@x.example>\nTo: A <a@x.example>\nSubject: Gdańsk hold\n"
        "MIME-Version: 1.0\nContent-Type: text/plain; charset=UTF-8\nContent-Transfer-Encoding: 8bit\n\n"
        "Counsel is Aurelia Hartwig at Hartwig Mielczarek i Wspólnicy in Gdańsk. Łużycka 14.\n"
    )
    p = tmp_path / "m.eml"
    p.write_bytes(raw.encode("utf-8"))
    text = email_parser.parse(p)["text"]
    assert "Wspólnicy in Gdańsk" in text and "Łużycka 14" in text
    assert "Sofia Brandão" in text and "\\u" not in text and "�" not in text
