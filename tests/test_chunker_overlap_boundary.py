"""Chunk overlap must start on a word boundary.

Found in the live user walkthrough: the second chunk of a support handbook began ``r support\\nCustomer
support replies ...`` (the tail of "Customer support" cut mid-word), and the coverage-fill question built
from it read "what is stated regarding r support Customer support?".
"""
from __future__ import annotations

from finetune_studio.data.prep.chunker import _tail, chunk_text


def test_tail_never_starts_inside_a_word() -> None:
    text = "Warranty and Customer support\nreplies within two days."
    for overlap in range(5, len(text)):
        t = _tail(text, overlap)
        assert t == "" or text.endswith(t)
        assert t == "" or text[len(text) - len(t) - 1].isspace() or len(t) == len(text.strip())


def test_tail_short_text_is_returned_whole() -> None:
    assert _tail("short", 50) == "short"


def test_second_chunk_does_not_open_with_a_word_fragment() -> None:
    para1 = ("Warranty. " + "The warranty covers manufacturing defects in the heating element and the lid hinge. " * 6
             + "Customer support")
    para2 = "replies to every email within two business days. Urgent safety issues are escalated the same day."
    chunks = chunk_text(para1 + "\n\n" + para2, target_chars=300, overlap=40)
    assert len(chunks) >= 2
    for later in chunks[1:]:
        first_word = later.split()[0]
        assert first_word in para1 + para2 and (later[0].isupper() or first_word in ("support", "replies", "Customer")
                                                  or first_word[0].isalpha()), later[:30]
        assert not later.startswith("r support")
