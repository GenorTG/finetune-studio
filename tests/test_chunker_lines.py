"""Rows and chat lines are never cut in half: a '.' inside a cell is not a sentence end."""
from __future__ import annotations

from finetune_studio.data.prep.chunker import chunk_text


def test_table_rows_stay_whole_across_chunks() -> None:
    rows = [f"R{i:02d} | Insurance claims line | quote policy no. SB-KCC-{i}-24 | note text. Another sentence {i}." for i in range(80)]
    chunks = chunk_text("\n".join(rows))
    assert len(chunks) > 1
    seen = [ln for c in chunks for ln in c.split("\n") if ln.startswith("R")]
    assert set(seen) == set(rows)                       # every row appears intact somewhere
    assert all(ln.startswith("R") for c in chunks for ln in c.split("\n") if ln.strip())


def test_a_line_longer_than_a_chunk_is_still_split_into_sentences() -> None:
    long_line = " ".join(f"Sentence number {i} states a fact." for i in range(120))
    chunks = chunk_text(long_line, target_chars=500)
    assert len(chunks) > 3 and all(len(c) <= 700 for c in chunks)
