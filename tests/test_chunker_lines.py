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


def test_markdown_sections_start_their_own_passage_without_overlap() -> None:
    """Retrieval returns passages: a section must not end in, or open with, its neighbour's text (2026-10-21)."""
    body = "word " * 50
    text = (f"# Identity\n\n## Who I am\nMy name is Krzysztof. {body}\n\n"
            f"## My company\nI run Smart Samurai. {body}\n\n## Contact\nEmail me. {body}")
    chunks = chunk_text(text, target_chars=1200, overlap=200)
    assert [c.splitlines()[0] for c in chunks] == ["# Identity", "## My company", "## Contact"]
    assert "Smart Samurai" not in chunks[0] and "Krzysztof" not in chunks[1]   # no overlap across sections
    # a short lone title stays with the first section; #### and deeper never split
    assert chunk_text("## A\nshort\n\n#### detail\nx", target_chars=1200) == ["## A\nshort\n\n#### detail\nx"]
