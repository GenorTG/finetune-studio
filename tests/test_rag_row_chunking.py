"""Row-preserving chunking: a table row is never split or glued to its neighbour (data.rag_portable.chunking).

Reproduces the Korvane RAG miss (tests/corpus/korvane/RESULTS.md §8, e097): the flattened CSV put one row's note
("Visit 3 Oct 10:00") right next to the *next* row's id and company, because the old chunker re-joined words with spaces.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.data.parsers import parse
from finetune_studio.data.rag_portable import PortableRAG
from finetune_studio.data.rag_portable import store as store_mod
from finetune_studio.data.rag_portable.chunking import (
    FALLBACK_CHARS_PER_TOKEN,
    chunk_rows,
    estimate_tokens,
    is_table_row,
)
from finetune_studio.data.rag_portable.schema import EmbeddingModelInfo
from finetune_studio.rag.ingest import chunk_text

CSV = (
    "opp_id,company,owner,note\n"
    "OPP-24-0306,Nordhav Logistikk AS,Ingrid Solberg,Quote sent; waiting for the harbour authority\n"
    "OPP-24-0306-B,Nordhav Cold Chain,Ingrid Solberg,Second site added\n"
    "OPP-24-0307,Vestfold Fisk AS,Rafał Dybek,Contact Rafał Dybek on site. Visit 3 Oct 10:00\n"
    "OPP-24-0308,Samsø Andelsselskab,Mette Holm,Pilot approved for 3 reefers\n"
    "OPP-24-0309,Fruttoria Srl,Luca Bianchi,Forklift model Rocla TX-45 fitted with tag reader\n"
)


def _old_chunk(text: str, size: int = 400) -> list[str]:
    """The removed word-window chunker: split on whitespace, re-join with single spaces."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def _parsed_csv(tmp_path: Path) -> str:
    path = tmp_path / "pipeline.csv"
    path.write_text(CSV, encoding="utf-8")
    return parse(path)["text"]


def _row_lines(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if ln.strip()]


def test_old_chunker_fused_the_note_with_the_next_row(tmp_path: Path) -> None:
    """Guards the fixture: the layout the new chunker must not produce is the one the old one did."""
    text = _parsed_csv(tmp_path)
    (old,) = _old_chunk(text)
    assert "Visit 3 Oct 10:00 OPP-24-0308 | Samsø Andelsselskab" in old
    assert "\n" not in old


def test_every_csv_row_stays_on_its_own_line(tmp_path: Path) -> None:
    text = _parsed_csv(tmp_path)
    source_rows = set(_row_lines(text))
    chunks = chunk_rows(text, size=400, overlap=80)
    assert len(chunks) == 1
    assert "\n" in chunks[0]
    assert set(_row_lines(chunks[0])) == source_rows
    assert chunks[0].split("\n") == text.split("\n")


def test_note_never_shares_a_line_with_the_next_rows_company(tmp_path: Path) -> None:
    text = _parsed_csv(tmp_path)
    for size in (400, 60, 25, 8, 1):
        for chunk in chunk_rows(text, size=size, overlap=10):
            for line in chunk.split("\n"):
                assert not ("Visit 3 Oct 10:00" in line and "Samsø" in line), (size, line)
                assert not ("Visit 3 Oct 10:00" in line and "OPP-24-0308" in line), (size, line)


@pytest.mark.parametrize("size", [400, 40, 20, 10, 1])
def test_rows_are_never_split_whatever_the_budget(tmp_path: Path, size: int) -> None:
    """Even a budget smaller than one row yields whole rows: an oversized row is a chunk of its own."""
    text = _parsed_csv(tmp_path)
    source_rows = set(_row_lines(text))
    seen: set[str] = set()
    for chunk in chunk_rows(text, size=size, overlap=5):
        for line in _row_lines(chunk):
            assert line in source_rows, f"line is not a whole source row: {line!r}"
            seen.add(line)
    assert seen == source_rows


def test_oversized_row_is_a_chunk_of_its_own_not_cut() -> None:
    long_row = "ID-1 | " + " ".join(f"cell{i}" for i in range(200)) + " | end"
    text = f"before it\n{long_row}\nafter it"
    chunks = chunk_rows(text, size=20, overlap=0)
    assert long_row in chunks
    assert all(long_row not in c or c == long_row for c in chunks)


def test_header_is_repeated_on_continuation_chunks(tmp_path: Path) -> None:
    # The CSV parser writes the ``---`` separator only when the sniffer sees a header (this all-text file has none),
    # so build the header-aware shape explicitly: header, separator, rows.
    head, *rows = _parsed_csv(tmp_path).split("\n")
    header, sep = head, " | ".join(["---"] * 4)
    text = "\n".join([header, sep, *rows])
    chunks = chunk_rows(text, size=45, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        lines = chunk.split("\n")
        assert lines[0] == header and lines[1] == sep
    bodies = [ln for c in chunks for ln in c.split("\n")[2:]]
    assert bodies == text.split("\n")[2:]            # each data row exactly once, in order, no overlap rows


def test_table_without_header_has_no_invented_header() -> None:
    text = "\n".join(f"R{i} | alpha {i} | beta {i}" for i in range(30))   # xlsx / docx tables: no --- separator
    chunks = chunk_rows(text, size=40, overlap=0)
    assert len(chunks) > 1
    assert [ln for c in chunks for ln in c.split("\n")] == text.split("\n")


def test_row_in_a_later_chunk_keeps_the_forklift_model_with_its_columns() -> None:
    rows = [f"EQ-{i:03d} | Hall {i % 4} | Rocla TX-{40 + i} | serviced" for i in range(40)]
    text = "\n".join(["asset | location | model | state", "--- | --- | --- | ---", *rows])
    chunks = chunk_rows(text, size=80, overlap=0)
    holder = next(c for c in chunks if "EQ-035" in c)
    assert holder.startswith("asset | location | model | state")
    assert "EQ-035 | Hall 3 | Rocla TX-75 | serviced" in holder.split("\n")


def test_prose_lines_are_packed_whole_with_whole_line_overlap() -> None:
    lines = [f"Sentence number {i} says something short." for i in range(30)]
    chunks = chunk_rows("\n".join(lines), size=40, overlap=15)
    assert len(chunks) > 2
    for a, b in pairwise(chunks):
        a_lines, b_lines = a.split("\n"), b.split("\n")
        assert b_lines[0] in a_lines, "overlap = trailing whole lines of the previous chunk"
    assert {ln for c in chunks for ln in c.split("\n")} == set(lines)
    assert all(ln in lines for c in chunks for ln in c.split("\n"))


def test_overlong_prose_line_is_cut_on_word_boundaries_with_overlap() -> None:
    words = [f"w{i}" for i in range(300)]
    chunks = chunk_rows(" ".join(words), size=50, overlap=10, count_tokens=lambda t: len(t.split()))
    assert len(chunks) > 5
    seen: list[str] = []
    for chunk in chunks:
        pieces = chunk.split()
        assert 0 < len(pieces) <= 50
        seen.extend(pieces)
    assert set(seen) == set(words)
    first, second = chunks[0].split(), chunks[1].split()
    assert second[0] in first and len(set(first) & set(second)) <= 10


def test_prose_does_not_swallow_the_following_table() -> None:
    text = "Intro paragraph about pipeline.\n\nid | name\n--- | ---\nA1 | Alpha\nA2 | Beta\n\nClosing remark."
    (chunk,) = chunk_rows(text, size=400, overlap=40)
    assert chunk.split("\n") == text.split("\n")


def test_budget_is_measured_with_the_supplied_tokenizer() -> None:
    """Same text, two counters: a coarse tokenizer (1 token per char) packs fewer lines per chunk than a one-token-per-line one."""
    lines = [f"line {i:02d}" for i in range(20)]
    text = "\n".join(lines)
    per_char = chunk_rows(text, size=24, overlap=0, count_tokens=len)
    per_line = chunk_rows(text, size=4, overlap=0, count_tokens=lambda t: 1)
    assert all(sum(len(ln) for ln in c.split("\n")) <= 24 for c in per_char)
    assert all(len(c.split("\n")) <= 4 for c in per_line)
    assert len(per_char) > len(per_line) > 1


def test_fallback_estimate_is_documented_chars_per_token() -> None:
    assert FALLBACK_CHARS_PER_TOKEN == 3.5
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 35) == 10
    lines = [f"{'x' * 34}{i}" for i in range(10)]          # 35 chars = 10 estimated tokens per line
    chunks = chunk_rows("\n".join(lines), size=30, overlap=0)
    assert [len(c.split("\n")) for c in chunks] == [3, 3, 3, 1]


def test_degenerate_inputs() -> None:
    assert chunk_rows("", 400, 80) == []
    assert chunk_rows("   \n\n  ", 400, 80) == []
    assert chunk_rows("one line", 400, 80) == ["one line"]
    assert chunk_rows("a\r\nb\r\n", 400, 80) == ["a\nb"]
    assert chunk_rows("a\n\n\n\nb", 400, 80) == ["a\n\nb"]
    assert chunk_rows("x y z", 3, 3) == ["x y z"]            # overlap >= size is clamped, no loop


def test_is_table_row() -> None:
    assert is_table_row("a | b | c")
    assert is_table_row("| a | b |")
    assert is_table_row("--- | ---")
    assert not is_table_row("--- ")
    assert not is_table_row("plain prose, no cells")
    assert not is_table_row("either/or|not a table")          # no spaces around the pipe


def test_chunk_text_wrapper_keeps_ids_and_metadata(tmp_path: Path) -> None:
    text = _parsed_csv(tmp_path)
    chunks = chunk_text(text, chunk_size=45, overlap=0, metadata={"filename": "pipeline.csv"}, doc_id="d1")
    assert [c.id for c in chunks] == [f"d1_{i}" for i in range(len(chunks))]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.document_id == "d1" and c.metadata == {"filename": "pipeline.csv"} for c in chunks)
    assert chunk_text("   ") == []
    custom = chunk_text(text, chunk_size=3, overlap=0, count_tokens=lambda t: 1)
    assert all(len(c.text.split("\n")) <= 3 for c in custom)


def test_embedder_exposes_its_tokenizer_for_chunk_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.data.rag_portable import embedders

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=True, verbose=True):
            assert add_special_tokens is False and verbose is False
            return {"input_ids": list(range(len(text.split())))}

    class FakeModel:
        tokenizer = FakeTokenizer()
        max_seq_length = 512

        def get_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts, **_kw):
            return np.zeros((len(texts), 4), dtype="float32")

    monkeypatch.setattr(embedders, "_load_sentence_transformer", lambda *_a, **_k: FakeModel())
    encode, _info = embedders.get_embedder(name="some/model", device="cpu")
    assert encode.count_tokens("one two three") == 3
    assert encode.max_seq_tokens == 512


def test_build_writes_row_chunks_sized_with_the_embedder_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    calls: list[str] = []

    def fake_get_embedder(name: str, device: str = "auto"):
        def encode(texts):
            return np.ones((len(texts), 4), dtype="float32")

        def count_tokens(text: str) -> int:
            calls.append(text)
            return len(text.split())

        encode.count_tokens = count_tokens
        encode.max_seq_tokens = 512

        return encode, EmbeddingModelInfo(name="fake", dim=4)

    monkeypatch.setattr(store_mod, "get_embedder", fake_get_embedder)
    src = tmp_path / "src"
    src.mkdir()
    (src / "pipeline.csv").write_text(CSV, encoding="utf-8")
    corpus = tmp_path / "corpus"
    rag = PortableRAG(corpus)
    result = rag.build_from_directory(src, name="t", embedder="fake", chunk_size=30, overlap=0, device="cpu")
    assert result["chunks"] >= 2
    assert calls, "chunk sizes must come from the embedder tokenizer"

    import pandas as pd

    df = pd.read_parquet(corpus / "chunks.parquet")
    source_rows = set(_row_lines(parse(src / "pipeline.csv")["text"]))
    for text in df["text"]:
        assert "\n" in text
        for line in _row_lines(text):
            assert line in source_rows
    assert PortableRAG(corpus).load().manifest.chunk_settings.splitter == "rows+tokens"
