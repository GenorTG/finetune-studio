"""Tests for data/prep/coverage_fill.py — the no-chunk-left-behind filler."""
from __future__ import annotations

import pytest

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.prep.coverage_fill import (
    _make_pairs_from_chunk,
    fill_coverage_gaps,
    split_sentences,
)


@pytest.fixture()
def proj(tmp_path, monkeypatch):
    """Point the fs root into tmp; project_dir creates itself on demand."""
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    import importlib

    import finetune_studio.data.fs.paths as paths

    paths._ROOT = tmp_path
    paths._PROJECTS = tmp_path / "projects"
    # qa/fs helpers read project_dir() dynamically each call, so a module
    # attr patch is enough — no reload needed. Sanity check:
    d = pfs.project_dir("testproj")
    assert str(d).startswith(str(tmp_path))
    return "testproj"


CHARTER = (
    "Charter of the Ledger-Keepers of the Vaelindrath Concord. "
    "A Ledger-Keeper is sworn at the age of 231 and serves 158 years. "
    "Their oath-stone is carved from margin-stone. "
    "The Concord pays a Ledger-Keeper 1875 crowns per annum plus 71 measures of black-whiskey. "
    "Dismissal requires the countersignature of a Salt-Speaker and two Ledger-Keepers."
)


def test_split_sentences_basic() -> None:
    sents = split_sentences(CHARTER)
    assert len(sents) >= 4
    assert any("1875 crowns" in s for s in sents)


def test_make_pairs_deterministic_and_grounded() -> None:
    a = _make_pairs_from_chunk(CHARTER, 1, seen_questions=set())
    b = _make_pairs_from_chunk(CHARTER, 1, seen_questions=set())
    assert a == b  # deterministic
    assert a
    for q, ans in a:
        # answer text must appear *in* the chunk (extractive, not invented)
        assert ans.rstrip(".").lower() in CHARTER.lower()
        assert len(q) > 10


def test_fill_creates_pairs_for_uncovered_chunk(proj: str) -> None:
    source_id = "deadbeef1234"
    result = fill_coverage_gaps(proj, source_id, chunk_texts={1: CHARTER})
    assert result.pairs_created > 0
    assert result.chunks_filled == 1
    assert result.chunks_still_uncovered == []
    rows = pfs.list_qa_pairs(proj, source_id=source_id)
    assert rows, "no pairs written"
    assert all(r["status"] == "approved" for r in rows)
    assert all(r.get("origin") == "coverage_fill" for r in rows)
    for r in rows:
        assert r["answer"].rstrip(".").lower() in CHARTER.lower()


def test_fill_is_idempotent(proj: str) -> None:
    source_id = "deadbeef1234"
    first = fill_coverage_gaps(proj, source_id, chunk_texts={1: CHARTER})
    assert first.pairs_created > 0
    second = fill_coverage_gaps(proj, source_id, chunk_texts={1: CHARTER})
    assert second.pairs_created == 0  # chunk now covered
    rows = pfs.list_qa_pairs(proj, source_id=source_id)
    assert len(rows) == first.pairs_created  # no duplicates


def test_fill_reports_uncoverable_chunk(proj: str) -> None:
    noise = "~~~ *** ###"
    result = fill_coverage_gaps(proj, "secretsrc", chunk_texts={1: noise})
    assert result.pairs_created == 0
    assert result.chunks_filled == 0
    # surfaced, not silently dropped
    assert result.chunks_still_uncovered == [{"chunk_idx": 1, "chars": len(noise)}]


def test_fill_only_touches_uncovered_chunks(proj: str) -> None:
    source_id = "coveredsrc"
    pfs.write_qa_pair(proj, {
        "id": "seed0001", "source_id": source_id, "chunk_idx": 1,
        "question": "Existing?", "answer": "Existing answer.", "status": "approved",
    })
    result = fill_coverage_gaps(proj, source_id, chunk_texts={1: CHARTER, 2: CHARTER})
    assert result.pairs_created > 0
    rows = pfs.list_qa_pairs(proj, source_id=source_id)
    chunks_touched = {r["chunk_idx"] for r in rows}
    assert chunks_touched == {1, 2}
    assert [r for r in rows if r["id"] == "seed0001"]  # seed untouched
    assert sum(1 for r in rows if r["chunk_idx"] == 1) == 1  # no extra on covered


def test_fill_all_project_gaps_aggregates_and_registers(proj: str) -> None:
    from finetune_studio.data.prep.coverage_fill import fill_all_project_gaps

    sid = "aggsource01"
    pfs.write_qa_source(proj, {
        "id": sid, "sha256": "aggsha123456", "filename": "charter.txt",
        "mime_type": "text/plain", "char_count": len(CHARTER),
        "chunk_count": 1, "parser": "text_v1", "status": "ready",
        "data_path": "", "path": "",
    })
    # store chunks under the content-addressed file dir the loader reads
    d = pfs.file_dir(proj, "aggsha123456")
    (d / "chunks").mkdir()
    (d / "chunks" / "0001.txt").write_text(CHARTER, encoding="utf-8")

    summary = fill_all_project_gaps(proj)
    assert summary["pairs_created"] > 0
    assert summary["chunks_filled"] >= 1
    rows = pfs.list_qa_pairs(proj, source_id=sid)
    assert rows and all(r.get("origin") == "coverage_fill" for r in rows)


def test_fill_all_handles_missing_parsed_artifacts(proj: str) -> None:
    """A source whose declared chunks can't be loaded surfaces, never crashes."""
    from finetune_studio.data.prep.coverage_fill import fill_all_project_gaps

    # Declares 2 chunks but has no stored chunk files (artifact loss).
    pfs.write_qa_source(proj, {
        "id": "ghostsrc0001", "sha256": "noton磁盘sh", "filename": "lost.txt",
        "mime_type": "text/plain", "char_count": 900, "chunk_count": 2,
        "parser": "text_v1", "status": "ready", "data_path": "", "path": "",
    })
    summary = fill_all_project_gaps(proj)
    ghost = [u for u in summary["uncovered_chunks"] if u.get("source") == "ghostsrc0001"]
    assert ghost and any("missing" in str(u.get("error")) for u in ghost)
