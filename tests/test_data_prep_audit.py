"""Regression tests for the A2 data-prep audit (2026-10-01).

Pins four fixes found while tracing the QA-pair lifecycle end to end
(ingest -> chunk -> generate -> coverage_fill -> export -> audit):

1. coverage_fill.fill_coverage_gaps now stamps the caller's sha256/filename
   onto extractive-pair provenance instead of hardcoding "" for both.
2. coverage_fill.fill_coverage_gaps now surfaces a no-content chunk in
   ``chunks_still_uncovered`` instead of silently dropping it — the module
   docstring promises uncoverable chunks are "surfaced as uncovered, never
   silently dropped", but the no-content path used to violate that.
3. ingest.load_existing_chunks indexes chunk files by their own NNNN
   filename instead of glob() order, so a missing chunk file leaves a gap
   instead of silently shifting every later chunk's index down by one.
4. audit.audit_qa_pairs compares the export line count against the same
   deduplicate_qa_pairs() the exporter actually runs, instead of raw
   len(pairs) — so a legitimate (source, chunk, question) collapse is not
   misreported as export data loss.
"""
from __future__ import annotations

import json

import pytest

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.audit import audit_qa_pairs
from finetune_studio.data.prep.coverage_fill import fill_coverage_gaps
from finetune_studio.data.prep.ingest import load_existing_chunks


@pytest.fixture()
def proj(tmp_path, monkeypatch):
    """Point the fs root into tmp; project_dir creates itself on demand."""
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    import finetune_studio.data.fs.paths as paths

    paths._ROOT = tmp_path
    paths._PROJECTS = tmp_path / "projects"
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


def test_fill_coverage_gaps_stamps_sha256_and_filename_provenance(proj: str) -> None:
    source_id = "provsrc0001"
    result = fill_coverage_gaps(
        proj, source_id, "deadbeefsha", chunk_texts={1: CHARTER}, filename="charter.txt",
    )
    assert result.pairs_created > 0
    rows = pfs.list_qa_pairs(proj, source_id=source_id)
    assert rows
    for r in rows:
        prov = r.get("provenance", {})
        assert prov.get("sha256") == "deadbeefsha"
        assert prov.get("filename") == "charter.txt"
        # top-level mirror (build_qa_record copies provenance.sha256 up)
        assert r.get("sha256") == "deadbeefsha"


def test_fill_coverage_gaps_surfaces_no_content_chunk_as_uncovered(proj: str) -> None:
    source_id = "blanksrc0001"
    result = fill_coverage_gaps(proj, source_id, chunk_texts={1: "   "})
    assert result.pairs_created == 0
    assert result.skipped_no_content == 1
    # Must be surfaced, not merely counted — this is what the export gate
    # (DataPrepRunner._run_inner) reads to decide whether to block export.
    assert result.chunks_still_uncovered == [
        {"chunk_idx": 1, "chars": 0, "reason": "no_content"}
    ]


def test_load_existing_chunks_preserves_index_across_gap(proj: str) -> None:
    sha = "gapsha1234567"
    fd = pfs.file_dir(proj, sha)
    chunks_dir = fd / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "0000.txt").write_text("first chunk", encoding="utf-8")
    # 0001.txt intentionally missing (simulates a partial/failed write)
    (chunks_dir / "0002.txt").write_text("third chunk", encoding="utf-8")

    chunks = load_existing_chunks(proj, sha)

    assert len(chunks) == 3
    assert chunks[0] == "first chunk"
    assert chunks[1] == ""  # the gap, not "third chunk" shifted down
    assert chunks[2] == "third chunk"


def test_audit_qa_pairs_does_not_flag_legitimate_dedup_as_loss(proj: str) -> None:
    source = {"id": "s1", "sha256": "a" * 64, "filename": "one.txt", "chunk_count": 1}
    pfs.write_qa_source(proj, source)
    # Two approved pairs for the SAME (source, chunk, question) — a
    # legitimate duplicate the exporter's deduplicate_qa_pairs() collapses
    # into one training row.
    pfs.write_qa_pair(proj, {
        "id": "q1", "source_id": "s1", "chunk_idx": 1,
        "question": "Who owns the policy?", "answer": "Mira owns the policy.",
        "chunk_text": "Mira owns the policy.", "status": "approved",
    })
    pfs.write_qa_pair(proj, {
        "id": "q2", "source_id": "s1", "chunk_idx": 1,
        "question": "who owns the policy?", "answer": "Mira owns the policy, per the charter.",
        "chunk_text": "Mira owns the policy.", "status": "approved",
    })
    export_path = pfs.project_dir(proj) / "train.jsonl"
    row = {"conversations": [
        {"from": "human", "value": "Who owns the policy?"},
        {"from": "gpt", "value": "Mira owns the policy, per the charter."},
    ], "source_id": "s1", "chunk_idx": 1}
    export_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = audit_qa_pairs(proj, exported_path=str(export_path))

    assert report["approved_pair_count"] == 2
    assert report["expected_export_count"] == 1
    assert report["duplicate_pairs_collapsed"] == 1
    assert report["export_count"] == 1
    assert "approved_pair_count_differs_from_export_count" not in [
        e.get("error") for e in report["errors"]
    ]
    assert report["passed"] is True
