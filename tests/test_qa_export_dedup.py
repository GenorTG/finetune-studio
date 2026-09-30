"""Regression: QA export dedup must never collapse different sources/chunks.

`deduplicate_qa_pairs` used to key solely on normalized question text. Two
unrelated sources whose auto-generated questions happen to collide (e.g.
coverage_fill's extractive templates reuse a chunk's leading text as the
question "subject", so two CSVs sharing an identical header row produce a
byte-identical question despite holding completely different row data) would
silently drop the losing source's entire answer from the exported dataset —
even though the per-source coverage gate had already certified every chunk
as covered. Found via a live 102-file audit: 5 CSV sources with the same
header collapsed to 1 exported row, and a PDF/DOCX pair of the same content
collapsed to 1. The fix scopes the dedup key by (source_id, chunk_idx,
question) so only a genuine same-chunk repeat question collapses.
"""

from __future__ import annotations

from finetune_studio.data.prep.export import deduplicate_qa_pairs


def _pair(source_id: str, chunk_idx: int, question: str, answer: str) -> dict:
    return {
        "source_id": source_id,
        "chunk_idx": chunk_idx,
        "question": question,
        "answer": answer,
        "category": "source-grounded",
    }


def test_same_question_different_sources_both_survive():
    """Two CSVs with an identical header-derived question must both export."""
    items = [
        _pair("csv-a", 1, "According to the source document, unit_id | class ---?",
              "UNIT-4200 | Halcyon | active | 32.2 | 2180-01-10"),
        _pair("csv-b", 1, "According to the source document, unit_id | class ---?",
              "UNIT-4300 | Corvid-9 | reserve | 95.8 | 2180-01-10"),
    ]
    result = deduplicate_qa_pairs(items)
    source_ids = {r["source_id"] for r in result}
    assert source_ids == {"csv-a", "csv-b"}, (
        "both sources' answers must survive dedup even with identical question text"
    )
    assert len(result) == 2


def test_same_question_different_chunks_same_source_both_survive():
    """Two chunks of the SAME source with a colliding question must both export."""
    items = [
        _pair("doc-a", 1, "What is this about?", "First chunk fact."),
        _pair("doc-a", 2, "What is this about?", "Second chunk fact."),
    ]
    result = deduplicate_qa_pairs(items)
    assert len(result) == 2
    chunk_idxs = {r["chunk_idx"] for r in result}
    assert chunk_idxs == {1, 2}


def test_true_duplicate_within_same_chunk_collapses_to_one():
    """A chunk re-asked the identical question twice still collapses to one target."""
    items = [
        _pair("doc-a", 1, "What is X?", "Answer one, slightly longer restated."),
        _pair("doc-a", 1, "what is x?", "Answer one."),
    ]
    result = deduplicate_qa_pairs(items)
    assert len(result) == 1
    # Shorter answer wins the tie-break within the same (source, chunk, question) —
    # a concise target beats a rambling one at equal category priority.
    assert result[0]["answer"] == "Answer one."


def test_curated_category_outranks_source_grounded_within_same_key():
    items = [
        _pair("doc-a", 1, "What is X?", "Grounded answer, quite long and detailed."),
        {**_pair("doc-a", 1, "What is X?", "Curated short."), "category": "source-grounded-curated"},
    ]
    result = deduplicate_qa_pairs(items)
    assert len(result) == 1
    assert result[0]["answer"] == "Curated short."
