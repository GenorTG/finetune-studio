"""Focused tests: strict QA validation and the on-disk pair record (mining jobs: tests/test_mining.py)."""
from __future__ import annotations

from finetune_studio.data.prep.parsers import parse_qa_json
from finetune_studio.data.prep.qa_validate import (
    Provenance,
    build_qa_record,
    normalize_question,
    validate_qa_batch,
    validate_qa_pair,
)

CHUNK = (
    "Helios Industries was founded in 1987 in Zurich. "
    "The chief executive officer is Mira Chen. "
    "The flagship product is the Aurora battery pack with 400 Wh/kg density."
)


class TestValidateAccepted:
    def test_grounded_pair_accepted(self) -> None:
        result = validate_qa_pair(
            "Who is the CEO of Helios Industries?",
            "The chief executive officer is Mira Chen.",
            CHUNK,
        )
        assert result.accepted is True
        assert result.reasons == ()

    def test_batch_accepts_valid_and_tracks_seen(self) -> None:
        seen: set[str] = set()
        batch = validate_qa_batch(
            [
                {
                    "q": "When was Helios Industries founded?",
                    "a": "Helios Industries was founded in 1987 in Zurich.",
                },
            ],
            CHUNK,
            seen_questions=seen,
        )
        assert len(batch.accepted) == 1
        assert batch.counters.accepted == 1
        assert normalize_question("When was Helios Industries founded?") in seen


class TestValidateRejected:
    def test_empty_question_and_answer(self) -> None:
        result = validate_qa_pair("", "", CHUNK)
        assert result.accepted is False
        assert "empty_question" in result.reasons
        assert "empty_answer" in result.reasons

    def test_malformed_short_question(self) -> None:
        result = validate_qa_pair("Why?", "Helios Industries was founded in 1987.", CHUNK)
        assert result.accepted is False
        assert "malformed_question" in result.reasons

    def test_duplicate_question(self) -> None:
        seen = {normalize_question("Who is the CEO of Helios Industries?")}
        result = validate_qa_pair(
            "Who is the CEO of Helios Industries?",
            "The chief executive officer is Mira Chen.",
            CHUNK,
            seen_questions=seen,
        )
        assert result.accepted is False
        assert "duplicate_question" in result.reasons

    def test_unanswerable_from_chunk(self) -> None:
        result = validate_qa_pair(
            "What is the orbital period of Jupiter's largest moon?",
            "Ganymede orbits Jupiter roughly once every seven days.",
            CHUNK,
        )
        assert result.accepted is False
        assert "unanswerable_from_chunk" in result.reasons

    def test_ungrounded_answer(self) -> None:
        result = validate_qa_pair(
            "Who is the CEO of Helios Industries?",
            "The president of France lives in the Elysee palace complex.",
            CHUNK,
        )
        assert result.accepted is False
        assert "ungrounded_answer" in result.reasons

    def test_malformed_short_answer(self) -> None:
        result = validate_qa_pair(
            "Who is the CEO of Helios Industries?",
            "short",
            CHUNK,
        )
        assert result.accepted is False
        assert "malformed_answer" in result.reasons

    def test_malformed_oversized_answer(self) -> None:
        result = validate_qa_pair(
            "Who is the CEO of Helios Industries?",
            "x" * 4001,
            CHUNK,
        )
        assert result.accepted is False
        assert "malformed_answer" in result.reasons

    def test_stopwords_only_question_is_malformed(self) -> None:
        # Content-token filter empties stopword-only questions.
        result = validate_qa_pair(
            "What is it?",
            "Helios Industries was founded in 1987 in Zurich.",
            CHUNK,
        )
        assert result.accepted is False
        assert "malformed_question" in result.reasons

    def test_question_at_min_length_boundary_accepted(self) -> None:
        # Exactly _MIN_Q_LEN (=8) chars; content token overlaps the chunk.
        assert len("Founded?") == 8
        result = validate_qa_pair(
            "Founded?",
            "Helios Industries was founded in 1987 in Zurich.",
            CHUNK,
        )
        assert result.accepted is True
        assert result.reasons == ()

    def test_refusal_or_meta_answer(self) -> None:
        result = validate_qa_pair(
            "Who is the CEO of Helios Industries?",
            "As an AI language model I cannot answer that from the passage.",
            CHUNK,
        )
        assert result.accepted is False
        assert "refusal_or_meta" in result.reasons


class TestParsingFallbacksPreserved:
    def test_json_fence_still_parses_then_validate(self) -> None:
        raw = '```json\n[{"q": "Who is the CEO of Helios Industries?", "a": "The chief executive officer is Mira Chen."}]\n```'
        pairs = parse_qa_json(raw, 1)
        assert len(pairs) == 1
        result = validate_qa_pair(pairs[0]["q"], pairs[0]["a"], CHUNK)
        assert result.accepted is True

    def test_line_fallback_still_parses(self) -> None:
        # Line fallback is reached when JSON extraction fails. A second
        # numbered pair supplies the newline lookahead the regex needs
        # (strip() removes a lone trailing newline).
        raw = (
            "1. Q: Who is the CEO of Helios Industries?\n"
            "A: The chief executive officer is Mira Chen.\n"
            "2. Q: When was Helios founded?\n"
            "A: Helios Industries was founded in 1987.\n"
        )
        pairs = parse_qa_json(raw, 2)
        assert len(pairs) >= 1
        assert "Mira Chen" in pairs[0]["a"]


class TestPairRecord:
    def test_build_qa_record_includes_provenance(self) -> None:
        pair = validate_qa_pair(
            "Who is the CEO of Helios Industries?",
            "The chief executive officer is Mira Chen.",
            CHUNK,
        )
        assert pair.accepted
        prov = Provenance(
            source_id="abc123",
            sha256="deadbeef",
            filename="helios.txt",
            chunk_idx=2,
        )
        rec = build_qa_record(
            qa_id="qa01",
            pair=pair,
            provenance=prov,
            chunk_text=CHUNK[:100],
            score=0.8,
            created_at=1.0,
            origin="mine",
            extra={"job_id": "j1", "round_id": "r1"},
        )
        assert rec["sha256"] == "deadbeef"
        assert rec["provenance"]["filename"] == "helios.txt"
        assert rec["provenance"]["validation"] == "strict_v2"
        assert rec["validation"]["accepted"] is True
        assert rec["origin"] == "mine" and rec["job_id"] == "j1" and rec["round_id"] == "r1"
        assert "difficulty" not in rec and "style" not in rec
