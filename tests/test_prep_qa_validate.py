"""Focused tests: strict QA validation + DataPrepRunner acceptance/rejection."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from finetune_studio.data.prep.parsers import parse_qa_json
from finetune_studio.data.prep.qa_validate import (
    CoverageTracker,
    Provenance,
    build_qa_record,
    normalize_question,
    validate_qa_batch,
    validate_qa_pair,
)
from finetune_studio.data.prep.runner import DataPrepRunner

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


class TestProvenanceAndCoverage:
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
            difficulty="medium",
            style="factual",
            score=0.8,
            created_at=1.0,
        )
        assert rec["sha256"] == "deadbeef"
        assert rec["provenance"]["filename"] == "helios.txt"
        assert rec["provenance"]["validation"] == "strict_v1"
        assert rec["validation"]["accepted"] is True

    def test_coverage_tracker(self) -> None:
        cov = CoverageTracker(source_id="s1", chunks_total=3)
        cov.mark_accepted(1)
        cov.mark_accepted(1)
        cov.mark_accepted(3)
        info = cov.as_dict()
        assert info["chunks_with_accepted"] == 2
        assert info["chunks_uncovered"] == 1
        assert cov.uncovered_chunks([1, 2, 3]) == [2]


class TestRunnerValidationBehavior:
    def test_runner_writes_only_accepted_pairs(self) -> None:
        # Model returns one good + one refusal; only good should be written.
        raw = (
            '[{"q": "Who is the CEO of Helios Industries?",'
            ' "a": "The chief executive officer is Mira Chen."},'
            ' {"q": "What is the flagship product density?",'
            ' "a": "As an AI I cannot help with that."}]'
        )
        written: list[dict] = []

        def fake_store(*_a, **_k):
            meta = MagicMock()
            meta.sha256 = "a" * 64
            meta.byte_count = 10
            meta.uploaded_at = 1.0
            return ("path", meta)

        ingest = MagicMock()
        ingest.ok = True
        ingest.error = ""
        ingest.chunks = [CHUNK]
        ingest.reused = False
        ingest.parser = "txt"
        ingest.char_count = len(CHUNK)
        ingest.chunk_count = 1
        ingest.warnings = []

        with (
            patch("finetune_studio.data.prep.runner.pfs") as mock_pfs,
            patch("finetune_studio.data.prep.ingest.parse_and_chunk", return_value=ingest),
            patch(
                "finetune_studio.data.prep.generator.resolve_generator",
                return_value=lambda *_a, **_k: raw,
            ),
            patch(
                "finetune_studio.data.prep.generator.helper_resolution_error",
                return_value="",
            ),
        ):
            mock_pfs.store_file.side_effect = fake_store
            mock_pfs.write_qa_pair.side_effect = lambda _pid, qa: written.append(qa)
            mock_pfs.log_ingestion = MagicMock()
            mock_pfs.write_qa_source = MagicMock()

            runner = DataPrepRunner(
                "pid", b"hello", "helios.txt", qa_per_chunk=2, style="factual",
            )
            result = runner.run()

        assert result["ok"] is True
        assert result["qa"] == 1
        assert len(written) == 1
        assert written[0]["question"].startswith("Who is the CEO")
        assert "provenance" in written[0]
        assert result["rejection_counters"]["rejected"] >= 1
        assert result["coverage"]["chunks_with_accepted"] == 1

    def test_runner_rejects_duplicate_across_chunks(self) -> None:
        raw = (
            '[{"q": "Who is the CEO of Helios Industries?",'
            ' "a": "The chief executive officer is Mira Chen."}]'
        )
        written: list[dict] = []

        def fake_store(*_a, **_k):
            meta = MagicMock()
            meta.sha256 = "b" * 64
            meta.byte_count = 10
            meta.uploaded_at = 1.0
            return ("path", meta)

        ingest = MagicMock()
        ingest.ok = True
        ingest.error = ""
        ingest.chunks = [CHUNK, CHUNK]
        ingest.reused = False
        ingest.parser = "txt"
        ingest.char_count = len(CHUNK) * 2
        ingest.chunk_count = 2
        ingest.warnings = []

        with (
            patch("finetune_studio.data.prep.runner.pfs") as mock_pfs,
            patch("finetune_studio.data.prep.ingest.parse_and_chunk", return_value=ingest),
            patch(
                "finetune_studio.data.prep.generator.resolve_generator",
                return_value=lambda *_a, **_k: raw,
            ),
        ):
            mock_pfs.store_file.side_effect = fake_store
            mock_pfs.write_qa_pair.side_effect = lambda _pid, qa: written.append(qa)
            mock_pfs.log_ingestion = MagicMock()
            mock_pfs.write_qa_source = MagicMock()

            result = DataPrepRunner(
                "pid", b"hello", "helios.txt", qa_per_chunk=1,
            ).run()

        assert result["ok"] is True
        assert len(written) == 1
        assert result["rejection_counters"]["by_reason"].get("duplicate_question", 0) >= 1
