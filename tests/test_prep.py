"""Tests for data/prep package — chunker, parsers, export."""
import pytest

# ── Chunker ─────────────────────────────────────────────────────────────────

class TestChunker:
    """Tests for the paragraph/sentence-aware text splitter."""

    def test_empty_string(self):
        from finetune_studio.data.prep.chunker import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_single_short_paragraph(self):
        from finetune_studio.data.prep.chunker import chunk_text
        text = "This is a short paragraph."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short paragraph."

    def test_multiple_short_paragraphs_one_chunk(self):
        from finetune_studio.data.prep.chunker import chunk_text
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text)
        assert len(chunks) == 1

    def test_long_paragraph_split_into_multiple_chunks(self):
        from finetune_studio.data.prep.chunker import chunk_text
        # Build SEPARATE short paragraphs that each have sentence boundaries —
        # the chunker will try to group them, but when they exceed target_chars
        # it will split. Use paragraphs small enough to pack 2-3 before hitting limit.
        sent = "This is sentence one. "   # 23 chars
        para = sent * 20  # ~460 chars per para — 2 fit in 500
        text = f"Intro.\n\n{para}\n\n{para}\n\nOutro."  # two ~460-char paragraphs
        chunks = chunk_text(text, target_chars=500, overlap=50)
        # Should produce multiple chunks (the two paragraphs can't both fit in one)
        assert len(chunks) >= 2, f"Expected 2+ chunks, got {len(chunks)}: {[len(c) for c in chunks]}"
        # All sentences should be preserved
        full_text = " ".join(chunks)
        # Overlap tail can carry a few extra words, so count >= original (exact preserved content)
        assert full_text.count("sentence one") >= 40  # content preserved, overlap adds fragments

    def test_overlap_carries_tail_context(self):
        from finetune_studio.data.prep.chunker import chunk_text
        para1 = "A" * 400
        para2 = "B" * 400
        para3 = "C" * 400
        text = f"{para1}\n\n{para2}\n\n{para3}"
        chunks = chunk_text(text, target_chars=500, overlap=100)
        # Second chunk should start with tail of first
        if len(chunks) >= 2:
            assert chunks[1].startswith(chunks[0][-100:])

    def test_whitespace_only_paras_ignored(self):
        from finetune_studio.data.prep.chunker import chunk_text
        text = "   \n\n   \n\nActual para.\n\n   \n\n"
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert "Actual para" in chunks[0]

    def test_sentence_boundary_respected(self):
        from finetune_studio.data.prep.chunker import chunk_text
        # Long paragraph without internal newlines
        sent1 = "This is the first sentence. "
        sent2 = "Here is the second sentence. "
        sent3 = "Finally, the third sentence."
        text = (sent1 + sent2 + sent3) * 20  # make it long
        chunks = chunk_text(text, target_chars=600, overlap=80)
        # Should split, not lose sentences
        full_text = " ".join(chunks)
        assert "first sentence" in full_text
        assert "second sentence" in full_text
        assert "third sentence" in full_text


# ── Parsers ──────────────────────────────────────────────────────────────────

class TestFindMatchingBracket:
    """Tests for bracket matching utility used in JSON extraction."""

    def test_simple_array(self):
        from finetune_studio.data.prep.parsers import find_matching_bracket
        s = "[1, 2, 3]"
        pos = find_matching_bracket(s, 0)
        assert pos == len(s) - 1

    def test_nested_array(self):
        from finetune_studio.data.prep.parsers import find_matching_bracket
        s = "[[1, 2], [3, 4]]"
        pos = find_matching_bracket(s, 0)
        assert s[pos] == "]"
        assert pos == len(s) - 1

    def test_string_with_brackets(self):
        from finetune_studio.data.prep.parsers import find_matching_bracket
        s = '["[nested]", "more]"]'
        pos = find_matching_bracket(s, 0)
        assert s[pos] == "]"

    def test_escaped_quote_in_string(self):
        from finetune_studio.data.prep.parsers import find_matching_bracket
        s = '["quote\\"]inside", 2]'
        pos = find_matching_bracket(s, 0)
        assert s[pos] == "]"

    def test_invalid_start(self):
        from finetune_studio.data.prep.parsers import find_matching_bracket
        assert find_matching_bracket("not a bracket", 0) is None
        assert find_matching_bracket("[1,2]", 1) is None  # starts on element

    def test_deeply_nested(self):
        from finetune_studio.data.prep.parsers import find_matching_bracket
        s = "[[[[1]]], [2]]"
        pos = find_matching_bracket(s, 0)
        assert s[pos] == "]"
        assert pos == len(s) - 1


# ── Prompt helpers ───────────────────────────────────────────────────────────

class TestPromptConstants:
    """Smoke-test that prompt constants are well-formed."""

    def test_qa_system_prompt_exists(self):
        from finetune_studio.data.prep.prompts import QA_SYSTEM_PROMPT
        assert len(QA_SYSTEM_PROMPT) > 50
        # Should contain key instruction
        assert "JSON" in QA_SYSTEM_PROMPT

    def test_qa_user_template(self):
        from finetune_studio.data.prep.prompts import QA_USER_TEMPLATE
        assert "{" in QA_USER_TEMPLATE  # contains format placeholder
        assert "chunk" in QA_USER_TEMPLATE
        assert "difficulty" in QA_USER_TEMPLATE


# ── Export ───────────────────────────────────────────────────────────────────

class TestExportQA:
    """Tests for JSONL export of curated Q&A pairs."""

    def test_export_formats_are_valid_jsonl(self):
        import json
        from finetune_studio.data.prep.export import export_qa_jsonl
        from unittest.mock import patch

        # Mock pfs.list_qa_pairs to return sample data
        mock_pairs = [
            {
                "source_id": "src1",
                "chunk_idx": 0,
                "score": 1.0,
                "question": "What is fine-tuning?",
                "answer": "Fine-tuning is adapting a pre-trained model.",
                "status": "approved",
            }
        ]
        with patch("finetune_studio.data.prep.export.pfs") as mock_pfs:
            mock_pfs.list_qa_pairs.return_value = mock_pairs

            for fmt in ("sharegpt", "alpaca", "openai"):
                result = export_qa_jsonl("pid", fmt=fmt, only="approved")
                lines = [l for l in result.strip().split("\n") if l]
                assert len(lines) == 1
                obj = json.loads(lines[0])
                # sharegpt has 'conversations'; alpaca has 'instruction'; openai has 'messages'
                assert ("conversations" in obj or "instruction" in obj or "messages" in obj), \
                    f"Unexpected format for {fmt}: {list(obj.keys())}"

    def test_export_unknown_format_raises(self):
        from finetune_studio.data.prep.export import export_qa_jsonl
        from unittest.mock import patch
        with patch("finetune_studio.data.prep.export.pfs") as mock_pfs:
            mock_pfs.list_qa_pairs.return_value = []
            with pytest.raises(ValueError, match="Unknown format"):
                export_qa_jsonl("pid", fmt="made_up_format")

    def test_export_tolerates_minimal_pair_schema(self):
        """Pairs written by the data-prep chat tool's create_qa_pairs
        only carry id/source_id/question/answer/status — no chunk_idx,
        chunk_text, difficulty, style, or score. The exporter must
        still produce valid JSONL (regression test for the 500 we hit
        on fan-dragon when 2 of 29 pairs lacked these fields)."""
        import json
        from finetune_studio.data.prep.export import export_qa_jsonl
        from unittest.mock import patch

        minimal_pairs = [
            {
                "id": "qa_1",
                "source_id": "abc",
                "question": "What is fine-tuning?",
                "answer": "Adapting a pretrained model.",
                "status": "approved",
                "created_at": 1.0,
                "created_via": "data-prep-chat",
            }
        ]
        with patch("finetune_studio.data.prep.export.pfs") as mock_pfs:
            mock_pfs.list_qa_pairs.return_value = minimal_pairs
            result = export_qa_jsonl("pid", fmt="sharegpt", only="approved")
            lines = [l for l in result.strip().split("\n") if l]
            assert len(lines) == 1
            obj = json.loads(lines[0])
            assert obj["conversations"][0]["from"] == "human"
            assert obj["conversations"][1]["from"] == "gpt"
            assert obj["source_id"] == "abc"
            # chunk_idx and score should default, not crash
            assert obj["chunk_idx"] == 0
            assert obj["score"] in (0, 0.0)

    def test_export_alpaca_tolerates_minimal_pair_schema(self):
        from finetune_studio.data.prep.export import export_qa_jsonl
        from unittest.mock import patch
        minimal_pairs = [{
            "id": "qa_x", "source_id": "s", "question": "q",
            "answer": "a", "status": "approved",
        }]
        with patch("finetune_studio.data.prep.export.pfs") as mock_pfs:
            mock_pfs.list_qa_pairs.return_value = minimal_pairs
            result = export_qa_jsonl("pid", fmt="alpaca", only="approved")
            assert "q" in result and "a" in result

    def test_export_openai_tolerates_minimal_pair_schema(self):
        import json
        from finetune_studio.data.prep.export import export_qa_jsonl
        from unittest.mock import patch
        minimal_pairs = [{
            "id": "qa_x", "source_id": "s", "question": "q",
            "answer": "a", "status": "approved",
        }]
        with patch("finetune_studio.data.prep.export.pfs") as mock_pfs:
            mock_pfs.list_qa_pairs.return_value = minimal_pairs
            result = export_qa_jsonl("pid", fmt="openai", only="approved")
            obj = json.loads(result.strip())
            assert obj["messages"][0]["role"] == "user"
            assert obj["messages"][1]["role"] == "assistant"
