"""run_rag_case is corpus-agnostic: one retrieval, one generation, whatever the question or the table looks like.

It used to re-ask the model with a hard-coded "sum the actual_hours column, not variance_hours" correction whenever a question
mentioned "total actual" and the context had those two column names (a Korvane-specific patch). A run must record what the model
said the first time, so results are comparable across corpora and readers.
"""
from __future__ import annotations

from finetune_studio.testing import rag_suite
from finetune_studio.testing.suite import BenchmarkCase
from tests.test_rag_suite import FakeRag

TABLE = "month | site | actual_hours | variance_hours\n--- | --- | --- | ---\nMay | A | 54 | 3\nMay | B | 29 | -2"
QUESTION = "What was the total actual overtime hours in May?"


class CountingEngine:
    model_path = "/m/x"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def generate(self, messages, **_kw) -> str:
        self.calls.append(list(messages))
        return self.reply


def _case() -> BenchmarkCase:
    return BenchmarkCase(name="hours", category="qa", question=QUESTION, correct_answer="83")


def test_a_variance_style_answer_is_not_asked_again() -> None:
    rag = FakeRag({QUESTION: [{"rank": 1, "document_id": "d", "text": TABLE}]})
    engine = CountingEngine("54 plus 29, totaling 83 variance hours.")
    (result,) = rag_suite.run_rag_suite(engine, rag, [_case()])
    assert len(engine.calls) == 1
    assert result.case_result.model_answer == "54 plus 29, totaling 83 variance hours."
    assert [m["role"] for m in result.case_result.transcript] == ["system", "user", "assistant"]
    assert "actual_hours column" not in str(result.case_result.transcript)


def test_the_corpus_specific_helper_is_gone() -> None:
    assert not hasattr(rag_suite, "_needs_table_arithmetic_retry")


def test_the_shared_prompt_names_no_column() -> None:
    system = rag_suite.RAG_SYSTEM_PROMPT.lower()
    assert "variance" not in system and "actual_hours" not in system
    assert "row and column named by the question" in system
