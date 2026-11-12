from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import answers as answer_collection
import pytest
from sft_holdout import sft_fact_questions
from train_variants import build_rows


def _base_rows(count: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = [
        {"fact": f"fact-{i}", "messages": [{"role": "user", "content": f"Q{i}"},
                                             {"role": "assistant", "content": f"A{i}"}]}
        for i in range(count)
    ]
    facts = {
        row["fact"]: {"id": row["fact"], "attr": "asset:serial_no", "etype": "asset", "entity": row["fact"]}
        for row in rows
    }
    return rows, facts


def test_holdout_is_fact_disjoint_and_stable_across_grouped_augmentation() -> None:
    base_rows, facts = _base_rows(100)
    grouped_rows = build_rows(base_rows, facts)

    base_questions = sft_fact_questions(base_rows, base_rows)
    grouped_questions = sft_fact_questions(grouped_rows, base_rows)
    base_by_fact = {(question["type"], question["fact"]): question for question in base_questions}
    grouped_by_fact = {(question["type"], question["fact"]): question for question in grouped_questions}

    assert len(base_questions) == len(grouped_questions) == 20
    assert {question["id"] for question in base_questions} == {question["id"] for question in grouped_questions}
    assert grouped_by_fact.keys() == base_by_fact.keys()
    assert all(base_by_fact[key]["question"] == row["question"] for key, row in grouped_by_fact.items())
    trained = {fact for scope, fact in grouped_by_fact if scope == "sft-trained"}
    heldout = {fact for scope, fact in grouped_by_fact if scope == "sft-heldout"}
    assert len(trained) == len(heldout) == 10
    assert trained.isdisjoint(heldout)

    rowwise_rows = [{key: value for key, value in row.items() if key != "pair_id"} for row in grouped_rows]
    with pytest.raises(ValueError, match="leaks .* facts into training"):
        sft_fact_questions(rowwise_rows, base_rows)


def test_answer_collector_preserves_fact_and_key_for_source_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = {
        "id": "sft-trained-test",
        "type": "sft-trained",
        "question": "What is the serial number?",
        "answer": "The serial number is SN-001.",
        "accept": [],
        "wrong": [],
        "formats": [],
        "etype": "sft",
        "attr": "fact",
        "fact": "asset:example:serial_no",
    }
    monkeypatch.setattr(answer_collection, "chat", lambda *_args, **_kwargs: "The serial number is SN-001.")
    output = tmp_path / "answers.jsonl"

    count = answer_collection.collect(
        [question], output, mode="closed", base_url="http://127.0.0.1:1", workers=1, label="test"
    )
    row = json.loads(output.read_text())

    assert count == 1
    assert row["fact"] == question["fact"]
    assert row["key"] == question["answer"]
