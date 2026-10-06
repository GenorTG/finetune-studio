"""Grounded rows are quizzed the way they were trained: with their retrieved CONTEXT.

A grounded training row teaches "answer from the context in the system turn", so asking its
question bare tests a skill the row never taught (real run: plain rows 15/15 recalled, grounded
rows 1/9 bare but 9/9 with their own context).
"""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.testing.generate_suite import generate_suite_from_training_data
from finetune_studio.testing.suite import load_test_suite, run_suite

CONTEXT = "Answer using ONLY the context below.\n\nCONTEXT:\n[1] Vault 7 holds ledger 7."


def _dataset(tmp_path: Path) -> str:
    rows = [
        {"conversations": [
            {"from": "system", "value": CONTEXT},
            {"from": "human", "value": "Which vault holds ledger 7?"},
            {"from": "gpt", "value": "Vault 7 holds ledger 7."},
        ], "source_id": "s1", "chunk_idx": 1},
        {"conversations": [
            {"from": "human", "value": "Which vault holds ledger 8?"},
            {"from": "gpt", "value": "Vault 8 holds ledger 8."},
        ], "source_id": "s1", "chunk_idx": 2},
    ]
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


class _Engine:
    """Records the messages each case sent."""

    def __init__(self) -> None:
        self.sent: list[list[dict]] = []

    def generate(self, messages, **_kw):
        self.sent.append(messages)
        return "answer"


def test_generated_suite_carries_the_rows_system_turn(tmp_path: Path) -> None:
    r = generate_suite_from_training_data(_dataset(tmp_path), str(tmp_path / "out"))
    assert r["with_context_count"] == 1
    doc = json.loads(Path(r["suite_path"]).read_text(encoding="utf-8"))
    assert doc["meta"]["with_context_count"] == 1
    grounded, plain = doc["cases"]
    assert grounded["system_prompt"] == CONTEXT
    assert plain["system_prompt"] == ""


def test_loader_round_trips_system_prompt(tmp_path: Path) -> None:
    r = generate_suite_from_training_data(_dataset(tmp_path), str(tmp_path / "out"))
    grounded, plain = load_test_suite(r["suite_path"])
    assert grounded.system_prompt == CONTEXT and plain.system_prompt == ""


def test_runner_sends_the_case_context_and_leaves_plain_cases_bare(tmp_path: Path) -> None:
    r = generate_suite_from_training_data(_dataset(tmp_path), str(tmp_path / "out"))
    engine = _Engine()
    results = run_suite(engine, load_test_suite(r["suite_path"]))
    assert engine.sent[0] == [{"role": "system", "content": CONTEXT},
                              {"role": "user", "content": "Which vault holds ledger 7?"}]
    assert engine.sent[1] == [{"role": "user", "content": "Which vault holds ledger 8?"}]
    # the transcript is how the UI tells "answered with context" from "answered from memory"
    assert results[0].transcript[0]["role"] == "system"
    assert results[1].transcript[0]["role"] == "user"


def test_suite_level_system_prompt_still_applies_to_plain_cases(tmp_path: Path) -> None:
    r = generate_suite_from_training_data(_dataset(tmp_path), str(tmp_path / "out"))
    engine = _Engine()
    run_suite(engine, load_test_suite(r["suite_path"]), system_prompt="Be terse.")
    assert engine.sent[0][0]["content"] == CONTEXT   # the case's own context wins
    assert engine.sent[1][0] == {"role": "system", "content": "Be terse."}
