"""The post-training quiz reads every SFT row shape training accepts (testing/generate_suite.py)."""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.testing.generate_suite import generate_suite_from_training_data


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _cases(result: dict) -> list[dict]:
    return json.loads(Path(result["suite_path"]).read_text(encoding="utf-8"))["cases"]


def test_chat_messages_rows_become_quiz_cases(tmp_path: Path) -> None:
    # Live Track A: an uploaded (converted) dataset is `messages` JSONL; every row was skipped ("no valid cases").
    rows = [{"messages": [{"role": "system", "content": "Answer from the handbook."},
                          {"role": "user", "content": "What is the incident reference?"},
                          {"role": "assistant", "content": "INC-2024-0614."}]},
            {"messages": [{"role": "user", "content": "Which trailer?"}, {"role": "assistant", "content": "FL-GDY-774."}]}]
    result = generate_suite_from_training_data(_write(tmp_path / "d.jsonl", rows), str(tmp_path / "out"))
    assert result.get("case_count") == 2, result
    first = _cases(result)[0]
    assert (first["question"], first["correct_answer"], first["system_prompt"]) == (
        "What is the incident reference?", "INC-2024-0614.", "Answer from the handbook.")


def test_prompt_completion_rows_and_sharegpt_still_work(tmp_path: Path) -> None:
    rows = [{"prompt": "Capital of Poland?", "completion": "Warsaw."},
            {"conversations": [{"from": "human", "value": "Which vault?"}, {"from": "gpt", "value": "Vault 7."}]},
            {"prompt": "Bad?", "completion": "Bad answer.", "label": False}]   # KTO undesirable: never a quiz case
    result = generate_suite_from_training_data(_write(tmp_path / "d.jsonl", rows), str(tmp_path / "out"))
    assert sorted(c["correct_answer"] for c in _cases(result)) == ["Vault 7.", "Warsaw."]
