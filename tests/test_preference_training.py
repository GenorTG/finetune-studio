"""DPO data contracts, synchronous validation, and chosen-answer eval coverage."""
from __future__ import annotations

import json

pytest_plugins = ("tests.test_training_start_guard",)

from tests.test_training_start_guard import _project, _start


def _rows() -> list[dict]:
    return [
        {
            "prompt": [{"role": "user", "content": "What is the safe action?"}],
            "chosen": [{"role": "assistant", "content": "Check the lockout."}],
            "rejected": [{"role": "assistant", "content": "Ignore the lockout."}],
        },
        {
            "prompt": "How should I verify it?",
            "chosen": "Measure it twice.",
            "rejected": "Guess based on appearance.",
        },
    ]


def test_format_for_preference_normalizes_strings_and_bakes_system_prompt() -> None:
    from finetune_studio.training.data import format_for_preference

    output = format_for_preference(_rows(), "Be precise.")
    assert len(output) == 2
    assert output[0]["prompt"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "What is the safe action?"},
    ]
    assert output[1]["prompt"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "How should I verify it?"},
    ]
    assert output[1]["chosen"] == [{"role": "assistant", "content": "Measure it twice."}]


def test_format_for_preference_rejects_bad_roles_and_empty_pairs() -> None:
    from finetune_studio.training.data import format_for_preference

    row = _rows()[0]
    row["prompt"] = [{"role": "assistant", "content": "Wrong prompt role"}]
    try:
        format_for_preference([row])
    except ValueError as exc:
        assert "prompt must end with a user" in str(exc)
    else:
        raise AssertionError("malformed DPO prompt passed validation")

    row = _rows()[0]
    row["rejected"] = row["chosen"]
    try:
        format_for_preference([row])
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("identical preference pair passed validation")


def test_sft_preserves_tool_schemas_and_continued_pretraining_keeps_raw_text() -> None:
    from finetune_studio.training.data import (
        format_for_continued_pretraining,
        format_for_sft,
    )

    schema = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    tool_row = {
        "messages": [
            {"role": "user", "content": "Look this up."},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}]},
            {"role": "tool", "name": "lookup", "content": "Found it."},
            {"role": "assistant", "content": "Here is the result."},
        ],
        "tools": schema,
    }
    formatted = format_for_sft([tool_row])
    assert formatted[0]["tools"] == schema
    assert formatted[0]["messages"][1]["tool_calls"] == tool_row["messages"][1]["tool_calls"]
    from datasets import Dataset
    arrow_rows = Dataset.from_list(formatted, on_mixed_types="use_json")
    assert arrow_rows[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert format_for_continued_pretraining([{"text": "  Domain corpus.  "}]) == [{"text": "Domain corpus."}]


def test_continued_pretraining_rejects_chat_rows() -> None:
    from finetune_studio.training.data import format_for_continued_pretraining

    try:
        format_for_continued_pretraining([{"messages": [{"role": "user", "content": "text"}]}])
    except ValueError as exc:
        assert "requires non-empty 'text'" in str(exc)
    else:
        raise AssertionError("CPT accepted a chat-only row")


def test_one_row_sft_set_remains_trainable_without_a_fake_validation_row() -> None:
    from finetune_studio.training.data import split_data

    row = {"messages": [{"role": "user", "content": "q"}]}
    train, validation = split_data([row])
    assert train == [row]
    assert validation == []


def test_training_route_rejects_invalid_dpo_before_worker_start(
    client, fake_engine, fake_home, tmp_path,
) -> None:
    pid = _project(client)
    data_path = tmp_path / "bad-preferences.jsonl"
    data_path.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()

    response = _start(
        client, pid, str(data_path), str(model_path), training_mode="dpo",
    )

    assert response.status_code == 400
    assert "Invalid preference dataset" in response.json()["error"]
    assert not fake_engine.started


def test_training_route_passes_dpo_mode_to_engine(
    client, fake_engine, fake_home, tmp_path,
) -> None:
    pid = _project(client)
    data_path = tmp_path / "preferences.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()

    response = _start(
        client, pid, str(data_path), str(model_path), training_mode="dpo",
    )

    assert response.status_code == 200, response.text
    assert fake_engine.started["config"].training_mode == "dpo"


def test_training_route_requires_tool_schema_for_tool_sft(
    client, fake_engine, fake_home, tmp_path,
) -> None:
    pid = _project(client)
    data_path = tmp_path / "tool-sft.jsonl"
    data_path.write_text(json.dumps({"messages": [{"role": "user", "content": "Use a tool."}]}) + "\n", encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()

    response = _start(client, pid, str(data_path), str(model_path), training_mode="tool_sft")

    assert response.status_code == 400
    assert "function JSON-schema" in response.json()["error"]


def test_training_route_accepts_cpt_and_passes_mode_to_engine(
    client, fake_engine, fake_home, tmp_path,
) -> None:
    pid = _project(client)
    data_path = tmp_path / "domain.jsonl"
    data_path.write_text("".join(json.dumps({"text": f"Domain sample {n}."}) + "\n" for n in range(2)), encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()

    response = _start(client, pid, str(data_path), str(model_path), training_mode="continued_pretraining")

    assert response.status_code == 200, response.text
    assert fake_engine.started["config"].training_mode == "continued_pretraining"


def test_dpo_quiz_uses_chosen_not_rejected_response(tmp_path) -> None:
    from finetune_studio.testing.generate_suite import generate_suite_from_training_data

    data_path = tmp_path / "preferences.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    result = generate_suite_from_training_data(str(data_path), str(tmp_path / "suite"))

    assert result["case_count"] == 2
    suite = json.loads((tmp_path / "suite" / "suite_preferences.json").read_text())
    cases = suite["cases"]
    by_question = {case["question"]: case["correct_answer"] for case in cases}
    assert by_question["What is the safe action?"] == "Check the lockout."
    assert by_question["What is the safe action?"] != "Ignore the lockout."
    assert by_question["How should I verify it?"] == "Measure it twice."
