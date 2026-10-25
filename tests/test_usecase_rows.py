"""Use-case rows: every example validated, a seeded held-out split the training rows never contain, one instruction."""
from __future__ import annotations

import json

import pytest

from finetune_studio.usecases import rows as r
from finetune_studio.usecases.schema import SchemaError

LABELS = ("billing", "shipping", "account")


def _cls(n_per_label: int = 30) -> list[dict]:
    return [{"input": f"ticket {i} about {lab} number {i}", "label": lab}
            for lab in LABELS for i in range(n_per_label)]


def _ext(n: int = 80) -> list[dict]:
    return [{"input": f"Invoice {i} from Vendor{i}, total {i}.50 EUR",
             "output": {"invoice": str(i), "vendor": f"Vendor{i}", "total": i + 0.5}} for i in range(n)]


def _pairs(n: int = 80, answer: str = "answer") -> list[dict]:
    return [{"prompt": f"question {i}?", answer: f"reply {i}."} for i in range(n)]


def test_classification_split_is_stratified_disjoint_and_seeded() -> None:
    p = r.prepare("classification", _cls(), r.BuildOptions(seed=7))
    assert p.labels == sorted(LABELS) and p.examples == 90 and len(p.heldout_cases) == 20
    held_labels = [c["correct_answer"] for c in p.heldout_cases]
    assert set(held_labels) == set(LABELS) and min(held_labels.count(lab) for lab in LABELS) >= 6
    train_inputs = {row["messages"][1]["content"] for row in p.train_rows}
    assert not train_inputs & {c["question"] for c in p.heldout_cases}
    assert len(train_inputs) + len(p.heldout_cases) == 90
    again = r.prepare("classification", _cls(), r.BuildOptions(seed=7))
    assert again.heldout_cases == p.heldout_cases
    assert r.prepare("classification", _cls(), r.BuildOptions(seed=8)).heldout_cases != p.heldout_cases


def test_every_row_and_heldout_question_carry_the_same_instruction() -> None:
    p = r.prepare("classification", _cls(), r.BuildOptions())
    assert p.instruction.endswith("account, billing, shipping")
    for row in p.train_rows:
        assert row["messages"][0] == {"role": "system", "content": p.instruction}
        assert row["messages"][2]["role"] == "assistant" and row["messages"][2]["content"] in LABELS
    assert {c["system_prompt"] for c in p.heldout_cases} == {p.instruction}
    assert {c["category"] for c in p.heldout_cases} == {"usecase:classification"}


def test_a_custom_instruction_can_name_the_labels() -> None:
    p = r.prepare("classification", _cls(), r.BuildOptions(instruction="Route to one of: {labels}."))
    assert p.instruction == "Route to one of: account, billing, shipping."


def test_bad_rows_refuse_the_build_with_their_row_numbers() -> None:
    rows = [*_cls(), {"input": "", "label": "billing"}, {"text": "fine", "label": ""}, "not an object"]
    with pytest.raises(r.RowsError) as e:
        r.prepare("classification", rows, r.BuildOptions())
    assert "3 of 93 examples are not usable" in str(e.value)
    assert e.value.errors == ["row 91: missing input / text text", "row 92: missing label / class / category",
                              "row 93: not a JSON object"]


def test_a_label_with_too_few_examples_is_refused_by_name() -> None:
    rows = _cls() + [{"input": f"refund {i}", "label": "refunds"} for i in range(4)]
    with pytest.raises(r.TooFewRows, match=r"too few: refunds \(4\)"):
        r.prepare("classification", rows, r.BuildOptions())


def test_label_list_must_cover_the_examples_and_differ_beyond_case() -> None:
    with pytest.raises(r.RowsError, match="not in the label list: shipping"):
        r.prepare("classification", _cls(), r.BuildOptions(labels=["billing", "account"]))
    rows = _cls() + [{"input": f"x {i}", "label": "Billing"} for i in range(10)]
    with pytest.raises(r.RowsError, match="upper/lower case: Billing, billing"):
        r.prepare("classification", rows, r.BuildOptions())


def test_too_few_examples_says_how_many_are_needed() -> None:
    with pytest.raises(r.TooFewRows, match="needs at least 70 distinct examples .50 to train \\+ 20 held-out.*got 60"):
        r.prepare("style", _pairs(60), r.BuildOptions())


def test_duplicates_are_merged_and_conflicting_inputs_dropped_with_a_warning() -> None:
    rows = _pairs(80) + _pairs(5) + [{"prompt": "question 1?", "answer": "a different reply"}]
    p = r.prepare("style", rows, r.BuildOptions())
    assert p.dropped_duplicates == 4 and p.dropped_conflicts == 3 and p.examples == 79
    assert any("3 rows dropped" in w for w in p.warnings)


def test_extraction_targets_are_canonical_json_and_follow_the_schema() -> None:
    schema = {"type": "object", "required": ["invoice", "vendor", "total"], "additionalProperties": False,
              "properties": {"invoice": {"type": "string"}, "vendor": {"type": "string"}, "total": {"type": "number"}}}
    p = r.prepare("extraction", _ext(), r.BuildOptions(schema=schema))
    assert "JSON schema of the object" in p.instruction and '"additionalProperties": false' in p.instruction
    target = json.loads(p.heldout_cases[0]["correct_answer"])
    assert set(target) == {"invoice", "vendor", "total"}
    bad = _ext()
    bad[3]["output"] = {"invoice": 3, "vendor": "V"}
    bad[5]["output"] = "{not json"
    with pytest.raises(r.RowsError) as e:
        r.prepare("extraction", bad, r.BuildOptions(schema=schema))
    assert e.value.errors[0] == ("row 4: output does not follow the schema: $: missing required field 'total'; "
                                 "$.invoice: expected string, got int")
    assert e.value.errors[1] == "row 6: output is not valid JSON"


def test_extraction_without_schema_lists_the_fields_and_refuses_non_objects() -> None:
    p = r.prepare("extraction", _ext(), r.BuildOptions())
    assert p.instruction.endswith("Fields: invoice, vendor, total")
    with pytest.raises(r.RowsError, match="1 of 80"):
        r.prepare("extraction", [*_ext(79), {"input": "x", "output": [1, 2]}], r.BuildOptions())


def test_an_unsupported_schema_keyword_is_refused_not_ignored() -> None:
    with pytest.raises(SchemaError, match="unsupported schema keyword.*pattern"):
        r.prepare("extraction", _ext(), r.BuildOptions(schema={"type": "object", "pattern": "^a"}))


def test_style_uses_the_style_guide_as_the_instruction_and_distillation_reads_teacher_answers() -> None:
    p = r.prepare("style", _pairs(80), r.BuildOptions(style_guide="Short, warm, no jargon."))
    assert p.instruction == "Short, warm, no jargon." and p.train_rows[0]["messages"][0]["role"] == "system"
    plain = r.prepare("style", _pairs(80), r.BuildOptions())
    assert plain.instruction == "" and plain.train_rows[0]["messages"][0]["role"] == "user"
    d = r.prepare("distillation", _pairs(130, "teacher_answer"), r.BuildOptions(teacher="big-model"))
    assert d.teacher == "big-model" and len(d.heldout_cases) == 26
    with pytest.raises(r.TooFewRows):
        r.prepare("distillation", _pairs(110, "teacher_answer"), r.BuildOptions())


def test_small_sets_warn_about_the_recommended_range_and_the_quiz_size() -> None:
    p = r.prepare("style", _pairs(80), r.BuildOptions())
    assert any("gains were measured with 100-500" in w for w in p.warnings)
    assert any("20 held-out questions" in w for w in p.warnings)


@pytest.mark.parametrize("raw", [
    b'[{"prompt": "a", "answer": "b"}]',
    b'{"rows": [{"prompt": "a", "answer": "b"}]}',
    b'{"prompt": "a", "answer": "b"}\n\n',
])
def test_upload_accepts_a_list_a_rows_object_or_jsonl(raw: bytes) -> None:
    assert r.parse_upload(raw) == [{"prompt": "a", "answer": "b"}]


def test_upload_errors_are_specific() -> None:
    with pytest.raises(r.RowsError, match="empty"):
        r.parse_upload(b"  ")
    with pytest.raises(r.RowsError, match="line 2 is not valid JSON"):
        r.parse_upload(b'{"a": 1}\n{oops\n')
