"""Use-case metrics: extraction exact match + field-F1, classification accuracy + confusion matrix; invalid = fail."""
from __future__ import annotations

import json

import pytest

from finetune_studio.usecases import metrics as m
from finetune_studio.usecases import schema as s

EXPECTED = {"invoice": "2026-114", "vendor": "Acme GmbH", "total": 1240.0, "tags": ["a", "b"]}
EXPECTED_JSON = json.dumps(EXPECTED)


@pytest.mark.parametrize(("reply", "only"), [
    (EXPECTED_JSON, True),
    ("```json\n" + EXPECTED_JSON + "\n```", True),
    ("<think>the user wants {json}</think>" + EXPECTED_JSON, True),
    ("Here you go: " + EXPECTED_JSON + " Hope it helps.", False),
])
def test_the_json_object_is_found_and_extra_text_is_reported(reply: str, only: bool) -> None:
    obj, json_only = m.parse_json_reply(reply)
    assert obj == EXPECTED and json_only is only


def test_braces_inside_strings_do_not_confuse_the_object_finder() -> None:
    obj, _ = m.parse_json_reply('Result: {"note": "a } and a \\" quote {", "n": 1} done')
    assert obj == {"note": 'a } and a " quote {', "n": 1}


def test_a_reply_without_an_object_fails_and_is_invalid() -> None:
    score = m.score_extraction("I cannot extract that.", EXPECTED_JSON)
    assert (score.verdict, score.json_valid, score.f1) == ("fail", False, 0.0)


def test_exact_object_passes_and_number_forms_and_list_order_do_not_matter_for_f1() -> None:
    same = m.score_extraction(json.dumps({**EXPECTED, "total": 1240}), EXPECTED_JSON)
    assert same.verdict == "pass" and same.exact and same.f1 == 1.0
    reordered = m.score_extraction(json.dumps({**EXPECTED, "tags": ["b", "a"]}), EXPECTED_JSON)
    assert reordered.verdict == "partial" and reordered.f1 == 1.0  # not the exact object, but every leaf is right


def test_field_f1_counts_right_wrong_missing_and_extra_leaves() -> None:
    reply = {"invoice": "2026-114", "vendor": "ACME  gmbh", "total": 999, "extra": 1}
    precision, recall, f1 = m.field_f1(reply, EXPECTED)
    assert (precision, recall) == (2 / 4, 2 / 5)  # invoice + vendor (case/space-insensitive) right
    assert f1 == pytest.approx(2 * 0.5 * 0.4 / 0.9)
    assert m.field_f1({}, {}) == (1.0, 1.0, 1.0)


def test_schema_validity_is_reported_next_to_the_score() -> None:
    schema = {"type": "object", "required": ["invoice"], "properties": {"invoice": {"type": "string"}}}
    s.check_schema(schema)
    assert m.score_extraction('{"invoice": 5}', '{"invoice": "5"}', schema).schema_valid is False
    assert m.score_extraction('{"invoice": "5"}', '{"invoice": "5"}', schema).schema_valid is True
    assert m.score_extraction('{"invoice": "5"}', '{"invoice": "5"}').schema_valid is None


LABELS = ["billing", "shipping", "account access"]


@pytest.mark.parametrize(("reply", "label", "only"), [
    ("billing", "billing", True),
    ("  **Billing**.", "billing", True),
    ("Label: shipping", "shipping", True),
    ("`account access`", "account access", True),
    ("This is about billing.", "billing", False),
    ("<think>maybe shipping</think>billing", "billing", True),
    ("billing or shipping", m.NO_LABEL, False),
    ("I am not sure.", m.NO_LABEL, False),
    ("rebilling", m.NO_LABEL, False),
])
def test_labels_are_read_strictly(reply: str, label: str, only: bool) -> None:
    assert m.read_label(reply, LABELS) == (label, only)


def test_confusion_matrix_has_a_no_label_column_and_per_label_recall() -> None:
    pairs = [("billing", "billing"), ("billing", "shipping"), ("shipping", "shipping"), ("account access", m.NO_LABEL)]
    c = m.confusion(pairs, LABELS)
    assert c["columns"] == [*LABELS, m.NO_LABEL]
    assert c["matrix"]["billing"] == {"billing": 1, "shipping": 1, "account access": 0, m.NO_LABEL: 1 - 1}
    assert c["matrix"]["account access"][m.NO_LABEL] == 1
    assert c["per_label"]["billing"] == {"support": 2, "precision": 1.0, "recall": 0.5}
    assert c["per_label"]["shipping"]["precision"] == 0.5 and c["per_label"]["account access"]["precision"] is None


def test_summaries_skip_errored_cases_and_report_the_template_metrics() -> None:
    cases = [{"model_answer": "billing", "correct_answer": "billing"},
             {"model_answer": "The answer is shipping", "correct_answer": "billing"},
             {"model_answer": "", "correct_answer": "shipping", "error": "load failed"}]
    out = m.summarize("classification", cases, {"labels": LABELS})
    assert (out["answered"], out["accuracy"], out["label_only"], out["no_label"]) == (2, 0.5, 0.5, 0.0)
    ext = m.summarize("extraction", [{"model_answer": EXPECTED_JSON, "correct_answer": EXPECTED_JSON},
                                     {"model_answer": "no", "correct_answer": EXPECTED_JSON}], {})
    assert (ext["exact_match"], ext["field_f1"], ext["json_valid"], ext["json_only"]) == (0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="not scored by code"):
        m.summarize("style", cases, {})


def test_schema_subset_validator() -> None:
    schema = {"type": "object", "additionalProperties": False, "required": ["a"],
              "properties": {"a": {"type": "integer"}, "b": {"type": "array", "items": {"enum": ["x", "y"]}},
                             "c": {"type": ["string", "null"]}}}
    s.check_schema(schema)
    assert s.validate({"a": 1, "b": ["x"], "c": None}, schema) == []
    assert s.validate({"a": 1.0}, schema) == []  # 1.0 is an integer value in JSON
    assert s.validate({"a": True, "b": ["z"], "d": 1}, schema) == [
        "$: unexpected field 'd'", "$.a: expected integer, got bool", "$.b[0]: 'z' is not one of ['x', 'y']"]
    for bad, msg in [({"type": "objekt"}, "unknown type"), ({"additionalProperties": {}}, "true or false"),
                     ({"required": "a"}, "list of field names"), ([], "JSON object")]:
        with pytest.raises(s.SchemaError, match=msg):
            s.check_schema(bad)
