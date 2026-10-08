"""Bring-your-own quiz: import, discovery in the Testing page, run scoring (all-values pass, expected-abstention)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from finetune_studio.testing.judge import is_abstention
from finetune_studio.testing.suite import CaseResult, _judge_each, load_test_suite
from finetune_studio.testing.suite_import import SuiteImportError, import_suite

QUIZ = Path(__file__).resolve().parent / "corpus" / "korvane" / "eval" / "korvane_quiz_core.jsonl"


def test_the_shipped_korvane_quiz_is_the_two_eval_sets_together() -> None:
    eval_dir = QUIZ.parent
    answerable = [json.loads(x) for x in (eval_dir / "paraphrase_core.jsonl").read_text().splitlines() if x.strip()]
    unanswerable = [json.loads(x) for x in (eval_dir / "unanswerable_core.jsonl").read_text().splitlines() if x.strip()]
    quiz = [json.loads(x) for x in QUIZ.read_text().splitlines() if x.strip()]
    assert [(r["id"], r["q"]) for r in quiz] == [(r["id"], r["q"]) for r in answerable + unanswerable]
    assert all(r.get("expect_abstain") for r in quiz[len(answerable):]) and all(r["expect"] for r in quiz[:len(answerable)])


def test_import_normalises_short_rows_and_marks_unanswerable_ones(project_env) -> None:
    _client, pid, _ = project_env
    meta = import_suite(pid, "korvane quiz.jsonl", QUIZ.read_bytes())
    assert meta["case_count"] == 122 and meta["abstain_cases"] == 20 and meta["name"] == "korvane-quiz"
    cases = load_test_suite(meta["path"])
    first, last = cases[0], cases[-1]
    assert first.keywords == ["INC-2024-0614"] and not first.expect_abstain and first.correct_answer.startswith("Expected:")
    assert last.expect_abstain and last.category == "unanswerable"


@pytest.mark.parametrize("raw,fragment", [
    (b"", "empty"),
    (b"not json at all\n{", "line 1"),
    (b'[{"q": "no expectation"}]', "expect_abstain"),
    (b'[{"id": "a", "q": "x?", "expect": ["1"]}, {"id": "a", "q": "y?", "expect": ["2"]}]', "unique"),
    (b'[{"id": "a", "expect": ["1"]}]', "no question"),
])
def test_import_rejects_unusable_files_with_a_reason(project_env, raw: bytes, fragment: str) -> None:
    _client, pid, _ = project_env
    with pytest.raises(SuiteImportError, match=fragment):
        import_suite(pid, "bad.json", raw)


def _judged(question: str, answer: str, *, keywords: list[str] | None = None, abstain: bool = False) -> CaseResult:
    r = CaseResult(case_name="c", category="quiz", question=question, correct_answer="Expected: x", model_answer=answer,
                   keywords=keywords or [], expect_abstain=abstain)
    _judge_each([r])
    return r


def test_all_expected_values_pass_some_are_partial_none_fail() -> None:
    kws = ["HELV-PH-0917", "18"]
    assert _judged("q", "Trailer HELV-PH-0917 carried 18 pallets.", keywords=kws).verdict == "pass"
    assert _judged("q", "Trailer HELV-PH-0917 carried a few pallets.", keywords=kws).verdict == "partial"
    assert _judged("q", "I am not sure.", keywords=kws).verdict in ("fail", "partial")


def test_an_unanswerable_question_passes_only_when_the_model_declines() -> None:
    declined = _judged("q", "The provided documents do not contain this information.", abstain=True)
    invented = _judged("q", "The registered capital is PLN 5,000,000.", abstain=True)
    assert (declined.verdict, declined.scoring_method) == ("pass", "abstention_expected")
    assert invented.verdict == "fail" and "confident answer" in invented.judge_reasoning
    assert is_abstention("That isn't something the files state.", broad=True) and not is_abstention("PLN 5,000,000", broad=True)


def test_upload_route_registers_the_quiz_in_the_suite_list(project_env) -> None:
    client, pid, _ = project_env
    r = client.post(f"/api/benchmarks/projects/{pid}/suites/import", files={"file": ("quiz.jsonl", io.BytesIO(QUIZ.read_bytes()))})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["case_count"] == 122
    listed = client.get("/api/benchmarks/suites", params={"project_id": pid}).json()
    mine = [s for s in listed if s.get("source") == "project_import"]
    assert len(mine) == 1 and mine[0]["path"] == body["path"] and mine[0]["case_count"] == 122
    bad = client.post(f"/api/benchmarks/projects/{pid}/suites/import", files={"file": ("x.json", io.BytesIO(b"[]"))})
    assert bad.status_code == 400 and "error" in bad.json()
    assert client.post("/api/benchmarks/projects/nope/suites/import", files={"file": ("x.json", io.BytesIO(b"[]"))}).status_code == 404
