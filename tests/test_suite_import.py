"""Bring-your-own quiz: import, discovery in the Testing page, and what the run records for the judge to read."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.testing.judge import JudgeCase, build_judge_messages
from finetune_studio.testing.suite import load_test_suite, run_suite
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
    assert first.keywords == ["INC-2024-0614"] and not first.expect_abstain and first.correct_answer == "INC-2024-0614"
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


def _imported_cases(project_env):
    _client, pid, _ = project_env
    return load_test_suite(import_suite(pid, "korvane quiz.jsonl", QUIZ.read_bytes())["path"])


def test_running_an_imported_quiz_records_answers_and_never_a_verdict(project_env) -> None:
    class Echo:
        def generate(self, messages, **_kw):
            return "The provided documents do not contain this information."

    results = run_suite(Echo(), _imported_cases(project_env)[:3])
    assert [r.model_answer for r in results] == ["The provided documents do not contain this information."] * 3
    assert all(r.verdict == "" and r.judge == "none" for r in results)  # judged afterwards, never by a matcher
    assert results[0].keywords == ["INC-2024-0614"]


def test_the_judge_sees_the_expected_values_and_the_unanswerable_flag(project_env) -> None:
    cases = _imported_cases(project_env)
    answerable = cases[0]
    prompt = build_judge_messages(JudgeCase(
        answerable.question, answerable.correct_answer, "some answer", keywords=list(answerable.keywords),
        expect_abstain=answerable.expect_abstain))[1]["content"]
    # a short row's answer key IS its expected values, so the prompt does not list them twice
    assert "ANSWER KEY:\nINC-2024-0614" in prompt and "KEY VALUES" not in prompt and "UNANSWERABLE" not in prompt
    native = JudgeCase("q?", "Invoice 77 was paid on 4 March.", "paid", keywords=["77", "4 March"])
    assert "KEY VALUES:\n- 77\n- 4 March" in build_judge_messages(native)[1]["content"]
    unanswerable = cases[-1]
    prompt = build_judge_messages(JudgeCase(
        unanswerable.question, unanswerable.correct_answer, "PLN 5,000,000",
        keywords=list(unanswerable.keywords), expect_abstain=unanswerable.expect_abstain))[1]["content"]
    assert "UNANSWERABLE" in prompt and "KEY VALUES" not in prompt


def test_quiz_cases_survive_a_saved_run_with_their_expectations(project_env) -> None:
    """keywords / expect_abstain are stored on the case row, so the judge step sees them after a restart."""
    _client, pid, _ = project_env
    cases = _imported_cases(project_env)
    rid = db.create_run(pid, "run")["id"]
    bid = db.create_benchmark(rid, "quiz", {}, status="running", kind="suite")["id"]
    for c in (cases[0], cases[-1]):
        db.create_case(bid, rid, c.name, c.category, c.question, c.correct_answer, "x", [],
                       keywords=list(c.keywords), expect_abstain=c.expect_abstain)
    first, last = db.list_cases(bid)
    assert first["keywords"] == ["INC-2024-0614"] and not first["expect_abstain"]
    assert last["expect_abstain"] is True


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
