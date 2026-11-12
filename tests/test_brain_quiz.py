"""A quiz written from the documents: bad items are dropped, unanswerable questions are verified, it saves as a suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio.testing import brain_quiz as bq

PASSAGE = "The Oslo warehouse has 412 pallet positions and is managed by Ingrid Solberg since March 2021, per the site file. Deliveries arrive at the north gate and every pallet is scanned on arrival."


def row(doc: str = "d1", text: str = PASSAGE) -> dict[str, str]:
    return {"document_id": doc, "filename": f"{doc}.txt", "text": text}


def reply(**item) -> str:
    return json.dumps(item)


def good() -> str:
    return reply(question="How many pallet positions does the Oslo warehouse have?", answer="412", must_contain=["412"])


def test_a_good_item_is_kept_with_its_key() -> None:
    report = bq.BrainQuizReport()
    case = bq.answerable_case(row(), good(), report)
    assert case == {"question": "How many pallet positions does the Oslo warehouse have?", "correct_answer": "412",
                    "expect": ["412"], "category": "documents"}
    assert report.dropped == {}


@pytest.mark.parametrize(("text", "reason"), [
    ("not json at all", "not JSON"),
    (reply(), "no usable fact"),
    (reply(question="Short?", answer="412"), "question does not stand alone"),
    (reply(question="What number does this document give for positions?", answer="412"), "question does not stand alone"),
    (reply(question="How many pallet positions does the Oslo warehouse have?", answer="500", must_contain=["500"]),
     "answer has a value that is not in the passage"),
    (reply(question="Does the Oslo warehouse really have 412 pallet positions?", answer="412", must_contain=["412"]),
     "question gives the answer away"),
])
def test_bad_items_are_dropped_with_the_reason(text: str, reason: str) -> None:
    report = bq.BrainQuizReport()
    assert bq.answerable_case(row(), text, report) is None
    assert report.dropped == {reason: 1}


def test_an_unanswerable_question_needs_confirmation() -> None:
    q = reply(question="What is the annual rent of the Oslo warehouse building?")
    report = bq.BrainQuizReport()
    no = bq.unanswerable_case(row(), q, [row()], lambda _p: reply(answered=False), report)
    assert no == {"question": "What is the annual rent of the Oslo warehouse building?", "expect_abstain": True, "category": "unanswerable"}
    assert bq.unanswerable_case(row(), q, [row()], lambda _p: reply(answered=True), report) is None
    assert bq.unanswerable_case(row(), q, [row()], lambda _p: "unclear", report) is None    # no clear "no" = not kept
    assert report.dropped == {"unanswerable: the documents may answer it": 2}


def test_passages_are_spread_over_documents_and_short_ones_skipped() -> None:
    rows = [row("big", PASSAGE + str(i)) for i in range(50)] + [row("small")] + [row("tiny", "too short")]
    picked = bq.spread_passages(rows, 4)
    assert len(picked) == 4 and {r["document_id"] for r in picked} == {"big", "small"}
    assert sum(r["document_id"] == "small" for r in picked) == 1


def test_a_failing_helper_call_loses_that_item_only() -> None:
    calls = {"n": 0}

    def flaky(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("provider down")
        return good()

    cases, report = bq.build_cases([row(str(i)) for i in range(6)], flaky, questions=6)
    assert report.helper_errors == 1 and any(not c.get("expect_abstain") for c in cases)


def _index(pid: str, rows: list[dict[str, str]]) -> None:
    import pandas as pd

    from finetune_studio.data.fs.paths import brain_index_dir

    d = brain_index_dir(pid)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "chunks.parquet")


def test_the_api_saves_a_runnable_suite(client, monkeypatch) -> None:
    from finetune_studio.data.prep import preference
    from finetune_studio.testing.suite import load_test_suite

    pid = client.post("/api/projects", json={"name": "quiz"}).json()["id"]
    assert client.post(f"/api/projects/{pid}/testing/brain-quiz", json={"questions": 5}).json()["code"] == "no_index"
    _index(pid, [row(str(i)) for i in range(8)])
    monkeypatch.setattr(preference, "helper_generate", lambda **_k: lambda _p: good())
    done = client.post(f"/api/projects/{pid}/testing/brain-quiz", json={"questions": 5}).json()
    assert done["case_count"] >= 1 and Path(done["path"]).is_file()
    assert load_test_suite(done["path"])[0].correct_answer == "412"
    assert client.get(f"/api/projects/{pid}/testing/brain-quiz/progress").json()["running"] is False
    assert client.post(f"/api/projects/{pid}/testing/brain-quiz", json={"questions": 1}).status_code == 422


def test_the_page_can_write_a_quiz() -> None:
    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/project_testing.html").read_text()
    assert 'id="t-bq-btn"' in html and "/testing/brain-quiz" in html and "addEventListener('click', buildBrainQuiz)" in html


def test_the_unanswerable_check_uses_its_own_deterministic_verifier() -> None:
    """The "is it answered after all?" call goes to ``verify`` (temperature 0 in the app), never the writer: at the
    writer's temperature the same passages flipped between yes and no (beta walkthrough 2026-11-12)."""
    rows = [row(str(i)) for i in range(6)]
    writer_saw, verifier_saw = [], []

    def writer(prompt: str) -> str:
        writer_saw.append(prompt)
        if prompt.startswith("You write one question that a company's documents do NOT answer"):
            return reply(question="What is the annual rent of the Oslo warehouse building?")
        return good()

    def verifier(prompt: str) -> str:
        verifier_saw.append(prompt)
        return reply(answered=False)

    cases, _ = bq.build_cases(rows, writer, questions=6, verify=verifier)
    assert verifier_saw and all(p.startswith("Could a careful reader answer") for p in verifier_saw)
    assert not any(p.startswith("Could a careful reader") for p in writer_saw)
    assert any(c.get("expect_abstain") for c in cases)


def test_the_verifier_counts_answers_by_implication() -> None:
    for marker in ("by a clear implication", "a rule for a whole range, plan or group answers it for each member",
                   '"included", "free" or "no charge" answers a question about the cost'):
        assert marker in bq._VERIFY
