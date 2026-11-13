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


def test_unanswerable_verifier_searches_top_ten_hybrid_hits_across_documents() -> None:
    question = "What does the CF-7 monthly plan cost with consumables?"
    q = reply(question=question)
    hits = [
        {"filename": "price-list.xlsx", "text": "CF-7 monthly plan: £24.50."},
        {"filename": "service-terms.docx", "text": "Consumables can be added to any plan."},
    ]
    seen: list[str] = []

    def verify(prompt: str) -> str:
        seen.append(prompt)
        return reply(answered=True)

    search_calls: list[tuple[str, int]] = []

    def hybrid_search(query: str, top_k: int) -> list[dict]:
        search_calls.append((query, top_k))
        return hits

    report = bq.BrainQuizReport()
    case = bq.unanswerable_case(row(), q, [row()], verify, report, hybrid_search)

    assert case is None
    assert search_calls == [(question, 10)]
    assert "price-list.xlsx" in seen[0] and "service-terms.docx" in seen[0]
    assert "£24.50" in seen[0] and "added to any plan" in seen[0]
    assert report.dropped == {"unanswerable: the documents may answer it": 1}


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

    from finetune_studio.smart_brain.store import BrainIndex

    class Query:
        def search(self, _question: str, *, top_k: int, hybrid: bool) -> list[dict]:
            assert top_k == 10 and hybrid is True
            return []

    monkeypatch.setattr(BrainIndex, "load", lambda _self: Query())
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


def test_the_unanswerable_check_also_reads_the_passages_sharing_its_words() -> None:
    # Walkthrough 2026-11-13 M7: "How often are the spent coffee grounds collected from a site?" was answered by two
    # files (the plan sheet: collection included; the scheme page: weekly), hybrid search's top hits held neither
    # whole, so it stayed "unanswerable" and the judge failed the model's correct answer.
    question = "How often are the spent coffee grounds collected from a site?"
    rows = [row("plans", "Service plans comparison. Every plan includes the Grounds Again collection of spent coffee "
                         "grounds from each site at no extra charge."),
            row("scheme", "Grounds Again scheme: our van collects the spent coffee grounds weekly, every Thursday, "
                          "and turns them into compost."),
            row("other", "Opening hours of the Leeds office are 9 to 5.")]
    seen: list[str] = []

    def verify(prompt: str) -> str:
        seen.append(prompt)
        return reply(answered=True)

    hits = [{"filename": "other.txt", "text": rows[2]["text"]}]
    case = bq.unanswerable_case(row(), reply(question=question), rows, verify, bq.BrainQuizReport(),
                                lambda q, k: hits)
    assert case is None
    assert "weekly, every Thursday" in seen[0] and "at no extra charge" in seen[0]
    assert seen[0].count("Opening hours of the Leeds office") == 1                      # each passage once
    assert "spread over two or more passages" in bq._VERIFY


def test_a_reviewer_can_mark_an_unanswerable_question_answerable(client, tmp_path: Path) -> None:
    from finetune_studio import db
    from finetune_studio.testing import run_store

    pid = client.post("/api/projects", json={"name": "agent-test-answerable"}).json()["id"]
    suite = tmp_path / "smart-brain-quiz.json"
    question = "How often are the spent coffee grounds collected from a site?"
    suite.write_text(json.dumps([{"name": "q0037", "question": question, "expect_abstain": True,
                                  "category": "unanswerable"}]))
    owner = run_store.resolve_owner_run(pid, "api:x")
    bench = run_store.start_run(owner, "Smart Brain quiz", model_path="api:x", kind="brain",
                                config={"suite_path": str(suite)}, total=1)
    cid = db.create_case(bench["id"], owner, "q0037", "unanswerable", question, "", "Weekly, every Thursday.", [],
                         expect_abstain=True)
    r = client.post(f"/api/testing/projects/{pid}/runs/{bench['id']}/cases/{cid}/answerable", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["suite_updated"] is True and body["case"]["expect_abstain"] is False
    assert body["case"]["correct_answer"] == "Weekly, every Thursday." and body["case"]["verdict"] == "pass"
    saved = json.loads(suite.read_text())[0]
    assert "expect_abstain" not in saved and saved["correct_answer"] == "Weekly, every Thursday."
    again = client.post(f"/api/testing/projects/{pid}/runs/{bench['id']}/cases/{cid}/answerable", json={})
    assert again.status_code == 409
    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/project_testing.html").read_text()
    assert 'id="rv-answerable"' in html and "/answerable'" in html
