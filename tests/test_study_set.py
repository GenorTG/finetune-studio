"""Study set (data/prep/study_set.py): each approved fact several ways, every generated text gated."""
from __future__ import annotations

import json

import pytest

from finetune_studio.data.prep import study_set as study

PAIR = {"id": "p1", "question": "What is the reference number of Amendment No. 2?",
        "answer": "Amendment No. 2 has reference number KV-MSA-2023-017-A2.", "source_id": "f.doc", "chunk_idx": 1,
        "chunk_text": "Amendment No. 2 (ref KV-MSA-2023-017-A2) dated 3 July 2024."}


def _reply(questions=(), answers=(), passage_q="", passage="") -> str:
    return json.dumps({"questions": list(questions), "answers": list(answers),
                       "passage_question": passage_q, "passage": passage})


def test_question_gate_keeps_values_names_and_hides_the_answer() -> None:
    q, a = "What did Korvane pay Oakhaven in 2024?", "Korvane paid Oakhaven EUR 12,500 in 2024."
    assert study.question_gate(q, a, "In 2024, how much was paid by Korvane to Oakhaven?", set()) is None
    assert study.question_gate(q, a, "How much did Korvane pay Oakhaven?", set()) == "question_lost_value"
    assert study.question_gate(q, a, "In 2024, how much was paid to the dairy?", set()) == "question_lost_name"
    assert study.question_gate(q, a, "Did Korvane pay Oakhaven 12,500 in 2024?", set()) == "question_leaks_answer"
    assert study.question_gate(q, a, q, {study.normalize_question(q)}) == "question_duplicate"


def test_answer_gate_keeps_every_value_and_invents_none() -> None:
    a = "The excursion lasted 9 hours on 14 June 2024."
    allowed = study.values(a)
    assert study.answer_gate(a, "On 14 June 2024 the excursion went on for 9 hours.", allowed) is None
    assert study.answer_gate(a, "The excursion lasted 9 hours.", allowed) == "answer_lost_value"
    assert study.answer_gate(a, "On 14 June 2024 it lasted 9 hours at 8 C.", allowed) == "answer_new_value"
    assert study.answer_gate(a, "The excursion lasted 9 hours on 14 June 2024!", allowed) == "answer_same_as_reviewed"


def test_rows_for_a_pair_with_every_kind_and_the_quiz_held_out() -> None:
    reply = _reply(
        questions=["Which reference number does Amendment No. 2 carry?",
                   "What reference number was given to Amendment No. 2?"],
        answers=["The reference number of Amendment No. 2 is KV-MSA-2023-017-A2."],
        passage_q="What do the documents say about Amendment No. 2?",
        passage="Amendment No. 2 was filed under KV-MSA-2023-017-A2. It is dated 3 July 2024.")
    report = study.StudyReport()
    quiz = [(study.normalize_question(q), study._words(q)) for q in ["What reference number was given to Amendment No. 2?"]]
    rows = study.study_rows_for(PAIR, reply, study.StudyOptions(), quiz, report)
    assert [r["study_kind"] for r in rows] == ["reworded", "restated", "passage"]
    assert report.dropped == {"near_heldout_quiz": 1}
    assert rows[1]["conversations"][0]["value"] == PAIR["question"]
    assert "3 July 2024" in rows[2]["conversations"][1]["value"]  # from the source passage: allowed


def test_build_keeps_originals_survives_failed_calls_and_reports() -> None:
    other = {**PAIR, "id": "p2", "question": "When is Amendment No. 2 dated?", "chunk_idx": 2,
             "answer": "Amendment No. 2 is dated 3 July 2024."}
    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        if "When is Amendment" in prompt:
            raise RuntimeError("helper down")
        return _reply(questions=["Under which reference number is Amendment No. 2 filed?"])

    rows, report = study.build_study_rows([PAIR, other, dict(PAIR)], generate,
                                          study.StudyOptions(reworded=1, restated=0, passage=False), workers=2)
    assert len(calls) == 2  # the duplicate pair is asked once
    assert [r["study_kind"] for r in rows] == ["original", "reworded", "original"]
    assert report.calls_failed == 1 and report.pairs_without_new_row == 1 and report.kept["reworded"] == 1


def test_unparseable_reply_is_counted_not_fatal() -> None:
    rows, report = study.build_study_rows([PAIR], lambda _p: "sorry, no JSON", study.StudyOptions())
    assert [r["study_kind"] for r in rows] == ["original"]
    assert report.dropped == {"unparseable_reply": 1}


def test_options_are_bounded() -> None:
    with pytest.raises(ValueError):
        study.StudyOptions(reworded=6)
    with pytest.raises(ValueError):
        study.StudyOptions(reworded=0, restated=0, passage=False)


def _approve(pid: str, item: dict) -> None:
    from finetune_studio.data.fs import qa as qa_fs

    qa_fs.write_qa_pair(pid, {**item, "status": "approved"})


def test_route_builds_registers_and_holds_out_project_quiz(client, monkeypatch) -> None:
    from finetune_studio.data.prep import preference
    from finetune_studio.testing.suite_import import suites_dir

    pid = client.post("/api/projects", json={"name": "Study"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/data-prep/study", json={})
    assert r.status_code == 409 and r.json()["code"] == "no_approved_pairs"
    _approve(pid, PAIR)
    suites_dir(pid).mkdir(parents=True, exist_ok=True)
    (suites_dir(pid) / "quiz.json").write_text(json.dumps(
        [{"name": "q1", "question": "Which reference number does Amendment No. 2 carry?", "correct_answer": "x"}]))
    reply = _reply(questions=["Which reference number does Amendment No. 2 carry?",
                              "Under which reference number is Amendment No. 2 filed?"])
    monkeypatch.setattr(preference, "helper_generate", lambda **_: (lambda _p: reply))
    r = client.post(f"/api/projects/{pid}/data-prep/study", json={"reworded": 2, "restated": 0, "passage": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["rows"] == 2 and body["kept"]["reworded"] == 1
    assert body["dropped"] == {"near_heldout_quiz": 1}
    assert body["train_url"].endswith(f"dataset={body['dataset']['id']}")
    assert client.get(f"/api/projects/{pid}/data-prep/study/progress").json()["running"] is False
    assert client.post(f"/api/projects/{pid}/data-prep/study", json={"reworded": 9}).status_code == 422


def test_route_reports_no_helper(client, monkeypatch) -> None:
    from finetune_studio.data.prep import preference

    pid = client.post("/api/projects", json={"name": "Study2"}).json()["id"]
    _approve(pid, PAIR)
    monkeypatch.setattr(preference, "helper_generate", lambda **_: None)
    r = client.post(f"/api/projects/{pid}/data-prep/study", json={})
    assert r.status_code == 409 and r.json()["code"] == "no_model"


def test_data_prep_page_has_the_study_card(client) -> None:
    pid = client.post("/api/projects", json={"name": "Study3"}).json()["id"]
    page = client.get(f"/projects/{pid}/data-prep").text
    for el in ("dp-study-card", "dp-study-build", "dp-study-reworded", "dp-study-progress"):
        assert f'id="{el}"' in page


def test_cli_build_study_registers_the_dataset(monkeypatch, capsys) -> None:
    import sys
    from contextlib import contextmanager

    from finetune_studio import db
    from finetune_studio.cli import _registry
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.data.prep import preference

    db.init_db()
    proj = db.create_project("study-cli", base_model="x/test")
    qa_fs.write_qa_pair(proj["id"], {**PAIR, "status": "approved"})

    @contextmanager
    def loaded():
        yield None

    reply = _reply(questions=["Under which reference number is Amendment No. 2 filed?"])
    monkeypatch.setattr(preference, "helper_loaded_for_cli", loaded)
    monkeypatch.setattr(preference, "helper_generate", lambda **_: (lambda _p: reply))
    monkeypatch.setattr(sys, "argv", ["fts", "dataset", "build-study", "--project", "study-cli",
                                      "--reworded", "1", "--restated", "0", "--no-passage", "--json"])
    try:
        _registry.main()
    except SystemExit as exc:
        assert not exc.code
    out = json.loads(capsys.readouterr().out)
    assert out["dataset"]["rows"] == 2 and out["kept"]["reworded"] == 1
