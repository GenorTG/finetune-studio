"""Reworded held-out quiz (testing/reworded_quiz.py): the trainer's held-out facts, each question reworded once."""
from __future__ import annotations

import json
import re

import pytest

from finetune_studio.benchmarks.suite_defs import format_auto_suite_label
from finetune_studio.smart_brain.prompt import CONTEXT_MARKER
from finetune_studio.testing import reworded_quiz as rq
from finetune_studio.training.data import sft_validation_rows


def _row(i: int, question: str, kind: str = "original", system: str = "") -> dict:
    conv = [{"from": "human", "value": question}, {"from": "gpt", "value": f"Item {i} has code KV-{i:03d}."}]
    if system:
        conv.insert(0, {"from": "system", "value": system})
    return {"conversations": conv, "pair_id": f"p{i}", "study_kind": kind, "source_id": "f.doc", "chunk_idx": i}


def _dataset(n: int = 40) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        rows.append(_row(i, f"What is the code of item {i}?"))
        rows.append(_row(i, f"Item {i} carries which code?", kind="reworded"))
    return rows


def _good(prompt: str) -> str:
    i = re.search(r"QUESTION: .*?item (\d+)", prompt, re.IGNORECASE).group(1)
    return json.dumps({"question": f"According to the records, which code was assigned to item {i}?"})


def _heldout_ids(rows: list[dict]) -> set[str]:
    return {r["pair_id"] for r in sft_validation_rows(rows)}


def test_one_fact_per_heldout_group_original_row_first_grounded_rows_skipped() -> None:
    rows = _dataset()
    held = _heldout_ids(rows)
    assert held, "the 10 % split must hold out at least one fact"
    grounded_pid = min(held)
    i = int(grounded_pid[1:])
    rows = [_row(i, r["conversations"][0]["value"], r["study_kind"], system=f"Answer.\n\n{CONTEXT_MARKER}Item text")
            if r["pair_id"] == grounded_pid else r for r in rows]
    report = rq.RewordReport()
    facts = rq.heldout_facts(rows, report)
    assert {f.key for f in facts} == held - {grounded_pid}
    assert all(f.question.startswith("What is the code") for f in facts)   # the original row, not its rewording
    assert report.skipped_grounded == 2 and report.facts == len(held) - 1 and report.heldout_rows == 2 * len(held)


@pytest.mark.parametrize("reworded,reason", [
    ("Which code was assigned to the item?", "question_lost_value"),
    ("Is KV-007 the code assigned to item 7?", "question_leaks_answer"),
    ("Item 7 carries which code?", "question_duplicate"),                  # a question the dataset already has
    ("What is the code of item 7, then?", "question_same_wording"),
])
def test_reworded_question_gates(reworded: str, reason: str) -> None:
    fact = rq.HeldoutFact(key="p7", question="What is the code of item 7?", answer="Item 7 has code KV-007.")
    seen = {rq.normalize_question("What is the code of item 7?"), rq.normalize_question("Item 7 carries which code?")}
    assert rq.reworded_question(fact, json.dumps({"question": reworded}), seen) == (None, reason)
    assert rq.reworded_question(fact, "no json here", seen) == (None, "unparseable_reply")
    ok = "According to the records, which code was assigned to item 7?"
    assert rq.reworded_question(fact, f"<think>x</think>{json.dumps({'question': ok})}", seen) == (ok, None)


def test_build_cases_one_call_per_fact_counts_drops_and_failed_calls() -> None:
    rows = _dataset()
    held = sorted(_heldout_ids(rows), key=lambda p: int(p[1:]))
    leak, fail = held[0], held[1]
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        i = re.search(r"QUESTION: .*?item (\d+)", prompt).group(1)
        if f"p{i}" == fail:
            raise RuntimeError("helper down")
        if f"p{i}" == leak:
            return json.dumps({"question": f"Is KV-{int(i):03d} the code of item {i}?"})
        return _good(prompt)

    cases, report = rq.build_reworded_cases(rows, generate)
    assert len(calls) == len(held)                       # once per fact, never retried
    assert report.kept == len(held) - 2 == len(cases)
    assert report.dropped == {"question_leaks_answer": 1} and report.calls_failed == 1
    first = cases[0]
    i = int(re.search(r"item (\d+)", first["question"]).group(1))
    assert first["category"] == "heldout_reworded" and first["correct_answer"] == f"Item {i} has code KV-{i:03d}."
    assert all("ANSWER (never put any of it into the question)" in p for p in calls)


def test_build_cases_stops_after_consecutive_failures_and_needs_heldout_rows() -> None:
    def broken(_prompt: str) -> str:
        raise RuntimeError("helper down")

    with pytest.raises(rq.NoUsableQuestions) as exc:
        rq.build_reworded_cases(_dataset(120), broken)
    assert exc.value.report.calls_failed == 5
    with pytest.raises(rq.NoHeldoutRows):
        rq.build_reworded_cases([_row(1, "What is the code of item 1?")], _good)   # one fact: nothing held out


def _register(pid: str, rows: list[dict], name: str = "golden set") -> dict:
    from finetune_studio.data.prep.dataset_build import register_dataset_file

    body = "".join(json.dumps(r) + "\n" for r in rows)
    return register_dataset_file(pid, f"{pid}-golden.jsonl", body, name, source="test").dataset


def test_route_saves_a_labelled_suite_that_study_sets_avoid(client, monkeypatch) -> None:
    from finetune_studio.data.prep import preference
    from finetune_studio.data.prep.study_set import heldout_questions
    from finetune_studio.testing.suite import load_test_suite

    pid = client.post("/api/projects", json={"name": "Reworded"}).json()["id"]
    url = f"/api/projects/{pid}/testing/reworded-quiz"
    r = client.post(url, json={})
    assert r.status_code == 409 and r.json()["code"] == "no_dataset"
    ds = _register(pid, _dataset())
    assert client.post(url, json={"dataset_id": "nope"}).status_code == 404
    monkeypatch.setattr(preference, "helper_generate", lambda **_: _good)
    r = client.post(url, json={"dataset_id": ds["id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    n = len(_heldout_ids(_dataset()))
    assert body["case_count"] == body["kept"] == n and body["dataset"]["id"] == ds["id"]
    assert body["name"] == "heldout-reworded-golden-set"
    assert body["label"] == f"held-out facts, new wording · golden-set ({n} cases)"
    cases = load_test_suite(body["path"])
    assert len(cases) == n and all(c.question.startswith("According to the records") for c in cases)
    assert set(heldout_questions(pid)) == {c.question for c in cases}      # future study sets stay away from it
    from finetune_studio.webui.routes.benchmarks import _discover_suites

    listed = _discover_suites(pid)
    mine = [s for s in listed if s.get("path") == body["path"]]
    assert len(mine) == 1 and mine[0]["source"] == "project_heldout_reworded" and mine[0]["label"] == body["label"]
    assert mine[0]["scoring"] == "judge"
    assert client.get(f"{url}/progress").json()["running"] is False
    assert client.post(url, json={"max_questions": 0}).status_code == 422
    assert client.post("/api/projects/nope/testing/reworded-quiz", json={}).status_code == 404


def test_route_without_a_helper_is_409_with_the_reason(client, monkeypatch) -> None:
    from finetune_studio.data.prep import generator, preference

    pid = client.post("/api/projects", json={"name": "Reworded2"}).json()["id"]
    _register(pid, _dataset())
    monkeypatch.setattr(preference, "helper_generate", lambda **_: None)
    monkeypatch.setattr(generator, "helper_resolution_error", lambda *_a, **_k: "Load the helper on the Pairs page first.")
    r = client.post(f"/api/projects/{pid}/testing/reworded-quiz", json={})
    assert r.status_code == 409
    assert r.json() == {"detail": "Load the helper on the Pairs page first.", "code": "no_model"}


def test_route_reports_when_no_rewording_passes(client, monkeypatch) -> None:
    from finetune_studio.data.prep import preference

    pid = client.post("/api/projects", json={"name": "Reworded3"}).json()["id"]
    _register(pid, _dataset())
    monkeypatch.setattr(preference, "helper_generate", lambda **_: (lambda _p: json.dumps({"question": "Which code was it, then?"})))
    r = client.post(f"/api/projects/{pid}/testing/reworded-quiz", json={})
    assert r.status_code == 422 and r.json()["code"] == "no_usable_questions"
    assert r.json()["dropped"] == {"question_lost_value": len(_heldout_ids(_dataset()))}


def test_testing_page_labels_the_reworded_quiz_and_the_auto_suite_stays_memorisation(client) -> None:
    pid = client.post("/api/projects", json={"name": "Reworded4"}).json()["id"]
    page = client.get(f"/projects/{pid}/testing").text
    for el in ("t-reworded-box", "t-rq-ds", "t-rq-btn", "t-rq-msg"):
        assert f'id="{el}"' in page
    assert "Build a reworded quiz · held-out facts, new wording" in page
    assert "memorisation" in format_auto_suite_label("golden", 10)
