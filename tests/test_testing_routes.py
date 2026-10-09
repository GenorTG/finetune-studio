"""/api/testing run + judge routes: 202 flows, project scoping, human verdicts, export, settings, model resolution.

The contract under test (Genor's run-then-judge design): ``run-suite`` / ``run-rag-suite`` / ``evaluate-training``
answer 202 + ``benchmark_id`` and run in the background; the saved run (raw transcripts, no verdict) is read from
``GET .../runs/{bid}`` and ``.../cases``; judging is its own job (``POST .../judge`` or the ``auto_judge`` option);
a person can set or retract any verdict. Only the inference engine and the judge are faked (tests/testing_run_support.py).
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.benchmarks.suite_defs import list_builtin_smoke_suites
from finetune_studio.db import judgements as jdb
from finetune_studio.webui import testing_jobs
from tests.test_rag_suite import FakeRag
from tests.testing_run_support import (
    FakeEngine,
    add_api_provider,
    install_engine,
    install_judge,
    saved_run,
    wait_run_finished,
    wait_until,
)

API = "/api/testing"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Fresh job registry + engine lock, a private settings file, and a model load that records itself."""
    monkeypatch.setattr(testing_jobs, "_ACTIVE", {})
    monkeypatch.setattr(testing_jobs, "ENGINE_LOCK", asyncio.Lock())
    monkeypatch.setattr("finetune_studio.webui.routes.settings.SETTINGS_PATH", tmp_path / "settings.json")
    loads: list[str] = []
    monkeypatch.setattr(testing_jobs, "_load_model", loads.append)
    monkeypatch.setattr("finetune_studio.webui.routes.testing.local_model_missing", lambda _p: False)  # fake paths below
    return loads


def _project(client: TestClient, name: str = "routes") -> str:
    r = client.post("/api/projects", json={"name": name, "base_model": "x/test"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _suite(tmp_path: Path, n: int = 3, name: str = "suite.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps([
        {"name": f"c{i}", "category": "geo", "question": f"Q{i}", "correct_answer": f"key{i}", "keywords": [f"key{i}"]}
        for i in range(n)
    ]), encoding="utf-8")
    return str(path)


def _merged_export(pid: str, tmp_path: Path, name: str = "out") -> str:
    merged = tmp_path / name / "merged"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_bytes(b"x" * 32)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    run = db.create_run(pid, "train", base_model="x/test")
    db.update_run(run["id"], status="done", output_path=str(tmp_path / name))
    return str(merged)


def _start_run(client: TestClient, pid: str, suite: str, **body: Any) -> dict[str, Any]:
    r = client.post(f"{API}/run-suite", json={"suite_path": suite, "project_id": pid, "model_path": "/m/model",
                                              "max_tokens": 16, **body})
    assert r.status_code == 202, r.text
    return r.json()


def _finished_run(client: TestClient, pid: str, tmp_path: Path, n: int = 3, **body: Any) -> str:
    bid = _start_run(client, pid, _suite(tmp_path, n), **body)["benchmark_id"]
    wait_run_finished(bid, judge=bool(body.get("auto_judge")))
    return bid


def _runs(pid: str, bid: str) -> str:
    return f"{API}/projects/{pid}/runs/{bid}"


# ── 202 flows ────────────────────────────────────────────────────────────────


def test_run_suite_answers_202_at_once_and_the_run_is_read_back_from_the_runs_routes(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    gate = threading.Event()
    install_engine(monkeypatch, FakeEngine(on_generate=lambda _q: gate.wait(10)))
    try:
        started = _start_run(client, pid, _suite(tmp_path, 2))
        bid = started["benchmark_id"]
        assert started["ok"] is True and started["benchmark"]["id"] == bid
        assert started["benchmark"]["status"] == "running" and started["benchmark"]["progress_total"] == 2
        assert started["benchmark"]["scoring"] == "judge" and started["benchmark"]["model_path"] == "/m/model"
        live = client.get(_runs(pid, bid)).json()  # the page can read the live state while the model is answering
        assert live["status"] == "running" and live["active"] == ["run"] and live["judge_status"] == ""
        jobs = client.get(f"{API}/status").json()["jobs"]  # also what the live stream pushes to the page
        assert [(j["benchmark_id"], j["kind"], j["gpu"], j["stopping"], j["status"], j["progress_total"]) for j in jobs] == [
            (bid, "run", True, False, "running", 2)]
    finally:
        gate.set()
    done = wait_run_finished(bid)
    assert done["status"] == "done"

    detail = client.get(_runs(pid, bid)).json()
    assert detail["status"] == "done" and detail["progress_done"] == detail["progress_total"] == 2
    assert detail["active"] == [] and detail["agreement"] == {"human_reviewed": 0, "judges": {}}
    assert detail["scores"]["awaiting"] == 2 and detail["scores"]["pass_rate"] is None  # no verdict yet
    assert detail["config"]["suite_path"].endswith("suite.json") and detail["config"]["max_tokens"] == 16

    listed = client.get(f"{API}/projects/{pid}/runs").json()
    assert [r["id"] for r in listed] == [bid] and listed[0]["suite_name"] == "suite.json"

    cases = client.get(_runs(pid, bid) + "/cases").json()
    assert [c["question"] for c in cases] == ["Q0", "Q1"]
    assert all("transcript" not in c and c["verdict"] == "" and c["judgements"] == [] for c in cases)
    assert [c["model_answer"] for c in cases] == ["answer to Q0", "answer to Q1"]
    full = client.get(_runs(pid, bid) + f"/cases/{cases[0]['id']}").json()
    assert full["transcript"][-1] == {"role": "assistant", "content": "answer to Q0"}


def test_run_suite_with_a_bad_request_never_starts_a_job(client: TestClient, tmp_path: Path) -> None:
    pid = _project(client)
    suite = _suite(tmp_path)
    post = client.post
    assert post(f"{API}/run-suite", json={"project_id": pid}).status_code == 400
    assert post(f"{API}/run-suite", json={"suite_path": suite}).status_code == 400
    assert post(f"{API}/run-suite", json={"suite_path": suite, "project_id": "nope"}).status_code == 404
    assert post(f"{API}/run-suite", json={"suite_path": str(tmp_path / "missing.json"), "project_id": pid}).status_code == 404
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert post(f"{API}/run-suite", json={"suite_path": str(empty), "project_id": pid}).status_code == 400
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert post(f"{API}/run-suite", json={"suite_path": str(broken), "project_id": pid}).status_code == 400
    for body in ({"max_tokens": "lots"}, {"temperature": "hot"}, {"auto_judge": "yes"}, {"auto_judge": 1}):
        r = post(f"{API}/run-suite", json={"suite_path": suite, "project_id": pid, "model_path": "/m/model", **body})
        assert r.status_code == 400, (body, r.text)
    assert client.get(f"{API}/projects/{pid}/runs").json() == [] and testing_jobs.active_jobs() == []


def test_run_rag_suite_answers_202_and_saves_retrieval_with_each_case(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    hit = {"rank": 1, "document_id": "doc-1", "source": "a.md", "filename": "a.md", "chunk_index": 0,
           "chunk_id": "c0", "score": 1.0, "text": "key0 is the answer."}
    rag = FakeRag({"Q0": [hit]})
    monkeypatch.setattr("finetune_studio.testing.rag_suite.load_portable_rag_query", lambda _p: rag)
    engine = install_engine(monkeypatch, FakeEngine(lambda _q: "key0"))
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    r = client.post(f"{API}/run-rag-suite", json={
        "suite_path": _suite(tmp_path, 1), "project_id": pid, "model_path": "/m/model",
        "corpus_path": str(corpus), "top_k": 3, "max_tokens": 16})
    assert r.status_code == 202, r.text
    bid = r.json()["benchmark_id"]
    assert r.json()["benchmark"]["kind"] == "rag" and r.json()["benchmark"]["suite_name"].startswith("rag · ")
    done = wait_run_finished(bid)
    assert done["status"] == "done" and rag.search_calls == [("Q0", 3)]
    assert done["scores"]["corpus_path"] == str(corpus) and "retrieval" in done["scores"]
    case = client.get(_runs(pid, bid) + "/cases").json()[0]
    assert case["verdict"] == "" and case["judge_input"]["chunks_retrieved"] == 1
    assert "key0 is the answer." in engine.asked[0]  # the model was shown the retrieved chunk
    assert client.get(_runs(pid, bid) + f"/cases/{case['id']}").json()["judge_input"]["context_text"]


def test_run_rag_suite_validates_before_it_starts(client: TestClient, tmp_path: Path) -> None:
    pid = _project(client)
    suite = _suite(tmp_path, 1)
    post = client.post
    assert post(f"{API}/run-rag-suite", json={"project_id": pid}).status_code == 400
    assert post(f"{API}/run-rag-suite", json={"suite_path": suite}).status_code == 400
    assert post(f"{API}/run-rag-suite", json={"suite_path": suite, "project_id": "nope"}).status_code == 404
    r = post(f"{API}/run-rag-suite", json={"suite_path": suite, "project_id": pid, "top_k": "many"})
    assert r.status_code == 400 and "integers" in r.json()["error"]
    r = post(f"{API}/run-rag-suite", json={"suite_path": suite, "project_id": pid, "model_path": "/m/model",
                                           "auto_judge": "yes"})
    assert r.status_code == 400
    assert client.get(f"{API}/projects/{pid}/runs").json() == []


def test_evaluate_training_answers_202_for_both_eval_kinds(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.db import datasets as datasets_db

    pid = _project(client)
    rows = tmp_path / "approved.jsonl"
    rows.write_text("".join(json.dumps({"conversations": [{"from": "human", "value": f"Q{i}?"},
                                                           {"from": "gpt", "value": f"A{i}"}]}) + "\n"
                            for i in range(20)), encoding="utf-8")
    ds = datasets_db.create_dataset(pid, "approved", str(rows), source="data-prep-export", qa_count=20)
    install_engine(monkeypatch, FakeEngine(lambda _q: "a"))

    leak = client.post(f"{API}/evaluate-training", json={"project_id": pid, "dataset_id": ds["id"], "max_cases": 5,
                                                         "model_path": "/m/model"})
    assert leak.status_code == 202, leak.text
    assert leak.json()["benchmark"]["kind"] == "training_leakage" and leak.json()["benchmark"]["progress_total"] == 5
    done = wait_run_finished(leak.json()["benchmark_id"])
    assert done["scores"]["eval_kind"] == "training_leakage" and done["scores"]["total"] == 5

    held = client.post(f"{API}/evaluate-training", json={"project_id": pid, "dataset_id": ds["id"],
                                                         "eval_kind": "heldout", "model_path": "/m/model"})
    assert held.status_code == 202, held.text
    assert held.json()["benchmark"]["kind"] == "heldout"
    assert wait_run_finished(held.json()["benchmark_id"])["scores"]["eval_kind"] == "heldout"
    assert client.post(f"{API}/evaluate-training", json={"project_id": pid, "dataset_id": "nope",
                                                         "model_path": "/m/model"}).status_code in (400, 404)


# ── auto-judge and the judge route ───────────────────────────────────────────


def test_run_with_auto_judge_in_the_body_judges_the_saved_cases_by_itself(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    provider = add_api_provider()
    install_engine(monkeypatch)
    install_judge(monkeypatch, {"Q1": "fail"}, concurrent=True)

    bid = _finished_run(client, pid, tmp_path, auto_judge=True, judge_provider_id=provider)
    detail = client.get(_runs(pid, bid)).json()
    assert detail["judge_status"] == "done" and detail["judge_provider_id"] == provider
    assert detail["judge_model"] == "fake-judge" and detail["judge_done"] == detail["judge_total"] == 3
    cases = client.get(_runs(pid, bid) + "/cases").json()
    assert [(c["verdict"], c["judge"]) for c in cases] == [("pass", "ai"), ("fail", "ai"), ("pass", "ai")]
    assert [j["verdict"] for c in cases for j in c["judgements"]] == ["pass", "fail", "pass"]
    assert detail["scores"]["pass_rate"] == pytest.approx(66.7) and detail["scores"]["by_judge"] == {"ai": 3}


def test_the_saved_auto_judge_setting_applies_to_runs_that_do_not_say_otherwise(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    provider = add_api_provider()
    install_engine(monkeypatch)
    asked = install_judge(monkeypatch, concurrent=True)
    assert client.get("/api/settings/testing").json()["auto_judge"] is False

    off = _finished_run(client, pid, tmp_path, n=2)  # default off: the run ends at the raw transcripts
    assert client.get(_runs(pid, off)).json()["judge_status"] == "" and asked == []

    saved = client.put("/api/settings/testing", json={"auto_judge": True, "judge_provider_id": provider})
    assert saved.status_code == 200 and saved.json()["auto_judge"] is True
    on = _finished_run(client, pid, tmp_path, n=2, auto_judge=None)
    row = client.get(_runs(pid, on)).json()
    assert row["judge_status"] == "done" and row["judge_provider_id"] == provider and sorted(asked) == ["Q0", "Q1"]

    asked.clear()
    opted_out = _finished_run(client, pid, tmp_path, n=2, auto_judge=False)
    assert client.get(_runs(pid, opted_out)).json()["judge_status"] == "" and asked == []


def test_judge_route_starts_a_separate_job_and_a_rejudge_keeps_the_earlier_opinion(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    provider = add_api_provider()
    install_engine(monkeypatch)
    install_judge(monkeypatch, {"Q0": "fail"}, concurrent=True, model="judge-a")
    bid = _finished_run(client, pid, tmp_path, n=2)

    r = client.post(_runs(pid, bid) + "/judge", json={"provider_id": provider})
    assert r.status_code == 202, r.text
    assert r.json()["ok"] is True and r.json()["benchmark"]["judge_status"] == "running"
    wait_run_finished(bid, judge=True)
    first = client.get(_runs(pid, bid) + "/cases").json()
    assert [c["verdict"] for c in first] == ["fail", "pass"]

    install_judge(monkeypatch, {"Q0": "pass"}, concurrent=True, model="judge-b")
    assert client.post(_runs(pid, bid) + "/judge", json={"provider_id": provider, "only_unjudged": False}).status_code == 202
    wait_until(lambda: db.get_benchmark(bid)["judge_model"] == "judge-b"
               and db.get_benchmark(bid)["judge_status"] == "done", what="the second judge")
    again = client.get(_runs(pid, bid) + "/cases").json()
    assert [c["verdict"] for c in again] == ["pass", "pass"]
    assert [j["judge_model"] for j in again[0]["judgements"]] == ["judge-a", "judge-b"]  # both opinions are kept


def test_judge_route_refuses_what_cannot_be_judged(client: TestClient, tmp_path: Path) -> None:
    pid = _project(client)
    rid = db.create_run(pid, "train")["id"]
    running = saved_run(rid, [("Q?", "k", "a")], status="running")
    exact = saved_run(rid, [("Q?", "k", "a")], scoring="exact")
    empty = saved_run(rid, [])
    ok = saved_run(rid, [("Q?", "k", "a")])
    judge = lambda bid, **body: client.post(_runs(pid, bid) + "/judge", json=body)
    r = judge(running)
    assert r.status_code == 400 and "still running" in r.json()["error"]
    r = judge(exact)
    assert r.status_code == 400 and "exact-match" in r.json()["error"]
    r = judge(empty)
    assert r.status_code == 400 and "no saved cases" in r.json()["error"]
    r = judge(ok, provider_id="no-such-provider")
    assert r.status_code == 400 and "unknown provider" in r.json()["error"]
    assert judge(ok, only_unjudged="yes").status_code == 400
    assert client.post(_runs(pid, ok) + "/judge", content="[1]", headers={"content-type": "application/json"}).status_code == 400
    assert client.post(_runs(pid, "nope") + "/judge", json={}).status_code == 404
    assert all(db.get_benchmark(b)["judge_status"] == "" for b in (running, exact, empty, ok))


# ── concurrency, cancel, delete ──────────────────────────────────────────────


def test_a_second_run_while_one_is_active_is_409_and_names_the_active_one(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    rid = db.create_run(pid, "train")["id"]
    other = saved_run(rid, [("Q?", "k", "a")])
    gate = threading.Event()
    install_engine(monkeypatch, FakeEngine(on_generate=lambda _q: gate.wait(10)))
    try:
        bid = _start_run(client, pid, _suite(tmp_path, 2))["benchmark_id"]
        r = client.post(f"{API}/run-suite", json={"suite_path": _suite(tmp_path), "project_id": pid,
                                                  "model_path": "/m/model"})
        assert r.status_code == 409 and r.json()["active"] == {"kind": "run", "benchmark_id": bid}
        assert bid in r.json()["error"]
        local = client.post(_runs(pid, other) + "/judge", json={"provider_id": "local-default"})
        assert local.status_code == 409 and local.json()["active"]["benchmark_id"] == bid
        assert client.delete(_runs(pid, bid)).status_code == 409  # not while a job is writing to it
    finally:
        gate.set()
    wait_run_finished(bid)
    assert len(client.get(f"{API}/projects/{pid}/runs").json()) == 2  # the refused runs left no rows


def test_cancel_route_stops_the_run_and_keeps_the_saved_cases(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    reached = threading.Event()
    release = threading.Event()

    def pause_on_second(question: str) -> None:
        if question == "Q1":
            reached.set()
            release.wait(10)

    install_engine(monkeypatch, FakeEngine(on_generate=pause_on_second))
    bid = _start_run(client, pid, _suite(tmp_path, 4))["benchmark_id"]
    try:
        assert reached.wait(10)
        r = client.post(_runs(pid, bid) + "/cancel")
        assert r.status_code == 200 and r.json() == {"ok": True}
        assert client.get(f"{API}/status").json()["jobs"][0]["stopping"] is True
    finally:
        release.set()
    done = wait_run_finished(bid)
    assert done["status"] == "cancelled" and done["progress_done"] == 2
    assert [c["question"] for c in client.get(_runs(pid, bid) + "/cases").json()] == ["Q0", "Q1"]
    again = client.post(_runs(pid, bid) + "/cancel")
    assert again.status_code == 409 and "nothing is running" in again.json()["error"]


def test_delete_removes_the_run_its_cases_and_its_judgements(client: TestClient, tmp_path: Path) -> None:
    pid = _project(client)
    rid = db.create_run(pid, "train")["id"]
    bid = saved_run(rid, [("Q?", "k", "a")])
    cid = db.list_cases(bid)[0]["id"]
    jdb.add_judgement(cid, kind="ai", verdict="pass", judge_model="m")
    assert client.delete(_runs(pid, bid)).json() == {"ok": True}
    assert db.get_benchmark(bid) is None and db.list_cases(bid) == [] and jdb.list_judgements(benchmark_id=bid) == []
    assert client.get(_runs(pid, bid)).status_code == 404
    assert client.delete(_runs(pid, bid)).status_code == 404


# ── project scoping ──────────────────────────────────────────────────────────


def test_runs_and_cases_of_another_project_are_404_on_every_route(client: TestClient) -> None:
    owner, outsider = _project(client, "owner"), _project(client, "outsider")
    rid = db.create_run(owner, "train")["id"]
    bid = saved_run(rid, [("Q?", "k", "a")])
    cid = db.list_cases(bid)[0]["id"]
    jdb.add_judgement(cid, kind="ai", verdict="pass", judge_model="m")
    base = _runs(outsider, bid)
    for method, url, body in (
        ("get", base, None), ("get", base + "/cases", None), ("get", base + f"/cases/{cid}", None),
        ("get", base + "/export", None), ("get", base + "/export?fmt=jsonl", None),
        ("delete", base, None), ("post", base + "/judge", {}), ("post", base + "/cancel", None),
        ("put", base + f"/cases/{cid}/verdict", {"verdict": "fail", "reasoning": "tamper"}),
    ):
        r = client.request(method, url, json=body)
        assert r.status_code == 404, (method, url, r.status_code)
    assert client.get(f"{API}/projects/nope/runs").status_code == 404
    assert client.get(f"{API}/projects/{outsider}/runs").json() == []
    # nothing leaked and nothing was changed
    assert db.get_benchmark(bid) is not None and db.get_case(cid)["verdict"] == "pass"
    assert client.get(_runs(owner, bid) + "/cases").status_code == 200
    assert [r["id"] for r in client.get(f"{API}/projects/{owner}/runs").json()] == [bid]


def test_a_case_of_another_run_in_the_same_project_is_404(client: TestClient) -> None:
    pid = _project(client)
    rid = db.create_run(pid, "train")["id"]
    first, second = saved_run(rid, [("Q?", "k", "a")]), saved_run(rid, [("Q?", "k", "a")])
    foreign = db.list_cases(second)[0]["id"]
    assert client.get(_runs(pid, first) + f"/cases/{foreign}").status_code == 404
    r = client.put(_runs(pid, first) + f"/cases/{foreign}/verdict", json={"verdict": "pass"})
    assert r.status_code == 404 and db.get_case(foreign)["verdict"] == ""
    assert client.get(_runs(pid, first) + "/cases/nope").status_code == 404


# ── human verdicts ───────────────────────────────────────────────────────────


def test_human_verdict_beats_the_ai_can_be_retracted_and_rescoring_follows(client: TestClient) -> None:
    pid = _project(client)
    rid = db.create_run(pid, "train")["id"]
    bid = saved_run(rid, [("Q1", "k", "a"), ("Q2", "k", "a")])
    first, second = [c["id"] for c in db.list_cases(bid)]
    jdb.add_judgement(first, kind="ai", verdict="fail", reasoning="missed a synonym", judge_model="gemma")
    jdb.add_judgement(second, kind="ai", verdict="pass", judge_model="gemma")
    url = _runs(pid, bid) + f"/cases/{first}/verdict"

    r = client.put(url, json={"verdict": "PASS", "reasoning": "the synonym is fine"})  # case-insensitive
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["scores"]["pass_rate"] == 100.0 and body["scores"]["by_judge"] == {"human": 1, "ai": 1}
    assert body["case"]["verdict"] == "pass" and body["case"]["judge"] == "human"
    assert body["case"]["judge_reasoning"] == "the synonym is fine"
    assert [j["kind"] for j in body["case"]["judgements"]] == ["ai", "human"]
    assert body["agreement"]["human_reviewed"] == 1 and body["agreement"]["judges"]["gemma"]["agreement"] == 0.0
    assert db.get_benchmark(bid)["scores"]["pass_rate"] == 100.0

    r = client.put(url, json={"verdict": ""})  # retraction: the AI opinion applies again
    assert r.status_code == 200 and r.json()["case"]["verdict"] == "fail" and r.json()["case"]["judge"] == "ai"
    assert r.json()["scores"]["pass_rate"] == 50.0 and r.json()["case"]["judgements"][-1]["reasoning"] == "retracted"
    assert db.get_case(first)["verdict"] == "fail"


def test_human_verdict_on_a_case_nobody_else_judged_and_its_retraction(client: TestClient) -> None:
    pid = _project(client)
    rid = db.create_run(pid, "train")["id"]
    bid = saved_run(rid, [("Q1", "k", "a")])
    cid = db.list_cases(bid)[0]["id"]
    url = _runs(pid, bid) + f"/cases/{cid}/verdict"
    r = client.put(url, json={"verdict": "partial"})
    assert r.status_code == 200 and r.json()["scores"]["partial"] == 1 and r.json()["scores"]["pass_rate"] == 0.0
    assert r.json()["case"]["judge_reasoning"] == "human verdict"
    r = client.put(url, json={"verdict": ""})
    assert r.json()["case"]["verdict"] == "" and r.json()["case"]["judge"] == "none"
    assert r.json()["scores"]["awaiting"] == 1 and r.json()["scores"]["pass_rate"] is None


@pytest.mark.parametrize("bad", ["maybe", "passed", 3, True, ["pass"], "pass fail"])
def test_a_bad_verdict_is_400_and_changes_nothing(client: TestClient, bad: Any) -> None:
    pid = _project(client)
    rid = db.create_run(pid, "train")["id"]
    bid = saved_run(rid, [("Q1", "k", "a")])
    cid = db.list_cases(bid)[0]["id"]
    r = client.put(_runs(pid, bid) + f"/cases/{cid}/verdict", json={"verdict": bad})
    assert r.status_code == 400, r.text
    assert db.get_case(cid)["verdict"] == "" and jdb.list_judgements(case_id=cid) == []
    for raw in ("[1]", "not json"):
        r = client.put(_runs(pid, bid) + f"/cases/{cid}/verdict", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 400


# ── export ───────────────────────────────────────────────────────────────────


def _judged_run(client: TestClient) -> tuple[str, str, str]:
    pid = _project(client)
    rid = db.create_run(pid, "train")["id"]
    bid = saved_run(rid, [("What is the capital of France?", "Paris", "It is Paris."), ("Q2?", "k2", "wrong")])
    first, second = [c["id"] for c in db.list_cases(bid)]
    jdb.add_judgement(first, kind="ai", verdict="pass", reasoning="Paris present", provider_id="p", judge_model="gemma")
    jdb.add_judgement(first, kind="human", verdict="pass", reasoning="agreed", judge_model="human")
    jdb.add_judgement(second, kind="ai", verdict="fail", reasoning="wrong", judge_model="gemma")
    return pid, bid, first


def test_export_json_holds_question_key_answer_transcript_and_judgements(client: TestClient) -> None:
    pid, bid, first = _judged_run(client)
    r = client.get(_runs(pid, bid) + "/export")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert f"test-run-{bid}.json" in r.headers["content-disposition"]
    payload = r.json()
    assert payload["run"]["id"] == bid and payload["run"]["suite_name"] == "quiz"
    cases = {c["id"]: c for c in payload["cases"]}
    case = cases[first]
    assert case["question"] == "What is the capital of France?" and case["correct_answer"] == "Paris"
    assert case["model_answer"] == "It is Paris."
    assert case["transcript"] == [{"role": "user", "content": "What is the capital of France?"}]  # full, unlike the list route
    assert [(j["kind"], j["verdict"], j["judge_model"]) for j in case["judgements"]] == [
        ("ai", "pass", "gemma"), ("human", "pass", "human")]
    assert case["verdict"] == "pass" and case["judge"] == "human"
    assert len(payload["cases"]) == 2


def test_export_jsonl_is_one_full_case_per_line(client: TestClient) -> None:
    pid, bid, first = _judged_run(client)
    r = client.get(_runs(pid, bid) + "/export", params={"fmt": "jsonl"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    assert f"test-run-{bid}.jsonl" in r.headers["content-disposition"]
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert len(lines) == 2 and r.text.endswith("\n")
    row = next(x for x in lines if x["id"] == first)
    for key in ("question", "correct_answer", "model_answer", "transcript", "judgements"):
        assert row[key], key
    assert {j["kind"] for j in row["judgements"]} == {"ai", "human"}


def test_export_of_an_unjudged_run_still_carries_the_raw_transcripts(client: TestClient) -> None:
    pid = _project(client)
    rid = db.create_run(pid, "train")["id"]
    bid = saved_run(rid, [("Q?", "k", "a")])
    case = client.get(_runs(pid, bid) + "/export").json()["cases"][0]
    assert case["judgements"] == [] and case["verdict"] == "" and case["transcript"]


# ── settings ─────────────────────────────────────────────────────────────────


def test_testing_settings_validation_and_roundtrip(client: TestClient) -> None:
    body = client.get("/api/settings/testing").json()
    assert body["auto_judge"] is False and body["judge_provider_id"] == "" and body["providers"]
    provider = add_api_provider()
    r = client.put("/api/settings/testing", json={"auto_judge": True, "judge_provider_id": provider})
    assert r.status_code == 200 and (r.json()["auto_judge"], r.json()["judge_provider_id"]) == (True, provider)
    assert provider in {p["id"] for p in r.json()["providers"]}
    assert client.get("/api/settings/testing").json()["effective_judge_provider_id"] == provider
    for bad, fragment in (({"judge_provider_id": "no-such-provider"}, "unknown provider"),
                          ({"auto_judge": "true"}, "auto_judge must be true or false")):
        r = client.put("/api/settings/testing", json=bad)
        assert r.status_code == 400 and fragment in r.json()["detail"]
    after = client.get("/api/settings/testing").json()
    assert (after["auto_judge"], after["judge_provider_id"]) == (True, provider)  # a rejected update saved nothing
    assert client.put("/api/settings/testing", json={"auto_judge": False, "judge_provider_id": ""}).json()["effective_judge_provider_id"]


# ── benchmarks route: exact suites only ──────────────────────────────────────


def test_benchmarks_route_refuses_a_project_quiz_and_still_scores_an_exact_suite(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    out = tmp_path / "trained"
    (out / "merged").mkdir(parents=True)
    (out / "merged" / "config.json").write_text("{}", encoding="utf-8")
    run = db.create_run(pid, "train", base_model="x/test")
    db.update_run(run["id"], status="done", output_path=str(out))

    mcq = list_builtin_smoke_suites()[0]
    from finetune_studio.testing.suite import load_test_suite

    key = {c.question: c.correct_answer for c in load_test_suite(mcq.path)}

    class KeyEngine(FakeEngine):
        def generate(self, messages: list[dict], **_kw: Any) -> str:
            return key[messages[-1]["content"]]

    monkeypatch.setattr("finetune_studio.testing.inference.InferenceEngine", lambda: KeyEngine(model_path=None))

    quiz = client.post(f"/api/benchmarks/projects/{pid}/runs/{run['id']}/run", json={"suite_path": _suite(tmp_path)})
    assert quiz.status_code == 400 and "Testing page" in quiz.json()["error"]
    assert db.list_benchmarks_for_project(pid) == []

    exact = client.post(f"/api/benchmarks/projects/{pid}/runs/{run['id']}/run", json={"suite_path": mcq.path})
    assert exact.status_code == 200, exact.text
    assert exact.json()["scores"]["pass_rate"] == 100.0 and exact.json()["benchmark"]["scoring"] == "exact"
    # an exact-scored run is not a judged one: the judge job refuses it, a person may still override a case
    bid = exact.json()["benchmark_id"]
    refused = client.post(_runs(pid, bid) + "/judge", json={})
    assert refused.status_code == 400 and "exact-match" in refused.json()["error"]
    assert client.get(f"{API}/projects/{pid}/runs").json()[0]["scoring"] == "exact"


def test_evaluate_training_on_the_benchmarks_route_starts_a_test_run_job(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.db import datasets as datasets_db

    pid = _project(client)
    out = tmp_path / "trained"
    (out / "merged").mkdir(parents=True)
    (out / "merged" / "model.safetensors").write_bytes(b"x")
    (out / "merged" / "config.json").write_text("{}", encoding="utf-8")
    run = db.create_run(pid, "train", base_model="x/test")
    db.update_run(run["id"], status="done", output_path=str(out))
    rows = tmp_path / "approved.jsonl"
    rows.write_text("".join(json.dumps({"conversations": [{"from": "human", "value": f"Q{i}?"},
                                                           {"from": "gpt", "value": f"A{i}"}]}) + "\n"
                            for i in range(3)), encoding="utf-8")
    ds = datasets_db.create_dataset(pid, "approved", str(rows), source="data-prep-export", qa_count=3)
    install_engine(monkeypatch, FakeEngine(lambda _q: "a", model_path=None))

    r = client.post(f"/api/benchmarks/projects/{pid}/runs/{run['id']}/evaluate-training", json={"dataset_id": ds["id"]})
    assert r.status_code == 202, r.text
    bid = r.json()["benchmark_id"]
    done = wait_run_finished(bid)
    assert done["status"] == "done" and done["run_id"] == run["id"] and done["scores"]["total"] == 3
    assert done["model_path"].endswith("trained/merged") and done["scores"]["pass_rate"] is None  # judged later


# ── which model a run uses ───────────────────────────────────────────────────


RUN_ROUTES = ("run-suite", "run-rag-suite", "evaluate-training")


def _body(route: str, pid: str, tmp_path: Path, corpus: Path, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"project_id": pid, **extra}
    if route != "evaluate-training":
        body["suite_path"] = _suite(tmp_path, 1)
    if route == "run-rag-suite":
        body["corpus_path"] = str(corpus)
    return body


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus"
    path.mkdir()
    return path


@pytest.mark.parametrize("route", RUN_ROUTES)
def test_without_a_model_or_a_merged_export_a_run_is_400_and_never_uses_the_loaded_model(
    client: TestClient, tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    from finetune_studio.db import datasets as datasets_db

    pid = _project(client)
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"conversations": [{"from": "human", "value": "Q?"}, {"from": "gpt", "value": "A"}]}) + "\n",
                    encoding="utf-8")
    datasets_db.create_dataset(pid, "rows", str(rows), source="data-prep-export", qa_count=1)
    resident = install_engine(monkeypatch, FakeEngine(model_path="/helper/or/other/page/model.gguf"))
    # an adapter-only run is not a model either
    adapter = tmp_path / "adapter-only" / "adapter"
    adapter.mkdir(parents=True)
    run = db.create_run(pid, "adapter-run")
    db.update_run(run["id"], status="done", output_path=str(tmp_path / "adapter-only"))

    r = client.post(f"{API}/{route}", json=_body(route, pid, tmp_path, corpus))
    assert r.status_code == 400, r.text
    assert "no model to test" in r.json()["error"]
    assert resident.asked == [] and resident.load_calls == []
    assert client.get(f"{API}/projects/{pid}/runs").json() == [] and testing_jobs.active_jobs() == []


@pytest.mark.parametrize("route", ["run-suite", "evaluate-training"])
def test_a_run_without_model_path_uses_the_latest_merged_export_of_that_project(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, _isolated: list[str],
) -> None:
    from finetune_studio.db import datasets as datasets_db

    pid = _project(client)
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"conversations": [{"from": "human", "value": "Q?"}, {"from": "gpt", "value": "A"}]}) + "\n",
                    encoding="utf-8")
    datasets_db.create_dataset(pid, "rows", str(rows), source="data-prep-export", qa_count=1)
    merged = _merged_export(pid, tmp_path)
    other_project_export = _merged_export(_project(client, "other"), tmp_path, "other-out")
    install_engine(monkeypatch, FakeEngine(model_path="/loaded/elsewhere"))

    r = client.post(f"{API}/{route}", json=_body(route, pid, tmp_path, tmp_path))
    assert r.status_code == 202, r.text
    assert r.json()["benchmark"]["model_path"] == merged != other_project_export
    wait_run_finished(r.json()["benchmark_id"])
    assert _isolated == [merged]


def test_an_explicit_model_path_wins_over_the_merged_export(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated: list[str],
) -> None:
    pid = _project(client)
    _merged_export(pid, tmp_path)
    install_engine(monkeypatch, FakeEngine(model_path=None))
    for key in ("model_path", "path"):
        r = client.post(f"{API}/run-suite", json={"suite_path": _suite(tmp_path, 1), "project_id": pid, key: "/chosen/q4.gguf"})
        assert r.status_code == 202, r.text
        assert r.json()["benchmark"]["model_path"] == "/chosen/q4.gguf"
        wait_run_finished(r.json()["benchmark_id"])
    assert _isolated == ["/chosen/q4.gguf"] * 2


def test_a_run_whose_model_cannot_load_fails_with_the_reason_and_is_listed(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _project(client)
    install_engine(monkeypatch)

    def boom(path: str) -> None:
        raise RuntimeError(f"cannot load {path}")

    monkeypatch.setattr(testing_jobs, "_load_model", boom)
    bid = _start_run(client, pid, _suite(tmp_path))["benchmark_id"]
    wait_run_finished(bid)
    row = client.get(_runs(pid, bid)).json()
    assert row["status"] == "failed" and "cannot load /m/model" in row["error"] and row["progress_done"] == 0
    assert client.get(_runs(pid, bid) + "/cases").json() == []


def test_the_untrained_base_model_is_a_testing_target_of_its_own(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid = _project(client)  # created with base_model "x/test"
    _merged_export(pid, tmp_path)
    install_engine(monkeypatch, FakeEngine(model_path="/loaded/elsewhere"))
    r = client.post(f"{API}/run-suite", json={"suite_path": _suite(tmp_path, 1), "project_id": pid, "model_path": "__base__"})
    assert r.status_code == 202, r.text
    assert r.json()["benchmark"]["model_path"] == "x/test"  # not the merged export, not whatever is loaded
    wait_run_finished(r.json()["benchmark_id"])


def test_the_base_model_choice_is_400_when_the_project_has_none(client: TestClient, tmp_path: Path) -> None:
    pid = client.post("/api/projects", json={"name": "no-base"}).json()["id"]
    r = client.post(f"{API}/run-suite", json={"suite_path": _suite(tmp_path, 1), "project_id": pid, "model_path": "__base__"})
    assert r.status_code == 400 and "no base model" in r.json()["error"]


def test_the_testing_page_lists_the_base_model_choice(client: TestClient) -> None:
    pid = _project(client)
    assert 'value="__base__"' in client.get(f"/projects/{pid}/testing").text
