"""Use-case evaluation: the untrained base and the run answer the same held-out questions; the result is paired.

Real jobs, rows and routes; the engine (answers depend on the resident model) and the judges are fakes. Pins: code
scores extraction / classification when the answers are saved (no AI judge), distillation is judged against the teacher,
style gets the blind A/B job, and every report pairs base vs run per question (McNemar / sign test).
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.db import judgements as jdb
from finetune_studio.testing.judge import LoadedJudge
from finetune_studio.webui import testing_jobs
from tests.testing_run_support import (
    FakeEngine,
    add_api_provider,
    install_engine,
    wait_until,
)

BASE, TUNED = "/m/base-model", "/m/runs/r1"
LABELS = ("billing", "shipping", "account")


@pytest.fixture(autouse=True)
def _models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    monkeypatch.setattr("finetune_studio.webui.routes.settings.SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr("finetune_studio.webui.testing_models.local_model_missing", lambda _p: False)
    state = {"model": ""}
    monkeypatch.setattr(testing_jobs, "_load_model", lambda path: state.update(model=path))
    return state


def _engine(monkeypatch: pytest.MonkeyPatch, state: dict[str, str], tuned, base) -> FakeEngine:
    return install_engine(monkeypatch, FakeEngine(lambda q: tuned(q) if state["model"] == TUNED else base(q)))


def _setup(client: TestClient, template: str, rows: list[dict], **opts) -> tuple[str, dict, str]:
    pid = client.post("/api/projects", json={"name": f"uc-{template}-{time.time_ns()}", "base_model": BASE}).json()["id"]
    r = client.post(f"/api/projects/{pid}/usecases", json={"template": template, "rows": rows, **opts})
    assert r.status_code == 201, r.text
    run = db.create_run(pid, "train", base_model=BASE)
    db.update_run(run["id"], status="done", output_path=TUNED)
    return pid, r.json(), run["id"]


def _evaluate(client: TestClient, pid: str, uid: str, rid: str, **extra) -> dict:
    r = client.post(f"/api/projects/{pid}/usecases/{uid}/evaluate", json={"run_id": rid, **extra})
    assert r.status_code == 202, r.text
    return r.json()


def _report(client: TestClient, pid: str, uid: str, gid: str, *, ready=lambda rep: True) -> dict:
    def done() -> bool:
        rep = client.get(f"/api/projects/{pid}/usecases/{uid}/evaluations/{gid}").json()
        return rep["status"]["run"] != "running" and not testing_jobs.active_jobs() and ready(rep)

    wait_until(done, what="the evaluation to finish")
    return client.get(f"/api/projects/{pid}/usecases/{uid}/evaluations/{gid}").json()


def _label_of(q: str) -> str:
    return re.search(r"about (\w+)", q).group(1)


def test_classification_base_vs_run_is_scored_by_code_and_paired(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, _models: dict[str, str],
) -> None:
    rows = [{"input": f"ticket {i} about {lab} number {i}", "label": lab} for lab in LABELS for i in range(30)]
    pid, m, rid = _setup(client, "classification", rows)
    engine = _engine(monkeypatch, _models, tuned=_label_of, base=lambda q: "Probably something about the user.")
    ev = _evaluate(client, pid, m["id"], rid)
    rep = _report(client, pid, m["id"], ev["group_id"])
    assert rep["status"] == {"run": "done", "judge": ""}
    assert rep["base"]["model_path"] == BASE and rep["tuned"]["model_path"] == TUNED
    assert rep["tuned"]["metrics"]["accuracy"] == 1.0 and rep["tuned"]["metrics"]["label_only"] == 1.0
    assert rep["base"]["metrics"]["accuracy"] == 0.0 and rep["base"]["metrics"]["no_label"] == 1.0
    conf = rep["tuned"]["metrics"]["confusion"]
    assert sum(conf["matrix"][lab][lab] for lab in LABELS) == 20
    assert rep["paired"]["gained"] == 20 and rep["paired"]["lost"] == 0 and rep["paired"]["verdict"] == "better"
    assert rep["paired"]["p_value"] < 1e-5
    # every answer was decided by code, recorded as an exact judgement; the AI judge never ran
    for bid in (ev["base_bid"], ev["tuned_bid"]):
        js = jdb.list_judgements(benchmark_id=bid)
        assert len(js) == 20 and {j["kind"] for j in js} == {"exact"}
        assert (db.get_benchmark(bid) or {})["scores"]["usecase"]["template"] == "classification"
    # same instruction for base and run: the system turn of every question
    assert len(engine.asked) == 40
    assert client.get(f"/api/projects/{pid}/usecases/{m['id']}").json()["evaluations"][0]["group_id"] == ev["group_id"]


def test_extraction_reports_exact_match_field_f1_and_json_validity(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, _models: dict[str, str],
) -> None:
    rows = [{"input": f"Invoice {i} from Vendor{i}", "output": {"invoice": str(i), "vendor": f"Vendor{i}"}}
            for i in range(80)]
    pid, m, rid = _setup(client, "extraction", rows)

    def tuned(q: str) -> str:
        i = q.split()[1]
        return json.dumps({"invoice": i, "vendor": f"Vendor{i}"})

    def base(q: str) -> str:
        i = q.split()[1]
        return f'Sure! Here is the data: {{"invoice": "{i}", "vendor": "unknown"}}'

    _engine(monkeypatch, _models, tuned=tuned, base=base)
    ev = _evaluate(client, pid, m["id"], rid)
    rep = _report(client, pid, m["id"], ev["group_id"])
    t, b = rep["tuned"]["metrics"], rep["base"]["metrics"]
    assert (t["exact_match"], t["field_f1"], t["json_only"]) == (1.0, 1.0, 1.0)
    assert (b["exact_match"], b["field_f1"], b["json_valid"], b["json_only"]) == (0.0, 0.5, 1.0, 0.0)
    assert rep["paired"]["verdict"] == "better"
    base_cases = db.list_cases(ev["base_bid"])
    assert {c["verdict"] for c in base_cases} == {"partial"}  # half the fields right: partial, not a pass


def test_distillation_is_judged_against_the_teacher_answer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, _models: dict[str, str],
) -> None:
    rows = [{"prompt": f"summarise item {i}", "teacher_answer": f"Item {i}: three points."} for i in range(130)]
    pid, m, rid = _setup(client, "distillation", rows)
    _engine(monkeypatch, _models, tuned=lambda q: f"Item {q.split()[-1]}: three points.", base=lambda q: "No idea.")
    keys: list[str] = []

    def chat(messages: list[dict]) -> str:
        prompt = messages[-1]["content"]
        key = prompt.split("ANSWER KEY:\n", 1)[1].split("\n\n", 1)[0]
        keys.append(key)
        answer = prompt.split("MODEL ANSWER:\n", 1)[1]
        return json.dumps({"facts": [], "extra_wrong": False, "reasoning": "x",
                           "verdict": "pass" if answer.strip() == key.strip() else "fail", "confidence": 0.9})

    @contextmanager
    def fake_open(provider_id: str) -> Iterator[LoadedJudge]:
        yield LoadedJudge(chat=chat, provider_id=provider_id, model="judge-x", label="J", concurrent=False)

    monkeypatch.setattr("finetune_studio.webui.testing_jobs.open_judge", fake_open)
    ev = _evaluate(client, pid, m["id"], rid, judge_provider_id=add_api_provider())
    rep = _report(client, pid, m["id"], ev["group_id"], ready=lambda r: r["status"]["judge"] == "done")
    assert all(k.startswith("Item ") for k in keys) and len(keys) == 52  # teacher answers are the answer key
    assert rep["tuned"]["scores"]["pass_rate"] == 100.0 and rep["base"]["scores"]["passed"] == 0
    assert rep["paired"]["gained"] == 26 and rep["paired"]["verdict"] == "better" and rep["notes"] == []


def test_style_runs_the_blind_ab_job_and_reports_a_sign_test(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, _models: dict[str, str],
) -> None:
    rows = [{"prompt": f"question {i}?", "answer": f"Short. Warm. ({i})"} for i in range(80)]
    pid, m, rid = _setup(client, "style", rows, style_guide="Short and warm.")
    _engine(monkeypatch, _models, tuned=lambda q: "Short. Warm.", base=lambda q: "Dear customer, thank you ...")
    ev = _evaluate(client, pid, m["id"], rid)
    rep = _report(client, pid, m["id"], ev["group_id"])
    assert rep["ab"] is None and rep["notes"] == ["run the blind A/B judge to get the style result"]

    def chat(messages: list[dict]) -> str:
        first = messages[1]["content"].split("ANSWER 1:\n", 1)[1].split("\n\nANSWER 2:", 1)[0]
        return json.dumps({"reasoning": "closer", "winner": "1" if first.startswith("Short") else "2"})

    @contextmanager
    def fake_open(provider_id: str) -> Iterator[LoadedJudge]:
        yield LoadedJudge(chat=chat, provider_id=provider_id, model="judge-x", label="J", concurrent=True)

    monkeypatch.setattr("finetune_studio.testing.judge.open_judge", fake_open)
    url = f"/api/projects/{pid}/usecases/{m['id']}/evaluations/{ev['group_id']}/ab"
    r = client.post(url, json={"judge_provider_id": add_api_provider()})
    assert r.status_code == 202, r.text
    rep = _report(client, pid, m["id"], ev["group_id"], ready=lambda r: (r["ab"] or {}).get("status") == "done")
    s = rep["ab"]["summary"]
    assert (s["wins"], s["losses"], s["ties"], s["errors"]) == (20, 0, 0, 0) and s["verdict"] == "trained preferred"
    assert rep["ab"]["judge_model"] == "judge-x" and len(rep["ab"]["results"]) == 20
    # the A/B is the style template's metric only
    pid2, m2, rid2 = _setup(client, "classification",
                            [{"input": f"t {i} about {lab}", "label": lab} for lab in LABELS for i in range(30)])
    ev2 = _evaluate(client, pid2, m2["id"], rid2)
    _report(client, pid2, m2["id"], ev2["group_id"])
    r = client.post(f"/api/projects/{pid2}/usecases/{m2['id']}/evaluations/{ev2['group_id']}/ab", json={})
    assert r.status_code == 400 and "scored by code" in r.json()["detail"]


def test_evaluation_refuses_unfinished_or_foreign_runs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, _models: dict[str, str],
) -> None:
    rows = [{"prompt": f"question {i}?", "answer": f"reply {i}"} for i in range(80)]
    pid, m, _rid = _setup(client, "style", rows)
    url = f"/api/projects/{pid}/usecases/{m['id']}/evaluate"
    assert client.post(url, json={"run_id": "nope"}).status_code == 404
    other = client.post("/api/projects", json={"name": "other"}).json()["id"]
    foreign = db.create_run(other, "x", base_model=BASE)["id"]
    assert client.post(url, json={"run_id": foreign}).status_code == 404
    running = db.create_run(pid, "busy", base_model=BASE)["id"]
    r = client.post(url, json={"run_id": running})
    assert r.status_code == 409 and "has no trained model" in r.json()["detail"]
    nobase = db.create_run(pid, "nobase")["id"]
    db.update_run(nobase, status="done", output_path=TUNED)
    r = client.post(url, json={"run_id": nobase})
    assert r.status_code == 409 and "base model" in r.json()["detail"]
    assert client.get(f"/api/projects/{pid}/usecases/{m['id']}/evaluations/deadbeef").status_code == 404


def test_a_testing_page_run_of_a_usecase_suite_is_scored_by_code_too(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, _models: dict[str, str],
) -> None:
    """The suite stays usable on the Testing page; its answers get the same code verdicts there."""
    rows = [{"input": f"ticket {i} about {lab} number {i}", "label": lab} for lab in LABELS for i in range(30)]
    pid, m, _rid = _setup(client, "classification", rows)
    _engine(monkeypatch, _models, tuned=_label_of, base=_label_of)
    r = client.post("/api/testing/run-suite", json={"project_id": pid, "suite_path": m["suite"]["path"],
                                                     "model_path": TUNED, "auto_judge": False})
    assert r.status_code == 202, r.text
    bid = r.json()["benchmark_id"]
    wait_until(lambda: (db.get_benchmark(bid) or {}).get("status") == "done" and not testing_jobs.active_jobs())
    scores = (db.get_benchmark(bid) or {})["scores"]
    assert scores["passed"] == 20 and scores["usecase"]["accuracy"] == 1.0
