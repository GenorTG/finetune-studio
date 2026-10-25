"""Use-case API: build from examples (dataset + held-out suite + manifest), honest 4xx, upload, teacher answers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.benchmarks.suite_defs import discover_suites
from finetune_studio.testing.suite import load_test_suite

LABELS = ("billing", "shipping", "account")


def _project(client: TestClient) -> str:
    r = client.post("/api/projects", json={"name": "uc", "base_model": "x/test"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _cls_rows(n: int = 30) -> list[dict]:
    return [{"input": f"ticket {i} about {lab} number {i}", "label": lab} for lab in LABELS for i in range(n)]


def test_templates_endpoint_lists_the_four_with_their_needs(client: TestClient) -> None:
    r = client.get("/api/usecases/templates")
    assert r.status_code == 200
    by_id = {t["id"]: t for t in r.json()}
    assert set(by_id) == {"extraction", "classification", "style", "distillation"}
    assert by_id["classification"]["minimum_rows"] == 70 and by_id["style"]["scoring"] == "ab"
    assert by_id["extraction"]["row_shape"] == {"input": "the source text", "output": by_id["extraction"]["row_shape"]["output"]}


def test_build_registers_the_dataset_writes_the_suite_and_the_manifest(client: TestClient) -> None:
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/usecases", json={"template": "classification", "name": "Ticket router",
                                                           "rows": _cls_rows()})
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["template"] == "classification" and m["train_rows"] == 70 and m["heldout"] == 20
    assert m["train_url"] == f"/projects/{pid}/training?dataset={m['dataset']['id']}"
    ds = db.get_dataset(m["dataset"]["id"])
    assert ds["project_id"] == pid and ds["qa_count"] == 70 and ds["source"] == "usecase"
    rows = [json.loads(line) for line in Path(m["dataset"]["path"]).read_text().splitlines()]
    assert len(rows) == 70 and rows[0]["messages"][0]["role"] == "system"
    suite = json.loads(Path(m["suite"]["path"]).read_text())
    assert suite["usecase"] == {"id": m["id"], "template": "classification", "name": "Ticket router",
                                "labels": sorted(LABELS), "schema": None, "style_guide": "", "teacher": ""}
    cases = load_test_suite(m["suite"]["path"])
    assert len(cases) == 20 and all(c.system_prompt == m["instruction"] for c in cases)
    assert m["suite"]["path"] in {s["path"] for s in discover_suites(pid)}  # Testing / Compare can run it again
    assert m["training"]["preset_id"] == "qlora" and m["training"]["overrides"]["num_epochs"] == 5
    listed = client.get(f"/api/projects/{pid}/usecases").json()
    assert [u["id"] for u in listed] == [m["id"]]
    assert client.get(f"/api/projects/{pid}/usecases/{m['id']}").json()["dataset"] == m["dataset"]


@pytest.mark.parametrize(("body", "status", "needle"), [
    ({"template": "summary", "rows": [{}]}, 400, "unknown use-case template"),
    ({"template": "style", "rows": []}, 400, "rows required"),
    ({"template": "style", "rows": [{"prompt": "a", "answer": "b"}] * 3}, 422, "needs at least 70"),
    ({"template": "style", "rows": [{"prompt": "a"}]}, 422, "not usable"),
    ({"template": "extraction", "rows": [{"input": "a", "output": {}}], "schema": {"oneOf": []}}, 400, "schema:"),
])
def test_bad_builds_are_honest_4xx(client: TestClient, body: dict, status: int, needle: str) -> None:
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/usecases", json=body)
    assert r.status_code == status and needle in r.json()["detail"], r.text
    if status == 422:
        assert r.json()["code"] in ("too_few_examples", "invalid_examples")
    assert client.get(f"/api/projects/{pid}/usecases").json() == []


def test_invalid_rows_come_back_listed(client: TestClient) -> None:
    pid = _project(client)
    rows = [*_cls_rows(), {"input": "x"}]
    r = client.post(f"/api/projects/{pid}/usecases", json={"template": "classification", "rows": rows})
    assert r.status_code == 422 and r.json()["errors"] == ["row 91: missing label / class / category"]


def test_unknown_project_and_use_case_are_404(client: TestClient) -> None:
    assert client.post("/api/projects/nope/usecases", json={"template": "style", "rows": [{}]}).status_code == 404
    pid = _project(client)
    assert client.get(f"/api/projects/{pid}/usecases/deadbeef").status_code == 404
    assert client.get(f"/api/projects/{pid}/usecases/../etc").status_code == 404
    assert client.delete(f"/api/projects/{pid}/usecases/deadbeef").status_code == 404


def test_upload_takes_jsonl_and_options(client: TestClient) -> None:
    pid = _project(client)
    body = "\n".join(json.dumps(r) for r in _cls_rows()).encode()
    r = client.post(f"/api/projects/{pid}/usecases/upload", files={"file": ("rows.jsonl", body)},
                    data={"options": json.dumps({"template": "classification", "name": "up"})})
    assert r.status_code == 201, r.text
    assert r.json()["name"] == "up"
    bad = client.post(f"/api/projects/{pid}/usecases/upload", files={"file": ("rows.jsonl", b"{oops")},
                      data={"options": json.dumps({"template": "classification"})})
    assert bad.status_code == 400 and "line 1 is not valid JSON" in bad.json()["detail"]
    opts = client.post(f"/api/projects/{pid}/usecases/upload", files={"file": ("rows.jsonl", body)},
                       data={"options": "[1]"})
    assert opts.status_code == 400 and "options" in opts.json()["detail"]


def test_delete_keeps_the_dataset(client: TestClient) -> None:
    pid = _project(client)
    m = client.post(f"/api/projects/{pid}/usecases", json={"template": "classification", "rows": _cls_rows()}).json()
    r = client.delete(f"/api/projects/{pid}/usecases/{m['id']}")
    assert r.status_code == 200 and r.json()["kept_dataset"] == m["dataset"]["id"]
    assert not Path(m["suite"]["path"]).exists() and Path(m["dataset"]["path"]).exists()
    assert client.get(f"/api/projects/{pid}/usecases").json() == []


def _prompts(n: int) -> list[dict]:
    return [{"prompt": f"summarise item {i}"} for i in range(n)]


def test_distillation_teacher_writes_missing_answers_and_drops_failures(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.usecases import teacher

    seen: list[list[dict]] = []

    def chat(messages: list[dict]) -> str:
        seen.append(messages)
        q = messages[-1]["content"]
        if q.endswith(" 7"):
            raise RuntimeError("HTTP 429")
        return f"<think>hm</think>Summary of {q}."

    monkeypatch.setattr(teacher, "open_teacher", lambda pid: (chat, "big-teacher"))
    pid = _project(client)
    rows = [*_prompts(130), {"prompt": "given", "teacher_answer": "already answered"}]
    r = client.post(f"/api/projects/{pid}/usecases", json={"template": "distillation", "rows": rows,
                                                           "teacher_provider_id": "api", "instruction": "Be short."})
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["teacher"] == "big-teacher" and m["examples"] == 130
    assert m["teacher_report"] == {"asked": 130, "answered": 129, "failed": 1, "errors": ["row 8: HTTP 429"],
                                   "model": "big-teacher"}
    assert all(msgs[0] == {"role": "system", "content": "Be short."} for msgs in seen)
    answers = {c.correct_answer for c in load_test_suite(m["suite"]["path"])}
    assert all(a.startswith("Summary of") or a == "already answered" for a in answers)
    assert client.get(f"/api/projects/{pid}/usecases/teacher/progress").json()["running"] is False


def test_a_local_teacher_is_refused_and_a_silent_teacher_is_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetune_studio.models.manager import get_manager
    from finetune_studio.usecases import teacher

    get_manager().upsert_provider(id="local-x", name="Local", kind="local_gguf", model_id="/m/x.gguf", base_url="")
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/usecases", json={"template": "distillation", "rows": _prompts(130),
                                                           "teacher_provider_id": "local-x"})
    assert r.status_code == 400 and "must be an API provider" in r.json()["detail"]

    def dead(_m: list[dict]) -> str:
        raise RuntimeError("down")

    monkeypatch.setattr(teacher, "open_teacher", lambda pid: (dead, "t"))
    r = client.post(f"/api/projects/{pid}/usecases", json={"template": "distillation", "rows": _prompts(130),
                                                           "teacher_provider_id": "api"})
    assert r.status_code == 502 and r.json()["teacher"]["failed"] == 130


def test_distillation_without_answers_or_teacher_lists_the_rows(client: TestClient) -> None:
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/usecases", json={"template": "distillation", "rows": _prompts(130)})
    assert r.status_code == 422 and r.json()["errors"][0] == "row 1: missing teacher_answer / answer / output"
