"""Tests for project training page Actions + promote-to-production."""
from __future__ import annotations


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Training Actions Test"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _done_run(pid: str, name: str = "good-run") -> dict:
    from finetune_studio import db

    run = db.create_run(project_id=pid, name=name, base_model="/m", data_path="/d")
    db.update_run(run["id"], status="done", output_path="/tmp/out/good")
    got = db.get_run(run["id"])
    assert got is not None
    return got


def test_resolve_production_run() -> None:
    from finetune_studio.webui.project_dashboard import resolve_production_run

    assert resolve_production_run({"production_run": ""}) is None
    runs = [{"id": "abc12345", "name": "best"}]
    assert resolve_production_run({"production_run": "abc12345"}, runs) == {
        "id": "abc12345",
        "name": "best",
    }
    missing = resolve_production_run({"production_run": "deadbeef"}, [])
    assert missing == {"id": "deadbeef", "name": "deadbeef"}


def test_training_page_has_actions_column(client) -> None:
    pid = _project(client)
    run = _done_run(pid)
    r = client.get(f"/projects/{pid}/training")
    assert r.status_code == 200
    body = r.text
    assert ">Actions<" in body
    assert "js-set-prod" in body
    assert "js-open-inf" in body
    assert f"/projects/{pid}/export?run={run['id']}" in body
    assert "⭐ Set production" in body
    assert "▶ Open in inference" in body
    assert "⬇ Export" in body
    # Detail panel still works with ?run=
    r2 = client.get(f"/projects/{pid}/training?run={run['id']}")
    assert r2.status_code == 200
    assert "Run details" in r2.text
    assert run["name"] in r2.text


def test_promote_returns_ok_and_run(client) -> None:
    from finetune_studio import db

    pid = _project(client)
    run = _done_run(pid, name="prod-candidate")
    r = client.post(f"/api/projects/{pid}/promote", json={"run_id": run["id"]})
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body.get("run", {}).get("id") == run["id"]
    assert body["run"]["name"] == "prod-candidate"
    proj = db.get_project(pid)
    assert proj is not None
    assert proj["production_run"] == run["id"]


def test_promote_unknown_run_errors(client) -> None:
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/promote", json={"run_id": "nope"})
    assert r.status_code == 200
    assert r.json().get("error") == "run not found"


def test_training_page_shows_production_pill_after_promote(client) -> None:
    pid = _project(client)
    run = _done_run(pid, name="ship-it")
    client.post(f"/api/projects/{pid}/promote", json={"run_id": run["id"]})
    r = client.get(f"/projects/{pid}/training")
    assert r.status_code == 200
    assert 'id="production-run-pill"' in r.text
    assert "ship-it" in r.text
    assert "⭐ production" in r.text


def test_export_page_accepts_run_query(client) -> None:
    pid = _project(client)
    run = _done_run(pid)
    r = client.get(f"/projects/{pid}/export?run={run['id']}")
    assert r.status_code == 200
    assert "export-run-radio" in r.text
    assert "params.get(\"run\")" in r.text
