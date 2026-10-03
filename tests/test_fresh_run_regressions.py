"""Base-model benchmark results must appear on the Benchmarks page.

Found in the 2026-10-02 fresh-DB browser run: RUN on the "base model
(untrained)" row stored its scores against the hidden ``__base_model__``
placeholder run, and the page filtered that run out, so the user saw
"No benchmark scores yet" after a successful run.
"""

from __future__ import annotations

from finetune_studio import db


def test_base_model_benchmark_listed_on_page(client) -> None:
    project = db.create_project("Base bench", base_model="Qwen/Qwen3-0.6B")
    pid = project["id"]
    base_run = db.create_run(pid, "__base_model__", base_model="Qwen/Qwen3-0.6B")
    db.update_run(base_run["id"], status="done", output_path="", notes="placeholder")
    db.create_benchmark(
        base_run["id"], "suite-x", {"pass_rate": 9.5, "weighted_score": 9.5}, 100,
        cases=[], model_path="Qwen/Qwen3-0.6B",
    )

    html = client.get(f"/projects/{pid}/benchmarks").text

    assert "No benchmark scores yet" not in html
    assert "base model (untrained)" in html
    assert "suite-x" in html


def test_base_probe_placeholder_hidden_from_run_lists(client) -> None:
    project = db.create_project("Probe hide", base_model="Qwen/Qwen3-0.6B")
    pid = project["id"]
    real = db.create_run(pid, "real run", base_model="Qwen/Qwen3-0.6B")
    probe = db.create_run(pid, "__base_model__", base_model="Qwen/Qwen3-0.6B")

    assert [r["id"] for r in db.list_runs(pid)] == [real["id"]]
    assert probe["id"] in {r["id"] for r in db.list_runs(pid, include_base_probe=True)}
    html = client.get(f"/projects/{pid}/training").text
    assert "__base_model__" not in html


def test_chat_context_offers_tuned_merged_model(client, tmp_path) -> None:
    """Chat must offer the project's fine-tuned (merged) model, not only bases."""
    project = db.create_project("Chat tuned", base_model="Qwen/Qwen3-0.6B")
    pid = project["id"]
    out = tmp_path / "run"
    (out / "merged").mkdir(parents=True)
    (out / "merged" / "config.json").write_text("{}")
    (out / "merged" / "model.safetensors").write_bytes(b"x" * 16)
    run = db.create_run(pid, "tuned", base_model="Qwen/Qwen3-0.6B")
    db.update_run(run["id"], status="done", output_path=str(out))
    db.update_project(pid, production_run=run["id"])

    body = client.get(f"/api/chat-v2/projects/{pid}/context").json()

    assert body["tuned_model_path"] == str(out / "merged")


def test_export_worker_refreshes_model_registry(monkeypatch):
    from finetune_studio.webui.routes import exports, models

    calls = []
    monkeypatch.setattr(models, "refresh_model_registry", lambda: calls.append(1) or 0)
    exports._refresh_registry_quietly()
    assert calls == [1]


def test_rag_settings_before_index_is_not_a_404(client):
    pid = client.post("/api/projects", json={"name": "Rag Fresh"}).json()["id"]
    r = client.get(f"/api/projects/{pid}/rag/settings")
    assert r.status_code == 200
    assert r.json() == {"exists": False}


def test_unknown_page_gets_styled_404_but_api_stays_json(client):
    page = client.get("/projects/nope/files", headers={"accept": "text/html"})
    assert page.status_code == 404
    assert "Page not found" in page.text
    api = client.get("/api/nope", headers={"accept": "text/html"})
    assert api.status_code == 404
    assert api.json() == {"detail": "Not Found"}
    bare = client.get("/projects/nope/files")
    assert bare.json() == {"detail": "Not Found"}


def test_project_overview_links_and_base_model_badge(client):
    pid = client.post("/api/projects", json={"name": "Links"}).json()["id"]
    from finetune_studio import db

    db.update_project(pid, base_model="/x/hf_models/Qwen__Qwen3-0.6B")
    page = client.get(f"/projects/{pid}").text
    assert f'href="/projects/{pid}/data" data-link>\n    <div class="label">Files' in page
    assert f"/projects/{pid}/data-prep#upload" not in page
    assert "base: Qwen/Qwen3-0.6B" in client.get("/projects").text
