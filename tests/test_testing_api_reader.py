"""Testing can use an AI assistant API connection as the Smart Brain reader, and the model list is clean
(walkthrough 2026-11-13 M8, m19).

Smart Brain mode listed only local files, defaulted to "— latest merged export —" in a project without exports,
showed another project's exports (seven runs of agent-test-quillmoor-s) and duplicates (Qwen3-0.6B and
Qwen/Qwen3-0.6B). A user without a GPU could not test the package's answers at all.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from finetune_studio.webui import testing_models as tm


def _m(path: str, name: str, fmt: str = "gguf", project_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(path=path, name=name, format=fmt, size_gb=1.0, project_id=project_id)


def test_library_list_is_this_projects_and_installed_models_once() -> None:
    found = [_m("/m/Qwen3-0.6B", "Qwen3-0.6B", "safetensors"),
             _m("/hf/Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", "safetensors"),
             _m("/out/projects/other/run1/merged", "quillmoor · run 1", "safetensors", project_id="other"),
             _m("/out/projects/mine/run2/q4.gguf", "mine · run 2", project_id="mine"),
             _m("/m/mmproj-x.gguf", "mmproj-x")]
    listed = tm.library_models_for_testing([], "mine", discovered=found)
    assert [m["name"] for m in listed] == ["mine · run 2", "Qwen3-0.6B"]


def test_registry_for_project_drops_other_projects_runs() -> None:
    from finetune_studio.models.registry import for_project

    found = [_m("/a", "A"), _m("/b", "B", project_id="other"), _m("/c", "C", project_id="mine"), _m("/a/", "A2")]
    assert [m.name for m in for_project(found, "mine")] == ["A", "C"]


@pytest.fixture()
def api_conn(client):
    from finetune_studio.models.manager import get_manager

    get_manager().upsert_provider(id="api-reader-test", name="Reader", kind="openai_compat", model_id="gpt-x",
                                  base_url="http://127.0.0.1:9/v1")
    return "api:api-reader-test"


def test_api_connection_is_a_valid_reader_only_for_smart_brain(client, api_conn: str) -> None:
    assert tm.target_model("p", api_conn) == api_conn
    with pytest.raises(ValueError, match="not an API connection"):
        tm.target_model("p", "api:nope")
    pid = client.post("/api/projects", json={"name": "agent-test-api-reader"}).json()["id"]
    r = client.post("/api/testing/run-suite", json={"project_id": pid, "suite_path": "/nope.json",
                                                    "model_path": api_conn})
    assert r.status_code in (400, 404)
    readers = tm.api_readers()
    assert {"value": api_conn, "id": "api-reader-test"}.items() <= next(
        r for r in readers if r["value"] == api_conn).items()
    assert tm.default_reader_for_testing([], readers) in {r["value"] for r in readers}
    assert tm.default_reader_for_testing([{"path": "/x/merged", "format": "safetensors"}], readers) == "/x/merged"


def test_testing_page_offers_the_connection_and_no_phantom_export(client, api_conn: str) -> None:
    pid = client.post("/api/projects", json={"name": "agent-test-api-reader-page"}).json()["id"]
    page = client.get(f"/projects/{pid}/testing").text
    assert 'id="t-model-api"' in page and f'value="{api_conn}" data-api="1"' in page
    assert "<option value=\"\">— latest merged export —</option>" not in page   # this project has no export
    assert "o.disabled = m !== 'brain'" in page and 'id="t-model-api-note"' in page


def test_a_brain_run_reads_with_the_api_connection(monkeypatch, api_conn: str) -> None:
    from finetune_studio.testing import api_reader
    from finetune_studio.webui import testing_jobs

    calls: dict = {}

    class FakeClient:
        def chat(self, messages, **gen):
            calls["gen"] = gen
            return "BL310 costs 39 GBP a month."

    monkeypatch.setattr("finetune_studio.models.assistant.connection_client", lambda cid: FakeClient())
    monkeypatch.setattr(testing_jobs, "_load_model", lambda path: pytest.fail("an API reader must not load a model"))
    seen: dict = {}

    def fake_suite(engine, query, cases, **kw):
        seen["engine"], seen["kw"] = engine, kw
        return []

    monkeypatch.setattr(testing_jobs, "run_brain_suite", fake_suite)
    monkeypatch.setattr(testing_jobs, "open_brain_query", lambda **kw: (object(), "/corpus"))
    monkeypatch.setattr(testing_jobs, "compute_retrieval_metrics", lambda results: {})
    monkeypatch.setattr("finetune_studio.smart_brain.identity.system_prompt", lambda pid: "")
    spec = testing_jobs.RunSpec(project_id="p", kind="brain", suite_name="q", cases=[], model_path=api_conn)
    import threading
    out = testing_jobs._execute_run(spec, "bid", "owner", threading.Event())
    engine = seen["engine"]
    assert isinstance(engine, api_reader.ApiReader) and engine.model_path == api_conn
    assert seen["kw"]["max_context_chars"] > 20000 and out["corpus_path"] == "/corpus"
    assert engine.generate([{"role": "user", "content": "q"}], max_tokens=64) == "BL310 costs 39 GBP a month."
    assert calls["gen"]["max_tokens"] == 64
