"""Chat page (``/api/chat-v2/projects/<pid>/chat``) asks a model the way Testing and the Smart Brain chat do (B7, 2026-11-01).

B7 found Chat answering a fact differently from the Testing page for the same GGUF. Tracing both paths showed the Chat
route (1) dropped the Top-k control, (2) built its own grounded prompt with no "answer only from the context, say so
when it is not there" instruction, unlike the Smart Brain chat and the context-grounded training rows, and (3) asked a
trained model under the project's *current* prompt while Testing asks it under the prompt the run was trained with.
"""
from __future__ import annotations

from unittest.mock import patch

from finetune_studio import db
from finetune_studio.smart_brain import prompt as brain_prompt


def _project(client, name: str, system_prompt: str = "") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code in (200, 201), r.text
    pid = r.json()["id"]
    if system_prompt:
        db.update_project(pid, system_prompt=system_prompt)
    return pid


def _done_run(pid: str, out: str, system_prompt: str, mode: str = "bake") -> str:
    run = db.create_run(project_id=pid, name="r", system_prompt=system_prompt, system_prompt_mode=mode)
    db.update_run(run["id"], status="done", output_path=out)
    return run["id"]


def _chat(client, pid: str, body: dict, *, model_path: str = "/models/base.gguf"):
    from finetune_studio.webui import app as webapp

    sent: list[tuple[list, dict]] = []
    with patch.object(webapp.inference_engine, "model", object()), \
         patch.object(webapp.inference_engine, "model_path", model_path), \
         patch.object(webapp.inference_engine, "n_ctx", 8192), \
         patch.object(webapp.inference_engine, "generate", side_effect=lambda m, **k: sent.append((m, k)) or "ok"):
        r = client.post(f"/api/chat-v2/projects/{pid}/chat", json=body)
    assert r.status_code == 200, r.text
    assert len(sent) == 1
    return r.json(), sent[0][0], sent[0][1]


def test_top_k_control_reaches_the_model(client) -> None:
    pid = _project(client, "chatfix-topk")
    _, _, kw = _chat(client, pid, {"messages": [{"role": "user", "content": "hi"}], "top_k": 7, "temperature": 0.3})
    assert kw["top_k"] == 7 and kw["temperature"] == 0.3


def test_trained_model_is_asked_under_its_trained_prompt_like_testing(client) -> None:
    pid = _project(client, "chatfix-trained", system_prompt="Project prompt changed after training.")
    _done_run(pid, "/out/run1", "Prompt the run was trained with.")
    body = {"messages": [{"role": "user", "content": "q"}], "system_prompt_override": "Project prompt changed after training."}
    data, msgs, _ = _chat(client, pid, body, model_path="/out/run1/gguf/model-q4_k_m.gguf")
    assert msgs[0] == {"role": "system", "content": "Prompt the run was trained with."}
    assert data["request"]["system_prompt_source"] == "trained_run"


def test_run_trained_without_a_prompt_gets_none_in_chat_too(client) -> None:
    pid = _project(client, "chatfix-noprompt", system_prompt="Persona.")
    _done_run(pid, "/out/run2", "", mode="none")
    data, msgs, _ = _chat(client, pid, {"messages": [{"role": "user", "content": "q"}]}, model_path="/out/run2/merged")
    assert msgs == [{"role": "user", "content": "q"}]
    assert data["request"]["system_prompt"] == ""


def test_an_edited_prompt_still_wins(client) -> None:
    pid = _project(client, "chatfix-edited", system_prompt="Persona.")
    _done_run(pid, "/out/run3", "Trained.")
    body = {"messages": [{"role": "user", "content": "q"}], "system_prompt_override": "My own prompt."}
    data, msgs, _ = _chat(client, pid, body, model_path="/out/run3/merged")
    assert msgs[0]["content"] == "My own prompt." and data["request"]["system_prompt_source"] == "edited"


def test_base_model_uses_the_project_prompt(client) -> None:
    pid = _project(client, "chatfix-base", system_prompt="Persona.")
    data, msgs, _ = _chat(client, pid, {"messages": [{"role": "user", "content": "q"}]})
    assert msgs[0]["content"] == "Persona." and data["request"]["system_prompt_source"] == "project"


def test_grounded_chat_uses_the_smart_brain_prompt_layout(client, monkeypatch) -> None:
    pid = _project(client, "chatfix-grounded")
    idx = db.create_brain_index(project_id=pid, name="handbook", store_path="/tmp/handbook")
    monkeypatch.setattr("finetune_studio.webui.routes.chat_v2._search_brain_index",
                        lambda store_path, query, top_k=5: [
                            {"text": "Incident INC-2024-0614 was opened.", "score": 0.9, "chunk_id": "c1",
                             "filename": "incident_report.md"}])
    body = {"messages": [{"role": "user", "content": "Which incident?"}], "enabled_index_ids": [idx["id"]]}
    data, msgs, _ = _chat(client, pid, body)
    hit = {"rank": 1, "text": "Incident INC-2024-0614 was opened.", "filename": "incident_report.md", "rrf_score": 0.9}
    expected = brain_prompt.build_system_prompt(brain_prompt.format_context_blocks([hit], max_chars=100_000))
    assert msgs[0] == {"role": "system", "content": expected}
    assert brain_prompt.DEFAULT_SYSTEM_PROMPT.split(".")[0] in msgs[0]["content"]
    assert msgs[1:] == [{"role": "user", "content": "Which incident?"}]
    assert data["chunks_retrieved"] == 1 and data["request"]["grounded"] is True
