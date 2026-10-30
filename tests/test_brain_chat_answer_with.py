"""Smart Brain chat can answer with the AI assistant's API connection instead of a loaded local model.

The best Quillmoor result (96.1 % judged, tests/corpus/quillmoor/RESULTS.md section 8) came from an API answer model
with follow-up search, but the studio's own chat only used the model loaded from the top-bar pill: a user with an API
connection and no GPU could not use grounded chat at all.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from finetune_studio.models import assistant

HITS = [{"rank": 1, "chunk_id": "c1", "filename": "warranty.txt", "source": "warranty.txt",
         "text": "Aurora Kettle products are covered by a warranty of THREE YEARS.", "rrf_score": 0.03}]


def _project(client) -> str:
    return client.post("/api/projects", json={"name": "answer with"}).json()["id"]


def _rag() -> MagicMock:
    q = MagicMock()
    q.search.return_value = HITS
    rag = MagicMock()
    rag.exists.return_value = True
    rag.load.return_value = q
    return rag


def test_brain_job_is_api_only() -> None:
    job = assistant.job("brain")
    assert job.api_only and job.default == assistant.ASSISTANT
    assert "retrieved" in job.sends


def test_chat_answers_with_the_assistant_connection_without_a_loaded_model(client) -> None:
    pid = _project(client)
    conn = MagicMock()
    conn.chat.return_value = "Three years [warranty.txt]."
    mgr = MagicMock()
    mgr.get_provider.return_value = {"id": "api-x", "kind": "openai_compat"}
    from finetune_studio.webui import app as webapp

    with patch("finetune_studio.smart_brain.BrainIndex", return_value=_rag()), \
         patch.object(webapp.inference_engine, "model", None), \
         patch("finetune_studio.models.assistant.job_provider_id", return_value="api-x"), \
         patch("finetune_studio.models.manager.get_manager", return_value=mgr), \
         patch("finetune_studio.models.assistant.connection_client", return_value=conn) as cc:
        r = client.post(f"/api/projects/{pid}/brain/chat",
                        json={"messages": [{"role": "user", "content": "How long is the warranty?"}],
                              "answer_with": "assistant", "follow_up": 0})
    assert r.status_code == 200, r.text
    assert r.json()["reply"] == "Three years [warranty.txt]."
    cc.assert_called_once_with("api-x")
    sent = conn.chat.call_args.args[0]
    assert "THREE YEARS" in sent[0]["content"] and sent[-1]["content"] == "How long is the warranty?"


def test_assistant_without_an_api_connection_is_a_plain_409(client) -> None:
    pid = _project(client)
    mgr = MagicMock()
    mgr.get_provider.return_value = {"id": "local-default", "kind": "local_gguf"}
    with patch("finetune_studio.smart_brain.BrainIndex", return_value=_rag()), \
         patch("finetune_studio.models.assistant.job_provider_id", return_value="local-default"), \
         patch("finetune_studio.models.manager.get_manager", return_value=mgr):
        r = client.post(f"/api/projects/{pid}/brain/chat",
                        json={"messages": [{"role": "user", "content": "q"}], "answer_with": "assistant"})
    assert r.status_code == 409
    assert r.json()["code"] == "no_api_assistant"
    assert "Settings -> AI assistant" in r.json()["detail"]


def test_unknown_answer_with_is_rejected(client) -> None:
    pid = _project(client)
    with patch("finetune_studio.smart_brain.BrainIndex", return_value=_rag()):
        r = client.post(f"/api/projects/{pid}/brain/chat",
                        json={"messages": [{"role": "user", "content": "q"}], "answer_with": "somebody"})
    assert r.status_code == 422


def test_brain_page_offers_the_choice_and_sends_it() -> None:
    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/brain.html").read_text()
    chat_card = html.split('id="brain-chat"', 1)[1].split('id="chat-messages"', 1)[0]
    assert "loaded chat model or the AI assistant's API connection" in chat_card
    assert "needs a chat model loaded" not in chat_card
    assert 'placeholder="Ask a question about your indexed documents…"' in html
    select = html.split('id="chat-answer-with"', 1)[1].split("</select>", 1)[0]
    assert 'value="model"' in select and 'value="assistant"' in select
    assert "answer_with: (document.getElementById('chat-answer-with') || {}).value || 'model'" in html
    assert "if (!inf.loaded && api) sel.value = 'assistant';" in html
    send_button = html.split('<button onclick="sendChat()" id="chat-btn"', 1)[1].split("</button>", 1)[0]
    assert "selected answer source" in send_button
    assert "Requires a model to be loaded first" not in send_button
    assert "initChatAnswerWith();" in html
