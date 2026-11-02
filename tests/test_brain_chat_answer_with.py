"""Smart Brain chat can answer with the AI assistant's API connection instead of a loaded local model.

The best Quillmoor result (96.1 % judged, tests/corpus/quillmoor/RESULTS.md section 8) came from an API answer model
with follow-up search, but the studio's own chat only used the model loaded from the top-bar pill: a user with an API
connection and no GPU could not use grounded chat at all.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
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


def _assistant_chat(client, conn: MagicMock):
    pid = _project(client)
    mgr = MagicMock()
    mgr.get_provider.return_value = {"id": "api-x", "kind": "openai_compat"}
    from finetune_studio.webui import app as webapp

    with patch("finetune_studio.smart_brain.BrainIndex", return_value=_rag()), \
         patch.object(webapp.inference_engine, "model", None), \
         patch("finetune_studio.models.assistant.job_provider_id", return_value="api-x"), \
         patch("finetune_studio.models.manager.get_manager", return_value=mgr), \
         patch("finetune_studio.models.assistant.connection_client", return_value=conn):
        return client.post(f"/api/projects/{pid}/brain/chat",
                           json={"messages": [{"role": "user", "content": "q"}], "answer_with": "assistant",
                                 "follow_up": 0})


def test_a_provider_failure_is_a_502_with_its_reason(client) -> None:
    conn = MagicMock()
    conn.chat.side_effect = RuntimeError("HTTP 401: invalid API key")
    r = _assistant_chat(client, conn)
    assert r.status_code == 502, r.text
    assert r.json()["code"] == "assistant_failed"
    assert "HTTP 401: invalid API key" in r.json()["detail"]


def test_an_api_answer_does_not_wait_for_the_local_engine(client) -> None:
    """The engine lock serialises local inference; an API answer held it for the whole provider round trip."""
    from finetune_studio.webui.engine_guard import ENGINE_LOCK

    seen: list[bool] = []
    conn = MagicMock()
    conn.chat.side_effect = lambda *_a, **_k: seen.append(ENGINE_LOCK.locked()) or "ok"
    r = _assistant_chat(client, conn)
    assert r.status_code == 200, r.text
    assert seen == [False]


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
    assert "sel.value = 'assistant'" not in html     # never auto-selected (Genor 2026-11-02, #51)
    send_button = html.split('<button onclick="sendChat()" id="chat-btn"', 1)[1].split("</button>", 1)[0]
    assert "selected answer source" in send_button
    assert "Requires a model to be loaded first" not in send_button
    assert "initChatAnswerWith();" in html


_TEMPLATE = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/brain.html"


def _run_answer_with(loaded: bool, api: bool, choose: str | None = None) -> dict:
    """Execute the page's initChatAnswerWith / updateChatAnswerHint in Node with stubbed fetches."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the template's chat handlers")
    html = _TEMPLATE.read_text()
    start = html.index('  // "Answer with": always the local model by default.')
    end = html.index("  // Fill the form controls from the saved manifest")
    source = html[start:end]
    harness = r"""
const options = {
  model: {value: 'model', textContent: 'the loaded chat model', disabled: false},
  assistant: {value: 'assistant', textContent: 'the AI assistant (API connection)', disabled: false},
};
const sel = {value: 'model', querySelector: q => options[q.match(/value="(\w+)"/)[1]]};
const hint = {textContent: '', hidden: true};
const elements = {'chat-answer-with': sel, 'chat-answer-hint': hint};
const window = {};
const context = {
  window, document: {getElementById: id => elements[id]},
  fetch: async url => ({ok: true, json: async () => url.includes('inference') ? {loaded: LOADED}
    : {jobs: [{id: 'brain', provider_kind: API ? 'openai_compat' : 'local_gguf', provider_label: 'OpenCode Go'}]}}),
};
context.updateChatAnswerHint = (...a) => window.updateChatAnswerHint(...a);
require('node:vm').createContext(context);
require('node:vm').runInContext(SOURCE, context);
(async () => {
  await window.initChatAnswerWith();
  if (CHOOSE) { sel.value = CHOOSE; window.updateChatAnswerHint(); }
  console.log(JSON.stringify({value: sel.value, hint: hint.textContent, hidden: hint.hidden,
                              option: options.assistant.textContent, disabled: options.assistant.disabled}));
})().catch(e => { console.error(e); process.exitCode = 1; });
"""
    script = (f"const LOADED = {json.dumps(loaded)}; const API = {json.dumps(api)}; "
              f"const CHOOSE = {json.dumps(choose)}; const SOURCE = {json.dumps(source)};\n{harness}")
    r = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_no_local_model_keeps_the_local_default_and_says_how_to_continue() -> None:
    """#51: an API connection is a deliberate choice, never a fallback when no local model is loaded."""
    out = _run_answer_with(loaded=False, api=True)
    assert out["value"] == "model"
    assert not out["hidden"]
    assert "Load a model in Inference" in out["hint"]
    assert "the AI assistant (OpenCode Go)" in out["hint"]
    assert "sends the question and the retrieved passages to that provider" in out["hint"]
    assert out["option"] == "the AI assistant (OpenCode Go)" and not out["disabled"]


def test_the_api_option_stays_selectable_and_names_the_provider() -> None:
    out = _run_answer_with(loaded=False, api=True, choose="assistant")
    assert out["value"] == "assistant"
    assert "OpenCode Go" in out["hint"] and "sent to that provider" in out["hint"]


def test_no_hint_when_a_local_model_is_loaded() -> None:
    out = _run_answer_with(loaded=True, api=True)
    assert out["value"] == "model" and out["hidden"] and out["hint"] == ""


def test_without_an_api_connection_the_hint_only_offers_loading_a_model() -> None:
    out = _run_answer_with(loaded=False, api=False)
    assert out["value"] == "model" and out["disabled"]
    assert "Load a model in Inference" in out["hint"] and "AI assistant" not in out["hint"]
