"""Pin the message layout /api/projects/<pid>/rag/chat sends to the model.

Experiment (2026-10-05, Qwen3-0.6B tuned on 48 no-context QA rows, 5 questions,
3 of them unseen details from the indexed docs): moving CONTEXT into the final
user turn (4 layouts tried) never beat the system-prompt layout — 2/5 grounded
either way, 2/5 vs 4/5 on a model tuned with context rows. Small-model
grounding is decided by the training data, not by this prompt, so the layout is
pinned here to stop un-measured rewrites.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

HITS = [{
    "rank": 1, "filename": "warranty.txt", "source": "/x/warranty.txt",
    "text": "Aurora Kettle products are covered by a warranty of THREE YEARS.",
    "rrf_score": 0.033,
}]


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "RAG chat prompt"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _chat(client, body: dict) -> tuple[dict, list[dict]]:
    """POST rag/chat with a fake corpus + engine; return (response json, messages sent)."""
    pid = _project(client)
    fake_q = MagicMock()
    fake_q.search.return_value = HITS
    fake_q.format_context.return_value = "[1] (source: warranty.txt, score 0.033)\n" + HITS[0]["text"]
    fake_rag = MagicMock()
    fake_rag.exists.return_value = True
    fake_rag.load.return_value = fake_q
    sent: list[dict] = []

    def _gen(msgs, **_kw):
        sent.extend(msgs)
        return "Three years [warranty.txt]."

    from finetune_studio.webui import app as webapp

    with patch("finetune_studio.data.rag_portable.PortableRAG", return_value=fake_rag), \
         patch.object(webapp.inference_engine, "model", object()), \
         patch.object(webapp.inference_engine, "generate", side_effect=_gen):
        r = client.post(f"/api/projects/{pid}/rag/chat", json=body)
    assert r.status_code == 200, r.text
    return r.json(), sent


def test_context_and_citation_rule_ride_in_system_turn(client) -> None:
    out, sent = _chat(client, {"messages": [{"role": "user", "content": "How long is the warranty?"}]})
    assert sent[0]["role"] == "system"
    assert "CONTEXT:\n" in sent[0]["content"]
    assert "THREE YEARS" in sent[0]["content"]
    assert "[brackets]" in sent[0]["content"]
    assert sent[-1] == {"role": "user", "content": "How long is the warranty?"}
    assert out["sources"][0]["filename"] == "warranty.txt"
    assert out["reply"].startswith("Three years")


def test_history_is_preserved_and_custom_system_prompt_used(client) -> None:
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How long is the warranty?"},
    ]
    _, sent = _chat(client, {"messages": msgs, "system_prompt": "Be terse."})
    assert sent[0]["content"].startswith("Be terse.")
    assert sent[1:] == msgs


def test_missing_user_message_is_400(client) -> None:
    pid = _project(client)
    fake_rag = MagicMock()
    fake_rag.exists.return_value = True
    with patch("finetune_studio.data.rag_portable.PortableRAG", return_value=fake_rag):
        r = client.post(f"/api/projects/{pid}/rag/chat", json={"messages": [{"role": "assistant", "content": "x"}]})
    assert r.status_code == 400
