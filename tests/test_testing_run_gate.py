"""Regression: the Testing Run button stays disabled until a quiz is picked; the RAG mode exposes its controls."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

_TESTING = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "project_testing.html"
)


def test_testing_template_disables_run_until_suite() -> None:
    html = _TESTING.read_text(encoding="utf-8")
    assert 'id="t-run-btn"' in html
    assert "Pick a quiz to enable Run" in html
    assert "btn.disabled = S.busy || !ready" in html  # the one place the Run button's state is decided
    assert "Pick a quiz first" in html
    # The raw transcript is a download / a fold-out, never the primary view.
    assert "rv-export-json" in html and "Full transcript" in html
    assert "JSON.stringify(scores" not in html


def test_testing_page_run_starts_disabled(client) -> None:
    pid = client.post("/api/projects", json={"name": "Testing Run Gate"}).json()["id"]
    body = client.get(f"/projects/{pid}/testing").text
    assert 'id="t-run-btn"' in body
    assert "disabled" in body.split('id="t-run-btn"', 1)[1].split(">", 1)[0]


def test_testing_template_rag_grounded_ui_contract() -> None:
    """RAG-grounded is a mode of the one Run form; it posts to /api/testing/run-rag-suite."""
    html = _TESTING.read_text(encoding="utf-8")
    assert 'name="t-mode"' in html and 'value="rag"' in html and "RAG-grounded" in html
    assert 'id="t-opts-rag"' in html
    assert 'id="t-rag-corpus"' in html and 'id="t-rag-topk"' in html and 'id="t-rag-maxtok"' in html
    assert "/api/testing/run-rag-suite" in html
    assert "$('t-opts-rag').hidden = m !== 'rag'" in html   # the RAG options only show in RAG mode
    # Retrieval trace is surfaced for RAG runs and per case.
    assert "answer text in retrieved chunks" in html
    assert "Retrieved context the model saw" in html
    assert "retrieval_hit" in html


def test_testing_page_renders_rag_controls(client: TestClient) -> None:
    pid = client.post("/api/projects", json={"name": "Testing RAG UI"}).json()["id"]
    body = client.get(f"/projects/{pid}/testing").text
    assert 'id="t-opts-rag"' in body and 'id="t-rag-corpus"' in body
    assert "RAG-grounded" in body
    assert "hidden" in body.split('id="t-opts-rag"', 1)[1].split(">", 1)[0]  # options start hidden
