"""Regression: Testing Run button stays disabled until a suite is picked."""

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
    assert "disabled" in html
    assert "Pick a suite to enable Run" in html
    assert "runBtn.disabled = true" in html
    assert "runBtn.disabled = false" in html
    # Primary results stay a table; raw JSON only under Debug.
    assert "case-results-table" in html
    assert "Debug JSON" in html
    assert "JSON.stringify(scores" not in html.split("Debug JSON")[0]


def test_testing_page_run_starts_disabled(client) -> None:
    pid = client.post(
        "/api/projects", json={"name": "Testing Run Gate"}
    ).json()["id"]
    body = client.get(f"/projects/{pid}/testing").text
    assert 'id="t-run-btn"' in body
    assert "disabled" in body
    assert "Pick a suite to enable Run" in body or "No suites" in body


def test_testing_template_rag_grounded_ui_contract() -> None:
    """Project Testing exposes Run with RAG against /api/testing/run-rag-suite."""
    html = _TESTING.read_text(encoding="utf-8")
    assert 'id="t-rag-card"' in html
    assert 'id="t-run-rag-btn"' in html
    assert "Run with RAG" in html
    assert 'id="t-rag-corpus"' in html
    assert 'id="t-rag-topk"' in html
    assert 'id="t-rag-status"' in html
    assert "runRagSuite" in html
    assert "/api/testing/run-rag-suite" in html
    assert "setRagRunEnabled" in html
    # Gate: RAG button starts disabled; enabled with suite via onSuitePicked.
    rag_tag = html.split('id="t-run-rag-btn"', 1)[1].split(">", 1)[0]
    assert "disabled" in rag_tag
    assert "Pick a suite above to enable RAG-grounded Run" in html
    # Results surface retrieval summary when RAG mode returns metrics.
    assert "t-rag-retrieval-summary" in html
    assert "rag-grounded" in html
    assert "retrieval_hit" in html


def test_testing_page_renders_rag_controls(client: TestClient) -> None:
    pid = client.post(
        "/api/projects", json={"name": "Testing RAG UI"}
    ).json()["id"]
    body = client.get(f"/projects/{pid}/testing").text
    assert 'id="t-rag-card"' in body
    assert 'id="t-run-rag-btn"' in body
    assert "Run with RAG" in body
    assert 'id="t-rag-corpus"' in body
    assert "disabled" in body.split('id="t-run-rag-btn"', 1)[1].split(">", 1)[0]
