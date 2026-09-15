"""Regression: Testing Run button stays disabled until a suite is picked."""

from __future__ import annotations

from pathlib import Path

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
