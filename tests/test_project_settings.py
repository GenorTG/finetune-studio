"""Tests for project settings page + WebUI log-tail API."""
from __future__ import annotations

from pathlib import Path


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Settings Log Tail Test"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def test_settings_page_renders_log_tail_card(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/settings")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="log-tail-card"' in body
    assert "WebUI log tail" in body
    assert 'id="log-tail"' in body
    assert "flRefreshLogs" in body
    assert "flToggleAutoRefresh" in body
    assert "Auto-refresh" in body
    assert "Refresh" in body
    # Backwards compat: project settings form still present.
    assert 'id="settings-form"' in body
    assert 'name="base_model"' in body
    assert 'name="system_prompt"' in body


def test_logs_endpoint_shape(client, tmp_path: Path, monkeypatch) -> None:
    pid = _project(client)
    log = tmp_path / "uvicorn.log"
    log.write_text("line-a\nline-b\nline-c\n", encoding="utf-8")
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.LOG_CANDIDATES",
        (str(log),),
    )
    r = client.get(f"/api/projects/{pid}/logs?lines=50")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "lines" in data and isinstance(data["lines"], list)
    assert data["path"] == str(log)
    assert "updated_at" in data and isinstance(data["updated_at"], str)
    assert data["lines"] == ["line-a", "line-b", "line-c"]


def test_logs_missing_file_fallback(client, tmp_path: Path, monkeypatch) -> None:
    pid = _project(client)
    missing = tmp_path / "nope.log"
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.LOG_CANDIDATES",
        (str(missing),),
    )
    r = client.get(f"/api/projects/{pid}/logs?lines=50")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["path"] == str(missing)
    assert len(data["lines"]) == 1
    assert "(no log file found at" in data["lines"][0]
    assert str(missing) in data["lines"][0]


def test_logs_lines_param_respected(
    client, tmp_path: Path, monkeypatch
) -> None:
    pid = _project(client)
    log = tmp_path / "uvicorn.log"
    log.write_text("\n".join(f"L{i}" for i in range(100)) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.LOG_CANDIDATES",
        (str(log),),
    )
    r = client.get(f"/api/projects/{pid}/logs?lines=30")
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["lines"]) <= 30
    assert len(data["lines"]) == 30
    assert data["lines"][0] == "L70"
    assert data["lines"][-1] == "L99"
