"""Tests for project settings page + WebUI log-tail API."""
from __future__ import annotations

from pathlib import Path

from finetune_studio.webui.routes import project_settings as ps


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
    assert 'id="log-live-badge"' in body
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
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.systemd_unit_active",
        lambda unit=ps.SYSTEMD_UNIT: False,
    )
    r = client.get(f"/api/projects/{pid}/logs?lines=50")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "lines" in data and isinstance(data["lines"], list)
    assert data["path"] == str(log)
    assert "updated_at" in data and isinstance(data["updated_at"], str)
    assert data["lines"] == ["line-a", "line-b", "line-c"]
    assert data["live"] is True
    assert data["stale"] is False
    assert data["source"] == "file"


def test_logs_missing_file_fallback(client, tmp_path: Path, monkeypatch) -> None:
    pid = _project(client)
    missing = tmp_path / "nope.log"
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.LOG_CANDIDATES",
        (str(missing),),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.systemd_unit_active",
        lambda unit=ps.SYSTEMD_UNIT: False,
    )
    r = client.get(f"/api/projects/{pid}/logs?lines=50")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["path"] == str(missing)
    assert data["live"] is False
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
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.systemd_unit_active",
        lambda unit=ps.SYSTEMD_UNIT: False,
    )
    r = client.get(f"/api/projects/{pid}/logs?lines=30")
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["lines"]) <= 30
    assert len(data["lines"]) == 30
    assert data["lines"][0] == "L70"
    assert data["lines"][-1] == "L99"


def test_logs_prefer_systemd_journal_when_active(tmp_path: Path, monkeypatch) -> None:
    stale = tmp_path / "uvicorn.log"
    stale.write_text("OLD BARE PID LOG\n", encoding="utf-8")
    # Force stale mtime age via monkeypatch of age helper.
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.LOG_CANDIDATES",
        (str(stale),),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.journal_tail",
        lambda n, unit=ps.SYSTEMD_UNIT: ["systemd-line-1", "systemd-line-2"],
    )
    payload = ps.build_logs_payload(40, prefer_systemd=True)
    assert payload["live"] is True
    assert payload["stale"] is False
    assert payload["source"] == "systemd"
    assert "journalctl" in payload["path"]
    assert payload["lines"] == ["systemd-line-1", "systemd-line-2"]
    # Must not leak the stale bare-file content when journal is available.
    assert "OLD BARE PID LOG" not in "\n".join(payload["lines"])


def test_logs_stale_file_when_systemd_active_but_journal_missing(
    tmp_path: Path, monkeypatch
) -> None:
    stale = tmp_path / "uvicorn.log"
    stale.write_text("stale-bare\n", encoding="utf-8")
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.LOG_CANDIDATES",
        (str(stale),),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.journal_tail",
        lambda n, unit=ps.SYSTEMD_UNIT: None,
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_settings.file_mtime_age_seconds",
        lambda path: 9999.0,
    )
    payload = ps.build_logs_payload(20, prefer_systemd=True)
    assert payload["live"] is False
    assert payload["stale"] is True
    assert payload["source"] == "unavailable"
    joined = "\n".join(payload["lines"])
    assert "not claiming live" in joined.lower() or "stale" in joined.lower()
    assert "stale-bare" not in joined  # do not present stale body as live tail
