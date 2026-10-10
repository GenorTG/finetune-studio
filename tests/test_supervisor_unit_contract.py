"""The systemd unit written by install-service.sh runs the supervisor, not a bare uvicorn."""
from __future__ import annotations

from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent / "install-service.sh").read_text(encoding="utf-8")


def test_unit_execstart_is_the_supervisor_with_control_group_kill():
    exec_lines = [ln for ln in SCRIPT.splitlines() if ln.startswith("ExecStart=")]
    assert len(exec_lines) == 1
    assert "-m finetune_studio.supervisor" in exec_lines[0] and "uvicorn" not in exec_lines[0]
    assert "KillMode=control-group" in SCRIPT


def test_installer_waits_on_the_cheap_health_route():
    assert "/api/health" in SCRIPT
